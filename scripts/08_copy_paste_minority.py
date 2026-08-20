#!/usr/bin/env python3
"""
08_copy_paste_minority.py  (BARU)
=================================
Copy-paste augmentation offline untuk kelas minoritas pada dataset
deteksi (format YOLO bbox).

KENAPA SCRIPT INI ADA
---------------------
Ultralytics punya parameter `copy_paste`, tapi itu HANYA bekerja untuk
task segmentasi karena butuh mask poligon untuk memotong objek secara
presisi. Dataset kamu format bbox, jadi parameter itu tidak berpengaruh
apa-apa kalau dinyalakan.

Sementara itu, `tangga` dan `tiang` adalah kelas minoritas yang justru
penting. Augmentasi fotometrik (bayangan, malam, hujan) memvariasikan
contoh yang SUDAH ADA, tapi tidak menambah JUMLAH contoh. Untuk kelas
yang cuma punya beberapa ratus instance, yang dibutuhkan adalah contoh
tambahan dalam konteks baru.

Copy-paste berbasis bbox lebih kasar daripada berbasis mask (potongannya
persegi, jadi ikut membawa sedikit background), tapi literatur konsisten
menunjukkan tetap efektif untuk kelas rare. Untuk mengurangi artefak,
script ini melakukan:

  - Blending tepi (bukan tempel mentah dengan garis potong tajam)
  - Pencocokan statistik warna sederhana antara patch dan gambar tujuan,
    supaya patch dari foto siang tidak menempel mencolok di foto sore
  - Penolakan penempatan yang tumpang tindih berat dengan objek lain
  - Penempatan pada zona vertikal yang masuk akal (tangga & tiang tidak
    muncul melayang di langit)

PERINGATAN JUJUR
----------------
Copy-paste bbox bisa menurunkan precision kalau berlebihan, karena model
belajar artefak tempelan. Mulai dari `--target-multiplier 2.0` dan ukur.
Kalau precision kelas itu turun tajam sementara recall tidak naik banyak,
kurangi atau matikan.

SELALU periksa hasilnya secara visual lewat --preview sebelum training.

Usage:
    python scripts/08_copy_paste_minority.py \
        --dataset /path/dataset_master_yolo \
        --output /path/dataset_master_yolo_cp \
        --minority-classes tangga tiang got_terbuka \
        --target-multiplier 2.5 \
        --preview
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Zona vertikal yang masuk akal untuk tiap kelas, dinyatakan sebagai
# rentang posisi TENGAH bbox relatif terhadap tinggi gambar.
# Kamera dada/kepala pengguna: langit di atas, tanah di bawah.
VERTICAL_ZONES = {
    "lubang": (0.45, 0.95),        # di tanah
    "got_terbuka": (0.45, 0.95),   # di tanah
    "tangga": (0.35, 0.90),        # tanah sampai agak ke atas
    "orang": (0.20, 0.85),
    "motor": (0.30, 0.90),
    "tiang": (0.10, 0.85),         # tiang tinggi, bisa dari atas ke bawah
}


# ─── IO ────────────────────────────────────────────────────────────────────────

def load_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Baca file label YOLO -> list (cls, cx, cy, w, h) ternormalisasi."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        out.append((cls, cx, cy, w, h))
    return out


def save_labels(path: Path, labels: list) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
             for c, cx, cy, w, h in labels]
    path.write_text("\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8")


def yolo_to_xyxy(cx, cy, w, h, W, H):
    return (int((cx - w / 2) * W), int((cy - h / 2) * H),
            int((cx + w / 2) * W), int((cy + h / 2) * H))


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    return (((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H,
            (x2 - x1) / W, (y2 - y1) / H)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


# ─── Bank patch ────────────────────────────────────────────────────────────────

def build_patch_bank(images_dir: Path, labels_dir: Path,
                     minority_ids: set[int], min_size: int = 28,
                     max_per_class: int = 400) -> dict[int, list]:
    """
    Kumpulkan potongan gambar untuk tiap kelas minoritas.
    Patch dipilih yang cukup besar supaya kualitasnya layak ditempel.
    """
    bank: dict[int, list] = defaultdict(list)

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        labels = load_labels(labels_dir / f"{img_path.stem}.txt")
        if not labels:
            continue
        if not any(c in minority_ids for c, *_ in labels):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        for cls, cx, cy, w, h in labels:
            if cls not in minority_ids:
                continue
            if len(bank[cls]) >= max_per_class:
                continue
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < min_size or y2 - y1 < min_size:
                continue
            patch = img[y1:y2, x1:x2].copy()
            bank[cls].append(patch)

    return bank


def match_color_stats(patch: np.ndarray, target: np.ndarray,
                      strength: float = 0.6) -> np.ndarray:
    """
    Sesuaikan statistik warna patch agar mendekati gambar tujuan.

    Tanpa ini, patch dari foto siang cerah yang ditempel ke foto sore
    kelihatan seperti stiker, dan model bisa belajar mendeteksi "stiker"
    alih-alih objeknya. Dilakukan di ruang LAB karena di situ perubahan
    pencahayaan lebih terpisah dari perubahan warna.
    """
    p_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        p_mean, p_std = p_lab[..., ch].mean(), p_lab[..., ch].std() + 1e-6
        t_mean, t_std = t_lab[..., ch].mean(), t_lab[..., ch].std() + 1e-6
        adjusted = (p_lab[..., ch] - p_mean) * (t_std / p_std) + t_mean
        p_lab[..., ch] = p_lab[..., ch] * (1 - strength) + adjusted * strength

    p_lab = np.clip(p_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(p_lab, cv2.COLOR_LAB2BGR)


def paste_patch(img: np.ndarray, labels: list, patch: np.ndarray,
                cls_id: int, cls_name: str, rng: random.Random,
                max_iou: float = 0.25,
                scale_range=(0.6, 1.5)) -> bool:
    """
    Tempel satu patch ke gambar. Return True kalau berhasil.
    `labels` dimodifikasi in-place.
    """
    H, W = img.shape[:2]

    scale = rng.uniform(*scale_range)
    ph = max(16, int(patch.shape[0] * scale))
    pw = max(16, int(patch.shape[1] * scale))
    if ph >= H * 0.8 or pw >= W * 0.8:
        return False

    patch_r = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)

    zone = VERTICAL_ZONES.get(cls_name, (0.2, 0.9))
    existing = [yolo_to_xyxy(cx, cy, w, h, W, H) for _, cx, cy, w, h in labels]

    for _ in range(30):
        cy_rel = rng.uniform(*zone)
        cy_px = int(cy_rel * H)
        y1 = cy_px - ph // 2
        x1 = rng.randint(0, max(0, W - pw))
        y1 = max(0, min(y1, H - ph))
        x2, y2 = x1 + pw, y1 + ph

        if any(iou((x1, y1, x2, y2), e) > max_iou for e in existing):
            continue

        region = img[y1:y2, x1:x2]
        patch_c = match_color_stats(patch_r, region)

        # Blending tepi: alpha 1 di tengah, meluruh di pinggir
        alpha = np.ones((ph, pw), dtype=np.float32)
        feather = max(2, min(ph, pw) // 10)
        for i in range(feather):
            v = (i + 1) / (feather + 1)
            alpha[i, :] = np.minimum(alpha[i, :], v)
            alpha[-(i + 1), :] = np.minimum(alpha[-(i + 1), :], v)
            alpha[:, i] = np.minimum(alpha[:, i], v)
            alpha[:, -(i + 1)] = np.minimum(alpha[:, -(i + 1)], v)
        alpha = alpha[..., None]

        img[y1:y2, x1:x2] = (
            region.astype(np.float32) * (1 - alpha)
            + patch_c.astype(np.float32) * alpha
        ).astype(np.uint8)

        labels.append((cls_id, *xyxy_to_yolo(x1, y1, x2, y2, W, H)))
        return True

    return False


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Copy-paste offline untuk kelas minoritas (YOLO bbox)"
    )
    ap.add_argument("--dataset", required=True,
                    help="Folder dataset YOLO (punya images/train, labels/train)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-yaml", default=None,
                    help="data.yaml untuk membaca nama kelas")
    ap.add_argument("--minority-classes", nargs="+",
                    default=["tangga", "tiang", "got_terbuka"])
    ap.add_argument("--target-multiplier", type=float, default=2.5,
                    help="Target lipat ganda instance kelas minoritas")
    ap.add_argument("--max-paste-per-image", type=int, default=2)
    ap.add_argument("--paste-prob", type=float, default=0.5,
                    help="Porsi gambar yang mendapat tempelan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview", action="store_true",
                    help="Simpan 12 contoh hasil untuk diperiksa mata")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    ds = Path(args.dataset)
    out = Path(args.output)

    img_dir = ds / "images" / "train"
    lbl_dir = ds / "labels" / "train"
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Struktur dataset tidak sesuai. Dicari:\n  {img_dir}\n  {lbl_dir}")
        sys.exit(1)

    # ── Nama kelas ──
    names = {}
    yaml_path = Path(args.data_yaml) if args.data_yaml else ds / "data.yaml"
    if yaml_path.exists():
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        raw = cfg.get("names", {})
        names = ({i: n for i, n in enumerate(raw)} if isinstance(raw, list)
                 else {int(k): v for k, v in raw.items()})
    if not names:
        names = {0: "lubang", 1: "got_terbuka", 2: "tangga",
                 3: "orang", 4: "motor", 5: "tiang"}
        print(f"  data.yaml tidak ditemukan, memakai mapping default: {names}")

    name_to_id = {v: k for k, v in names.items()}
    minority_ids = {name_to_id[n] for n in args.minority_classes
                    if n in name_to_id}
    if not minority_ids:
        print(f"Kelas minoritas tidak dikenali. Tersedia: {list(names.values())}")
        sys.exit(1)

    print("=" * 66)
    print("  COPY-PASTE KELAS MINORITAS")
    print("=" * 66)
    print(f"  Dataset  : {ds}")
    print(f"  Output   : {out}")
    print(f"  Minoritas: {[names[i] for i in sorted(minority_ids)]}")

    # ── Distribusi awal ──
    before: Counter = Counter()
    all_images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in IMG_EXTENSIONS)
    for p in all_images:
        for c, *_ in load_labels(lbl_dir / f"{p.stem}.txt"):
            before[c] += 1

    print(f"\n  {'Kelas':<14} {'sebelum':>10}")
    print(f"  {'-' * 26}")
    for i in sorted(names):
        print(f"  {names[i]:<14} {before.get(i, 0):>10}")

    # ── Bank patch ──
    print("\nMengumpulkan patch kelas minoritas...")
    bank = build_patch_bank(img_dir, lbl_dir, minority_ids)
    for cid, patches in sorted(bank.items()):
        print(f"  {names[cid]:<14}: {len(patches)} patch")
    if not any(bank.values()):
        print("  Tidak ada patch yang layak. Cek apakah kelas minoritas "
              "punya bbox yang cukup besar (minimal 28px).")
        sys.exit(1)

    # ── Berapa banyak yang perlu ditempel ──
    need = {}
    for cid in minority_ids:
        current = before.get(cid, 0)
        target = int(current * args.target_multiplier)
        need[cid] = max(0, target - current)
    print(f"\n  Instance tambahan yang ditargetkan: "
          f"{ {names[c]: n for c, n in need.items()} }")

    # ── Salin dataset ──
    print("\nMenyalin dataset...")
    if out.exists():
        shutil.rmtree(out)
    out_img = out / "images" / "train"
    out_lbl = out / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for split in ("valid", "val", "test"):
        src_i = ds / "images" / split
        src_l = ds / "labels" / split
        if src_i.exists():
            # Split evaluasi TIDAK diaugmentasi. Metrik harus mencerminkan
            # distribusi dunia nyata, bukan dunia hasil tempelan.
            shutil.copytree(src_i, out / "images" / split)
            if src_l.exists():
                shutil.copytree(src_l, out / "labels" / split)

    # ── Proses ──
    print("\nMenempel patch...")
    pasted: Counter = Counter()
    preview_saved = []
    remaining = dict(need)

    order = list(all_images)
    rng.shuffle(order)

    for img_path in order:
        labels = load_labels(lbl_dir / f"{img_path.stem}.txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        do_paste = (rng.random() < args.paste_prob
                    and any(v > 0 for v in remaining.values()))

        modified = False
        if do_paste:
            n_paste = rng.randint(1, args.max_paste_per_image)
            for _ in range(n_paste):
                candidates = [c for c, v in remaining.items()
                              if v > 0 and bank.get(c)]
                if not candidates:
                    break
                cid = rng.choice(candidates)
                patch = rng.choice(bank[cid])
                if paste_patch(img, labels, patch, cid, names[cid], rng):
                    pasted[cid] += 1
                    remaining[cid] -= 1
                    modified = True

        dst_img = out_img / img_path.name
        dst_lbl = out_lbl / f"{img_path.stem}.txt"

        if modified:
            cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if args.preview and len(preview_saved) < 12:
                vis = img.copy()
                H, W = vis.shape[:2]
                for c, cx, cy, w, h in labels:
                    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
                    color = (0, 0, 255) if c in minority_ids else (0, 200, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, names[c], (x1, max(14, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                preview_saved.append(vis)
        else:
            try:
                dst_img.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst_img)

        save_labels(dst_lbl, labels)

    # ── Hasil ──
    after: Counter = Counter()
    for p in sorted(out_lbl.glob("*.txt")):
        for c, *_ in load_labels(p):
            after[c] += 1

    print("\n" + "=" * 66)
    print("  HASIL")
    print("=" * 66)
    print(f"  {'Kelas':<14} {'sebelum':>10} {'sesudah':>10} {'perubahan':>11}")
    print(f"  {'-' * 48}")
    for i in sorted(names):
        b, a = before.get(i, 0), after.get(i, 0)
        delta = f"+{a - b}" if a > b else str(a - b)
        print(f"  {names[i]:<14} {b:>10} {a:>10} {delta:>11}")

    counts = [after.get(i, 0) for i in names if after.get(i, 0) > 0]
    if counts:
        print(f"\n  Rasio ketidakseimbangan: "
              f"{max(counts) / min(counts):.1f}x "
              f"(sebelumnya "
              f"{max(before.values()) / max(1, min(v for v in before.values() if v > 0)):.1f}x)")

    # ── data.yaml baru ──
    new_cfg = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/valid" if (out / "images" / "valid").exists()
               else "images/val",
        "nc": len(names),
        "names": {int(k): v for k, v in names.items()},
    }
    if (out / "images" / "test").exists():
        new_cfg["test"] = "images/test"
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)

    with open(out / "copy_paste_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "before": {names[i]: before.get(i, 0) for i in names},
            "after": {names[i]: after.get(i, 0) for i in names},
            "pasted": {names[i]: pasted.get(i, 0) for i in pasted},
            "args": vars(args),
        }, f, indent=2)

    # ── Preview ──
    if args.preview and preview_saved:
        tiles = [cv2.resize(v, (320, 240)) for v in preview_saved[:12]]
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)
                if len(tiles[i:i + 4]) == 4]
        if rows:
            grid = np.vstack(rows)
            pv = out / "copy_paste_preview.jpg"
            cv2.imwrite(str(pv), grid)
            print(f"\n  Preview: {pv}")
            print("  PERIKSA INI SEBELUM TRAINING. Kalau tempelannya "
                  "kelihatan seperti stiker yang jelas, model akan belajar "
                  "mendeteksi stiker, bukan objeknya. Turunkan "
                  "--target-multiplier kalau begitu.")

    print(f"\n  Dataset baru: {out.resolve()}")
    print(f"  Pakai dengan: python scripts/05_train_yolo.py "
          f"--data {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
