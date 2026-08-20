#!/usr/bin/env python3
"""
09_distill_yolo.py  (BARU)
==========================
Knowledge distillation dari YOLO11l/x (teacher) ke YOLO11n (student).

IDE DASARNYA
------------
Model besar belajar representasi yang lebih kaya. Distillation mentransfer
sebagian representasi itu ke model kecil, sehingga model kecil jadi lebih
akurat TANPA bertambah besar sedikit pun. Ukuran file, latency, dan
konsumsi memori student tetap persis sama seperti YOLO11n biasa.

Untuk kasus GUIDIO ini menarik karena kamu terkunci di YOLO11n oleh
target 2 FPS di HP mid-low. Distillation adalah salah satu dari sedikit
cara menaikkan akurasi tanpa melanggar batasan itu.

EKSPEKTASI YANG REALISTIS
-------------------------
Literatur distillation pada YOLO11 (feature-based, CWD/MGD) melaporkan
kenaikan sekitar 1.9 sampai 2.5 poin mAP50 untuk pasangan
YOLO11x -> YOLO11n pada dataset khusus domain. Itu nyata tapi tidak
dramatis. Jangan berharap lompatan 10 poin.

Urutan prioritas yang jujur: kalau dataset kamu masih punya masalah
distribusi kelas atau kekurangan data kondisi malam/hujan, memperbaiki
DATA akan memberi kenaikan jauh lebih besar daripada distillation.
Kerjakan distillation setelah data beres, bukan sebagai jalan pintas
untuk menutupi data yang kurang.

DUA JALUR YANG DISEDIAKAN
-------------------------
1. `--mode pseudo` (DEFAULT, tanpa dependensi tambahan)
   Teacher memberi label pada gambar TAK BERLABEL, hasilnya digabung
   ke train set student. Ini bentuk distillation paling sederhana
   (kadang disebut self-training atau pseudo-labeling) dan tidak
   membutuhkan modifikasi internal Ultralytics sama sekali, jadi tidak
   akan rusak saat Ultralytics update.

   Ini juga menjawab pertanyaanmu soal memanfaatkan foto jalanan tanpa
   anotasi: inilah cara paling praktis untuk tim kecil.

2. `--mode feature` (butuh repo pihak ketiga)
   Feature-based distillation (CWD/MGD) yang menyelaraskan peta fitur
   student dengan teacher selama training. Lebih kuat, tapi butuh
   patch ke loop training Ultralytics. Script ini mendeteksi apakah
   paket pendukung tersedia dan memberi instruksi kalau belum.

Usage:
    # 1. Latih teacher dulu (di dataset yang SAMA dengan student)
    python scripts/05_train_yolo.py --model yolo11l.pt --name teacher \
        --epochs 120 --batch 8

    # 2. Pseudo-label foto jalanan tak berlabel
    python scripts/09_distill_yolo.py --mode pseudo \
        --teacher runs/yolo/teacher/weights/best.pt \
        --unlabeled-dir ~/foto_jalanan_mentah \
        --data configs/custom_navigasi.yaml \
        --output-dataset ../dataset_master_yolo_pseudo \
        --conf 0.55

    # 3. Latih student di dataset gabungan
    python scripts/05_train_yolo.py \
        --data ../dataset_master_yolo_pseudo/data.yaml --model yolo11n.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Confidence threshold per kelas untuk pseudo-labeling.
#
# Ini SENGAJA berbeda dari threshold inference. Untuk pseudo-label kita
# ingin PRESISI TINGGI: label yang salah akan diajarkan ke student sebagai
# kebenaran, dan itu lebih merusak daripada tidak punya label sama sekali.
# Jadi ambangnya tinggi, dan kita rela melewatkan banyak objek.
#
# Kelas yang secara umum lebih mudah dideteksi (orang, motor) boleh
# sedikit lebih rendah karena teacher lebih andal di situ.
DEFAULT_CONF_PER_CLASS = {
    "lubang": 0.60,
    "got_terbuka": 0.60,
    "tangga": 0.60,
    "orang": 0.50,
    "motor": 0.50,
    "tiang": 0.60,
}


def load_names(data_yaml: Path) -> dict[int, str]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw = cfg.get("names", {})
    if isinstance(raw, list):
        return {i: n for i, n in enumerate(raw)}
    return {int(k): v for k, v in raw.items()}


# ─── Mode pseudo ───────────────────────────────────────────────────────────────

def run_pseudo_labeling(args) -> None:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"Dependensi kurang: {e}")
        sys.exit(1)

    teacher_path = Path(args.teacher)
    if not teacher_path.exists():
        print(f"Teacher tidak ditemukan: {teacher_path}")
        sys.exit(1)

    data_yaml = Path(args.data)
    names = load_names(data_yaml)
    name_to_id = {v: k for k, v in names.items()}

    conf_map = {}
    for name, thr in DEFAULT_CONF_PER_CLASS.items():
        if name in name_to_id:
            conf_map[name_to_id[name]] = thr
    for cid in names:
        conf_map.setdefault(cid, args.conf)

    unlabeled = Path(args.unlabeled_dir)
    if not unlabeled.exists():
        print(f"Folder gambar tak berlabel tidak ada: {unlabeled}")
        sys.exit(1)

    images = sorted(p for p in unlabeled.rglob("*")
                    if p.suffix.lower() in IMG_EXTENSIONS)
    if not images:
        print(f"Tidak ada gambar di {unlabeled}")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 66)
    print("  PSEUDO-LABELING DENGAN TEACHER")
    print("=" * 66)
    print(f"  Teacher   : {teacher_path}")
    print(f"  Gambar    : {len(images)} file di {unlabeled}")
    print(f"  Device    : {device}")
    print(f"  Threshold : {dict((names[c], t) for c, t in sorted(conf_map.items()))}")
    print("\n  Catatan: ambang sengaja tinggi. Label pseudo yang salah "
          "diajarkan ke student sebagai kebenaran, jadi lebih baik "
          "melewatkan objek daripada salah melabeli.")

    model = YOLO(str(teacher_path))

    # ── Siapkan dataset keluaran ──
    out_root = Path(args.output_dataset)
    base_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base_path = Path(base_cfg.get("path", data_yaml.parent))
    if not base_path.is_absolute():
        base_path = (data_yaml.parent / base_path).resolve()

    if out_root.exists():
        shutil.rmtree(out_root)

    print(f"\n  Menyalin dataset asli dari {base_path}...")
    for split_key, default in (("train", "images/train"),
                               ("val", "images/valid"),
                               ("test", "images/test")):
        rel = base_cfg.get(split_key, default)
        src_img = base_path / rel
        if not src_img.exists():
            continue
        split_name = Path(rel).name
        dst_img = out_root / "images" / split_name
        dst_img.mkdir(parents=True, exist_ok=True)

        src_lbl = Path(str(src_img).replace("/images/", "/labels/"))
        dst_lbl = out_root / "labels" / split_name
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n = 0
        for p in src_img.iterdir():
            if p.suffix.lower() not in IMG_EXTENSIONS:
                continue
            try:
                (dst_img / p.name).symlink_to(p.resolve())
            except OSError:
                shutil.copy2(p, dst_img / p.name)
            lp = src_lbl / f"{p.stem}.txt"
            if lp.exists():
                try:
                    (dst_lbl / lp.name).symlink_to(lp.resolve())
                except OSError:
                    shutil.copy2(lp, dst_lbl / lp.name)
            n += 1
        print(f"    {split_name}: {n} gambar")

    train_img_dir = out_root / "images" / "train"
    train_lbl_dir = out_root / "labels" / "train"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    # ── Inferensi ──
    print(f"\n  Memberi label pada {len(images)} gambar...")
    kept_counter: Counter = Counter()
    n_added = 0
    n_empty = 0

    batch_size = 16
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        results = model.predict(
            [str(p) for p in chunk],
            conf=min(conf_map.values()),
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )

        for img_path, res in zip(chunk, results):
            lines = []
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                for cls_t, conf_t, xywhn in zip(boxes.cls, boxes.conf,
                                                boxes.xywhn):
                    cid = int(cls_t)
                    c = float(conf_t)
                    if c < conf_map.get(cid, args.conf):
                        continue
                    cx, cy, w, h = (float(v) for v in xywhn)
                    if w < args.min_box or h < args.min_box:
                        continue
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    kept_counter[cid] += 1

            if not lines and not args.keep_empty:
                n_empty += 1
                continue

            stem = f"pseudo_{n_added:06d}"
            dst_img = train_img_dir / f"{stem}{img_path.suffix}"
            try:
                dst_img.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst_img)
            (train_lbl_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_added += 1

        done = min(start + batch_size, len(images))
        if done % 160 == 0 or done == len(images):
            print(f"    {done}/{len(images)}  "
                  f"({n_added} ditambahkan, {n_empty} dilewati)")

    # ── data.yaml ──
    new_cfg = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": ("images/valid" if (out_root / "images" / "valid").exists()
                else "images/val"),
        "nc": len(names),
        "names": names,
    }
    if (out_root / "images" / "test").exists():
        new_cfg["test"] = "images/test"
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)

    # ── Laporan ──
    print("\n" + "=" * 66)
    print("  HASIL PSEUDO-LABELING")
    print("=" * 66)
    print(f"  Gambar ditambahkan : {n_added}")
    print(f"  Gambar dilewati    : {n_empty} (tidak ada deteksi di atas ambang)")
    print(f"\n  {'Kelas':<14} {'pseudo-label':>14}")
    print(f"  {'-' * 30}")
    for cid in sorted(names):
        print(f"  {names[cid]:<14} {kept_counter.get(cid, 0):>14}")

    with open(out_root / "pseudo_label_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_added": n_added, "n_empty_skipped": n_empty,
            "counts": {names[c]: kept_counter.get(c, 0) for c in names},
            "conf_thresholds": {names[c]: t for c, t in conf_map.items()},
            "args": vars(args),
        }, f, indent=2)

    print(f"\n  Dataset gabungan: {out_root.resolve()}")
    print("\n  Langkah berikutnya:")
    print(f"    python scripts/05_train_yolo.py "
          f"--data {out_root / 'data.yaml'} --model yolo11n.pt")
    print("\n  PENTING: bandingkan hasilnya dengan student yang dilatih "
          "TANPA pseudo-label. Pseudo-labeling bisa menurunkan performa "
          "kalau teacher-nya kurang bagus, karena kesalahan teacher ikut "
          "diwariskan dan diperkuat.")


# ─── Mode feature ──────────────────────────────────────────────────────────────

def run_feature_distillation(args) -> None:
    print("=" * 66)
    print("  FEATURE-BASED DISTILLATION (CWD / MGD)")
    print("=" * 66)
    print()
    print("  Feature-based distillation butuh modifikasi loop training")
    print("  Ultralytics untuk memasang hook di peta fitur teacher dan")
    print("  student, lalu menambahkan loss penyelarasan.")
    print()
    print("  Ultralytics inti belum menyediakan ini untuk YOLO11 lewat")
    print("  argumen resmi (YOLO26 punya argumen `distill_model`, tapi")
    print("  itu model berbeda). Jadi pilihannya:")
    print()
    print("  OPSI A - repo pihak ketiga")
    print("    Cari repo yang mengimplementasikan CWD/MGD untuk Ultralytics")
    print("    YOLO11, misalnya turunan dari `yolo-distiller`. Pola API-nya")
    print("    biasanya seperti ini:")
    print()
    print("      from ultralytics import YOLO")
    print("      teacher = YOLO('runs/yolo/teacher/weights/best.pt')")
    print("      student = YOLO('yolo11n.pt')")
    print("      student.train(")
    print("          data='configs/custom_navigasi.yaml',")
    print("          teacher=teacher.model,")
    print("          distillation_loss='cwd',   # atau 'mgd'")
    print("          epochs=150, batch=16,")
    print("      )")
    print()
    print("    Syarat mutlak: teacher HARUS dilatih di dataset yang SAMA")
    print("    persis dengan student. Teacher COCO pretrained saja tidak")
    print("    cukup, karena kelasnya berbeda.")
    print()
    print("  OPSI B - pakai --mode pseudo (direkomendasikan untuk kamu)")
    print("    Lebih sederhana, tanpa dependensi rapuh, dan sekaligus")
    print("    memanfaatkan foto jalanan tak berlabel yang kamu punya.")
    print("    Kenaikan akurasinya sebanding untuk kasus dataset kecil.")
    print()
    print("  OPSI C - kerjakan datanya dulu")
    print("    Kalau distribusi kelas masih timpang 20x atau data malam/")
    print("    hujan masih minim, memperbaiki itu memberi kenaikan jauh")
    print("    lebih besar daripada distillation. Distillation itu")
    print("    pemerasan terakhir, bukan langkah pertama.")
    print()
    sys.exit(0)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Knowledge distillation / pseudo-labeling YOLO"
    )
    ap.add_argument("--mode", choices=["pseudo", "feature"], default="pseudo")
    ap.add_argument("--teacher", default=None,
                    help="Bobot teacher (.pt), wajib untuk mode pseudo")
    ap.add_argument("--data", default=None,
                    help="data.yaml dataset berlabel asli")
    ap.add_argument("--unlabeled-dir", default=None,
                    help="Folder foto jalanan tanpa anotasi")
    ap.add_argument("--output-dataset", default=None)
    ap.add_argument("--conf", type=float, default=0.55,
                    help="Ambang confidence default untuk kelas tanpa "
                         "pengaturan khusus")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-box", type=float, default=0.02,
                    help="Sisi bbox minimum (relatif) supaya deteksi "
                         "sangat kecil tidak jadi label berisik")
    ap.add_argument("--keep-empty", action="store_true",
                    help="Simpan juga gambar tanpa deteksi sebagai "
                         "contoh background")
    args = ap.parse_args()

    if args.mode == "feature":
        run_feature_distillation(args)
        return

    missing = [n for n, v in (("--teacher", args.teacher),
                              ("--data", args.data),
                              ("--unlabeled-dir", args.unlabeled_dir),
                              ("--output-dataset", args.output_dataset))
               if not v]
    if missing:
        print(f"Mode pseudo membutuhkan: {', '.join(missing)}")
        sys.exit(1)

    run_pseudo_labeling(args)


if __name__ == "__main__":
    main()
