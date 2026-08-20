#!/usr/bin/env python3
"""
05_train_yolo.py  (REVISI)
==========================
Training YOLO11n deteksi bahaya jalanan, dioptimalkan untuk kondisi
outdoor Indonesia yang sulit.

RINGKASAN PERUBAHAN DARI VERSI LAMA

  Augmentasi
    - Injeksi augmentasi outdoor: bayangan tajam & belang daun, hujan,
      kabut, silau, simulasi malam (gamma + lampu jalan + noise + warm
      shift), permukaan basah reflektif, motion blur berarah
    - Semua transform yang di-inject bersifat image-only, jadi bounding
      box tidak mungkin rusak. Geometri tetap ditangani Ultralytics.
    - `close_mosaic` diset eksplisit; mosaic mengecilkan objek dan itu
      merugikan lubang yang dilihat dari jarak jauh

  Ketidakseimbangan kelas
    - Analisis distribusi instance sebelum training, dengan peringatan
      kalau rasionya ekstrem
    - Oversampling gambar yang mengandung kelas minoritas lewat file
      list berbobot (bukan mengutak-atik loss, yang di YOLO sering tidak
      stabil)

  Evaluasi
    - mAP saja TIDAK cukup untuk aplikasi keselamatan. Script ini
      melaporkan recall per kelas dan memisahkan kelas bahaya
      (lubang, got_terbuka, tangga) dari kelas informasional
      (orang, motor, tiang)
    - Gate rilis eksplisit: recall kelas bahaya harus lolos ambang

  Robustness operasional
    - `device` auto-detect, tidak lagi hardcode `device=0` yang bikin
      crash di mesin tanpa GPU
    - Warmup + cosine LR
    - Seed dan deterministic flag
    - Preview augmentasi otomatis sebelum training dimulai

Usage:
    python scripts/05_train_yolo.py \
        --data configs/custom_navigasi.yaml \
        --epochs 150 \
        --aug-strength medium \
        --oversample-minority

    # Tanpa augmentasi outdoor (baseline pembanding):
    python scripts/05_train_yolo.py --no-outdoor-aug --name baseline
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.yolo_aug import (  # noqa: E402
    patch_ultralytics_albumentations,
    save_augmentation_preview,
)

# Kelas yang kegagalannya berkonsekuensi keselamatan.
# Untuk kelas ini, RECALL jauh lebih penting daripada precision:
# melewatkan lubang bisa mencelakai pengguna, sementara peringatan
# palsu cuma bikin pengguna melambat sebentar.
HAZARD_CLASSES = {"lubang", "got_terbuka", "tangga"}

# Ambang gate rilis. Angka-angka ini adalah titik awal yang masuk akal
# untuk aplikasi bantu tunanetra, bukan standar industri baku.
# Sesuaikan setelah uji lapangan dengan pengguna sungguhan.
RELEASE_GATES = {
    "hazard_recall_min": 0.85,
    "hazard_map50_min": 0.55,
    "overall_map50_min": 0.60,
}


# ─── Analisis dataset ──────────────────────────────────────────────────────────

def load_data_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_dir(cfg: dict, cfg_path: Path, split_key: str) -> Path | None:
    """Resolusi path split dari data.yaml (mendukung path absolut & relatif)."""
    if split_key not in cfg:
        return None
    base = Path(cfg.get("path", cfg_path.parent))
    if not base.is_absolute():
        base = (cfg_path.parent / base).resolve()
    p = Path(cfg[split_key])
    return p if p.is_absolute() else (base / p)


def analyze_class_distribution(images_dir: Path, names: dict) -> Counter:
    """Hitung jumlah instance per kelas dari file label YOLO."""
    labels_dir = Path(str(images_dir).replace("/images", "/labels"))
    if not labels_dir.exists():
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    counter: Counter = Counter()
    if not labels_dir.exists():
        return counter

    for lbl in labels_dir.glob("*.txt"):
        try:
            for line in lbl.read_text(encoding="utf-8",
                                      errors="ignore").splitlines():
                parts = line.split()
                if parts:
                    try:
                        counter[int(parts[0])] += 1
                    except ValueError:
                        continue
        except OSError:
            continue
    return counter


def print_distribution(counter: Counter, names: dict) -> dict:
    """Cetak distribusi kelas dan kembalikan ringkasannya."""
    total = sum(counter.values())
    if total == 0:
        print("  Tidak ada label ditemukan. Periksa struktur dataset.")
        return {}

    print(f"  {'Kelas':<14} {'instance':>10} {'porsi':>8}")
    print(f"  {'-' * 34}")
    rows = {}
    for idx in sorted(names):
        n = counter.get(idx, 0)
        pct = 100.0 * n / total
        flag = ""
        if names[idx] in HAZARD_CLASSES and pct < 5:
            flag = "  <-- minoritas & kritis"
        print(f"  {names[idx]:<14} {n:>10} {pct:>7.2f}%{flag}")
        rows[names[idx]] = n

    counts = [c for c in (counter.get(i, 0) for i in names) if c > 0]
    if counts:
        ratio = max(counts) / min(counts)
        print(f"\n  Rasio ketidakseimbangan: {ratio:.1f}x")
        if ratio > 20:
            print("  Ketidakseimbangan sangat parah. Oversampling saja tidak "
                  "akan cukup. Prioritaskan menambah data untuk kelas "
                  "minoritas, atau pakai copy-paste (scripts/08).")
        elif ratio > 8:
            print("  Ketidakseimbangan cukup parah. Pakai --oversample-minority.")
    return rows


# ─── Oversampling ──────────────────────────────────────────────────────────────

def build_oversampled_dataset(images_dir: Path, names: dict,
                              counter: Counter, out_dir: Path,
                              max_repeat: int = 4) -> Path:
    """
    Bikin dataset baru berisi symlink, dengan gambar yang mengandung
    kelas minoritas diduplikasi beberapa kali.

    KENAPA SYMLINK DAN BUKAN COPY
    Dataset deteksi bisa puluhan GB. Symlink bikin oversampling nyaris
    gratis dari sisi disk. Ultralytics membaca symlink seperti file biasa.

    KENAPA OVERSAMPLING GAMBAR DAN BUKAN BOBOT LOSS
    Di YOLO family, memodifikasi bobot loss per kelas sering bikin
    training tidak stabil karena loss objectness, box, dan classification
    saling terkait lewat assigner (TaskAlignedAssigner). Menduplikasi
    gambar di level dataset jauh lebih dapat diprediksi dan tidak
    menyentuh internal Ultralytics sama sekali.

    EFEK SAMPING YANG HARUS DISADARI
    Menduplikasi gambar juga menduplikasi kelas mayoritas yang kebetulan
    ada di gambar yang sama. Jadi ini memperbaiki rasio, tapi tidak
    sesempurna copy-paste. Kalau satu gambar berisi 1 lubang dan 8 orang,
    menduplikasinya 4x menambah 4 lubang tapi juga 32 orang.
    """
    labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    if not labels_dir.exists():
        print("  Folder label tidak ditemukan, oversampling dilewati.")
        return images_dir

    total = sum(counter.values())
    if total == 0:
        return images_dir

    # Target: tiap kelas minimal mencapai porsi rata-rata
    n_classes = len([i for i in names if counter.get(i, 0) > 0])
    target_share = 1.0 / max(1, n_classes)

    repeat_for_class = {}
    for idx in names:
        n = counter.get(idx, 0)
        if n == 0:
            continue
        share = n / total
        rep = int(np.clip(round(target_share / share), 1, max_repeat))
        repeat_for_class[idx] = rep

    out_img = out_dir / "images" / "train"
    out_lbl = out_dir / "labels" / "train"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        classes_here = set()
        for line in lbl_path.read_text(encoding="utf-8",
                                       errors="ignore").splitlines():
            parts = line.split()
            if parts:
                try:
                    classes_here.add(int(parts[0]))
                except ValueError:
                    continue

        rep = max((repeat_for_class.get(c, 1) for c in classes_here),
                  default=1)

        for k in range(rep):
            stem = img_path.stem if k == 0 else f"{img_path.stem}__rep{k}"
            dst_img = out_img / f"{stem}{img_path.suffix}"
            dst_lbl = out_lbl / f"{stem}.txt"
            try:
                dst_img.symlink_to(img_path.resolve())
                dst_lbl.symlink_to(lbl_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_path, dst_lbl)
            n_written += 1

    print(f"  Oversampling: {n_written} entri dibuat di {out_dir}")
    print(f"  Faktor pengulangan per kelas: "
          f"{ {names[i]: r for i, r in repeat_for_class.items()} }")
    return out_img


# ─── Evaluasi ──────────────────────────────────────────────────────────────────

def evaluate_and_report(model, data_cfg: str, imgsz: int, names: dict,
                        device, out_dir: Path) -> dict:
    """
    Validasi dengan penekanan pada recall kelas bahaya.

    Ultralytics mengembalikan metrik per kelas lewat `ap_class_index`,
    `ap50`, `p`, `r`. Perhatikan `ap_class_index` hanya berisi kelas yang
    PUNYA instance di validation set, jadi indeksnya tidak selalu 0..n-1.
    Versi lama mengasumsikan urutan itu selalu lengkap, yang bisa
    menghasilkan pemetaan nama kelas yang salah kalau ada kelas kosong.
    """
    print("\n" + "=" * 66)
    print("  VALIDASI")
    print("=" * 66)

    metrics = model.val(data=data_cfg, imgsz=imgsz, device=device,
                        verbose=False)
    box = metrics.box

    per_class = {}
    idx_list = list(getattr(box, "ap_class_index", []))
    for i, cls_id in enumerate(idx_list):
        cls_id = int(cls_id)
        name = names.get(cls_id, f"cls{cls_id}")
        per_class[name] = {
            "map50": float(box.ap50[i]) if i < len(box.ap50) else 0.0,
            "map50_95": float(box.ap[i]) if i < len(box.ap) else 0.0,
            "precision": float(box.p[i]) if i < len(box.p) else 0.0,
            "recall": float(box.r[i]) if i < len(box.r) else 0.0,
        }

    missing = [names[i] for i in names if names[i] not in per_class]
    if missing:
        print(f"  PERINGATAN: kelas berikut tidak punya instance di "
              f"validation set: {missing}")
        print("  Metriknya tidak bisa dihitung. Kalau ini kelas bahaya, "
              "validation set kamu tidak layak dipakai menilai keselamatan.")

    print(f"\n  {'Kelas':<14} {'mAP50':>8} {'mAP50-95':>10} "
          f"{'precision':>10} {'recall':>9}")
    print(f"  {'-' * 55}")
    for name in [names[i] for i in sorted(names)]:
        if name not in per_class:
            print(f"  {name:<14} {'n/a':>8} {'n/a':>10} {'n/a':>10} {'n/a':>9}")
            continue
        m = per_class[name]
        mark = " *" if name in HAZARD_CLASSES else ""
        print(f"  {name:<14} {m['map50']:>8.4f} {m['map50_95']:>10.4f} "
              f"{m['precision']:>10.4f} {m['recall']:>9.4f}{mark}")
    print("\n  * = kelas bahaya, recall adalah metrik utamanya")

    hazard = {n: m for n, m in per_class.items() if n in HAZARD_CLASSES}
    info = {n: m for n, m in per_class.items() if n not in HAZARD_CLASSES}

    summary = {
        "overall_map50": float(box.map50),
        "overall_map50_95": float(box.map),
        "per_class": per_class,
        "hazard_recall_mean": (
            float(np.mean([m["recall"] for m in hazard.values()]))
            if hazard else 0.0
        ),
        "hazard_map50_mean": (
            float(np.mean([m["map50"] for m in hazard.values()]))
            if hazard else 0.0
        ),
        "info_recall_mean": (
            float(np.mean([m["recall"] for m in info.values()]))
            if info else 0.0
        ),
        "classes_missing_in_val": missing,
    }

    print(f"\n  mAP50 keseluruhan     : {summary['overall_map50']:.4f}")
    print(f"  mAP50-95 keseluruhan  : {summary['overall_map50_95']:.4f}")
    print(f"  Recall kelas bahaya   : {summary['hazard_recall_mean']:.4f}")
    print(f"  Recall kelas info     : {summary['info_recall_mean']:.4f}")

    # ── Gate rilis ──
    print("\n" + "=" * 66)
    print("  GATE RILIS")
    print("=" * 66)
    gates = [
        ("Recall kelas bahaya", summary["hazard_recall_mean"],
         RELEASE_GATES["hazard_recall_min"]),
        ("mAP50 kelas bahaya", summary["hazard_map50_mean"],
         RELEASE_GATES["hazard_map50_min"]),
        ("mAP50 keseluruhan", summary["overall_map50"],
         RELEASE_GATES["overall_map50_min"]),
    ]
    all_pass = True
    for label, value, threshold in gates:
        ok = value >= threshold
        all_pass = all_pass and ok
        status = "LOLOS" if ok else "GAGAL"
        print(f"  [{status}] {label:<22} {value:.4f} "
              f"(minimum {threshold:.2f})")

    summary["release_gate_passed"] = all_pass
    if not all_pass:
        print("\n  Model belum layak dipakai untuk navigasi. Langkah lanjut:")
        print("    1. Turunkan confidence threshold khusus kelas bahaya "
              "(scripts/10_tune_thresholds.py) - ini menaikkan recall "
              "dengan menukar sedikit precision")
        print("    2. Tambah data untuk kelas yang recall-nya rendah")
        print("    3. Coba knowledge distillation (scripts/09_distill_yolo.py)")
    else:
        print("\n  Semua gate lolos. Tetap lakukan uji lapangan sebelum rilis.")

    with open(out_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Training YOLO11n deteksi bahaya jalanan (revisi)"
    )
    ap.add_argument("--data", default=str(ROOT / "configs" / "custom_navigasi.yaml"))
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto",
                    help="'auto', 'cpu', '0', '0,1', dst")
    ap.add_argument("--project", default=str(ROOT / "runs" / "yolo"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--aug-strength", choices=["light", "medium", "heavy"],
                    default="medium")
    ap.add_argument("--no-outdoor-aug", action="store_true",
                    help="Matikan augmentasi outdoor (untuk baseline pembanding)")
    ap.add_argument("--close-mosaic", type=int, default=15,
                    help="Matikan mosaic N epoch terakhir")
    ap.add_argument("--oversample-minority", action="store_true")
    ap.add_argument("--max-repeat", type=int, default=4)

    ap.add_argument("--export-tflite", action="store_true",
                    help="Ekspor ke LiteRT INT8 setelah training")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"Dependensi belum lengkap: {e}")
        print("Install: pip install ultralytics torch torchvision")
        sys.exit(1)

    # ── Device ──
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cpu":
        print("PERINGATAN: training di CPU. Untuk 150 epoch ini akan makan "
              "waktu berhari-hari. Pakai GPU (Colab/Kaggle/Vast.ai).")

    if not args.name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "outdoor" if not args.no_outdoor_aug else "baseline"
        args.name = f"navigasi_yolo11n_{tag}_e{args.epochs}_{ts}"

    data_path = Path(args.data)
    cfg = load_data_config(data_path)
    names = cfg.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): v for k, v in names.items()}

    print("=" * 66)
    print("  GUIDIO - Training YOLO11n Deteksi Bahaya Jalanan")
    print("=" * 66)
    print(f"  Data      : {data_path}")
    print(f"  Model     : {args.model}")
    print(f"  Epochs    : {args.epochs}   imgsz: {args.imgsz}   batch: {args.batch}")
    print(f"  Device    : {device}")
    print(f"  Run name  : {args.name}")
    print(f"  Kelas     : {[names[i] for i in sorted(names)]}")

    # ── Analisis distribusi ──
    print("\n" + "=" * 66)
    print("  DISTRIBUSI KELAS (train)")
    print("=" * 66)
    train_dir = resolve_split_dir(cfg, data_path, "train")
    counter = Counter()
    if train_dir and train_dir.exists():
        counter = analyze_class_distribution(train_dir, names)
        print_distribution(counter, names)
    else:
        print(f"  Folder train tidak ditemukan: {train_dir}")
        print("  Analisis distribusi dilewati.")

    # ── Oversampling ──
    effective_data = str(data_path)
    if args.oversample_minority and train_dir and train_dir.exists() and counter:
        print("\n" + "=" * 66)
        print("  OVERSAMPLING KELAS MINORITAS")
        print("=" * 66)
        os_root = ROOT / "runs" / "_oversampled" / args.name
        new_train = build_oversampled_dataset(train_dir, names, counter,
                                              os_root, args.max_repeat)
        new_cfg = dict(cfg)
        new_cfg["path"] = str(os_root.resolve())
        new_cfg["train"] = "images/train"
        # val & test tetap menunjuk dataset asli (JANGAN di-oversample:
        # validation set harus mencerminkan distribusi dunia nyata)
        base = Path(cfg.get("path", data_path.parent))
        if not base.is_absolute():
            base = (data_path.parent / base).resolve()
        for key in ("val", "test"):
            if key in cfg:
                new_cfg[key] = str((base / cfg[key]).resolve())

        os_cfg_path = os_root / "data_oversampled.yaml"
        os_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(os_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)
        effective_data = str(os_cfg_path)
        print(f"  Config oversampled: {os_cfg_path}")
        print("  Catatan: val & test SENGAJA tidak di-oversample, supaya "
              "metrik tetap mencerminkan distribusi sebenarnya.")

    # ── Augmentasi outdoor ──
    print("\n" + "=" * 66)
    print("  AUGMENTASI")
    print("=" * 66)
    if args.no_outdoor_aug:
        print("  Augmentasi outdoor DIMATIKAN (mode baseline).")
    else:
        patch_ultralytics_albumentations(args.aug_strength, verbose=True)

        if not args.no_preview and train_dir and train_dir.exists():
            samples = [p for p in train_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if samples:
                preview_dir = Path(args.project) / args.name
                preview_dir.mkdir(parents=True, exist_ok=True)
                try:
                    save_augmentation_preview(
                        str(samples[len(samples) // 2]),
                        str(preview_dir / "augmentation_preview.jpg"),
                        args.aug_strength,
                    )
                except Exception as e:
                    print(f"  Preview gagal dibuat: {e}")

    # ── Training ──
    print("\n" + "=" * 66)
    print("  TRAINING")
    print("=" * 66)

    model = YOLO(args.model)

    model.train(
        data=effective_data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        deterministic=False,   # False jauh lebih cepat; seed tetap dipakai

        # ── Optimisasi ──
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        weight_decay=0.0005,

        # ── Augmentasi geometris (ditangani Ultralytics) ──
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,      # diturunkan dari 0.4 karena variasi cahaya sekarang
                        # ditangani augmentasi outdoor yang jauh lebih realistis
        degrees=8.0,    # goyangan langkah kaki
        translate=0.12,
        scale=0.45,     # dinaikkan: jarak objek sangat bervariasi saat berjalan
        shear=2.0,
        perspective=0.0008,
        fliplr=0.5,
        flipud=0.0,     # WAJIB 0: tangga terbalik tidak pernah ada
        mosaic=1.0,
        close_mosaic=args.close_mosaic,
        mixup=0.10,
        copy_paste=0.0,  # butuh label segmentasi; pakai scripts/08 untuk bbox
        erasing=0.25,    # dinaikkan dari 0.1: objek sering tertutup sebagian
    )

    run_dir = Path(args.project) / args.name
    best_pt = run_dir / "weights" / "best.pt"
    print(f"\n  Bobot terbaik: {best_pt}")

    # ── Evaluasi ──
    best = YOLO(str(best_pt))
    summary = evaluate_and_report(best, str(data_path), args.imgsz,
                                  names, device, run_dir)

    # ── Ekspor ──
    if args.export_tflite:
        print("\n" + "=" * 66)
        print("  EKSPOR LiteRT INT8")
        print("=" * 66)
        print("  Catatan: kalibrasi INT8 memakai dataset kustom kamu, "
              "BUKAN coco8. Ini penting supaya rentang quantization sesuai "
              "domain trotoar.")
        try:
            out = YOLO(str(best_pt)).export(
                format="tflite", int8=True, data=str(data_path),
                imgsz=args.imgsz,
            )
            print(f"  Berhasil: {out}")
        except TypeError:
            # Ultralytics versi baru memakai nama argumen berbeda
            try:
                out = YOLO(str(best_pt)).export(
                    format="litert", quantize=True, data=str(data_path),
                    imgsz=args.imgsz,
                )
                print(f"  Berhasil: {out}")
            except Exception as e:
                print(f"  Gagal: {e}")
                print(f"  Coba manual: yolo export model={best_pt} "
                      f"format=tflite int8=True data={data_path}")
        except Exception as e:
            print(f"  Gagal: {e}")
            print(f"  Coba manual: yolo export model={best_pt} "
                  f"format=tflite int8=True data={data_path}")

    print("\n" + "=" * 66)
    print("  SELESAI")
    print("=" * 66)
    print(f"  Backend (.pt) : {best_pt}")
    print(f"  Ringkasan     : {run_dir / 'validation_summary.json'}")
    print("\n  Langkah berikutnya:")
    print(f"    python scripts/10_tune_thresholds.py --weights {best_pt} "
          f"--data {data_path}")
    if not summary.get("release_gate_passed"):
        print("    (gate rilis belum lolos, lihat saran di atas)")


if __name__ == "__main__":
    main()
