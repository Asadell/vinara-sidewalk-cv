#!/usr/bin/env python3
"""
10_tune_thresholds.py  (BARU)
=============================
Cari confidence threshold OPTIMAL PER KELAS untuk deployment.

KENAPA SATU THRESHOLD UNTUK SEMUA KELAS ITU KELIRU
--------------------------------------------------
Default Ultralytics memakai satu `conf=0.25` untuk semua kelas. Untuk
aplikasi navigasi tunanetra, itu keputusan yang salah, karena biaya
kesalahan tiap kelas sangat berbeda:

  lubang / got_terbuka / tangga (kelas BAHAYA)
    False negative = pengguna tidak diperingatkan lalu terperosok.
    False positive = pengguna melambat sebentar tanpa perlu.
    Biayanya sangat timpang, jadi threshold harus RENDAH (recall tinggi).

  orang / motor (kelas INFORMASIONAL)
    False negative = pengguna tidak tahu ada orang lewat. Tidak ideal
    tapi biasanya masih aman.
    False positive = narasi TTS ikut ramai, pengguna terganggu, dan
    peringatan penting jadi tenggelam di antara peringatan sepele.
    Jadi threshold harus LEBIH TINGGI (precision lebih penting).

Script ini menyapu berbagai threshold per kelas, lalu memilih:
  - kelas bahaya: threshold TERTINGGI yang masih memenuhi target recall
    (maksimalkan precision dengan syarat recall aman terpenuhi)
  - kelas info  : threshold yang memaksimalkan F-beta dengan beta < 1
    (condong ke precision)

Hasilnya ditulis ke JSON yang bisa langsung dipakai backend dan Flutter.

Usage:
    python scripts/10_tune_thresholds.py \
        --weights runs/yolo/<run>/weights/best.pt \
        --data configs/custom_navigasi.yaml \
        --split val \
        --hazard-recall-target 0.90
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

HAZARD_CLASSES = {"lubang", "got_terbuka", "tangga"}


# ─── Utilitas ──────────────────────────────────────────────────────────────────

def load_config(data_yaml: Path):
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw = cfg.get("names", {})
    names = ({i: n for i, n in enumerate(raw)} if isinstance(raw, list)
             else {int(k): v for k, v in raw.items()})
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    return cfg, names, base


def load_gt(label_path: Path) -> list:
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8",
                                     errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        out.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


# ─── Pengumpulan prediksi ──────────────────────────────────────────────────────

def collect_predictions(model, images: list[Path], labels_dir: Path,
                        imgsz: int, device, iou_thresh: float,
                        base_conf: float = 0.01):
    """
    Jalankan inferensi sekali saja pada confidence sangat rendah,
    lalu simpan semua deteksi beserta skornya.

    Melakukan inferensi SEKALI lalu menyapu threshold secara offline
    jauh lebih cepat daripada menjalankan model ulang untuk tiap
    kandidat threshold.

    Returns:
        detections: dict cls -> list (score, is_true_positive)
        gt_counts : dict cls -> jumlah ground truth
    """
    detections = defaultdict(list)
    gt_counts = defaultdict(int)

    batch = 16
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict([str(p) for p in chunk], conf=base_conf,
                                iou=0.6, imgsz=imgsz, device=device,
                                verbose=False)

        for img_path, res in zip(chunk, results):
            gt = load_gt(labels_dir / f"{img_path.stem}.txt")
            for cls, *_ in gt:
                gt_counts[cls] += 1

            preds = []
            if res.boxes is not None and len(res.boxes) > 0:
                for cls_t, conf_t, xywhn in zip(res.boxes.cls, res.boxes.conf,
                                                res.boxes.xywhn):
                    cx, cy, w, h = (float(v) for v in xywhn)
                    preds.append((
                        int(cls_t), float(conf_t),
                        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                    ))

            # Cocokkan prediksi ke ground truth, skor tertinggi duluan
            preds.sort(key=lambda x: -x[1])
            matched = set()
            for cls, score, box in preds:
                best_iou, best_idx = 0.0, -1
                for gi, (gcls, *gbox) in enumerate(gt):
                    if gi in matched or gcls != cls:
                        continue
                    v = iou_xyxy(box, tuple(gbox))
                    if v > best_iou:
                        best_iou, best_idx = v, gi
                is_tp = best_iou >= iou_thresh and best_idx >= 0
                if is_tp:
                    matched.add(best_idx)
                detections[cls].append((score, is_tp))

        done = min(start + batch, len(images))
        if done % 160 == 0 or done == len(images):
            print(f"    {done}/{len(images)} gambar")

    return detections, gt_counts


# ─── Sweep ─────────────────────────────────────────────────────────────────────

def sweep_class(dets: list, n_gt: int, thresholds: np.ndarray) -> list[dict]:
    """Hitung precision/recall/F1 pada tiap threshold untuk satu kelas."""
    if n_gt == 0:
        return []
    scores = np.array([d[0] for d in dets]) if dets else np.array([])
    tps = np.array([d[1] for d in dets]) if dets else np.array([])

    rows = []
    for t in thresholds:
        if scores.size == 0:
            rows.append({"threshold": float(t), "precision": 0.0,
                         "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0,
                         "fn": n_gt})
            continue
        keep = scores >= t
        tp = int(tps[keep].sum())
        fp = int(keep.sum() - tp)
        fn = n_gt - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, n_gt)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        rows.append({"threshold": float(t), "precision": precision,
                     "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    return rows


def pick_hazard_threshold(rows: list[dict], recall_target: float) -> dict:
    """
    Untuk kelas bahaya: ambil threshold TERTINGGI yang recall-nya masih
    memenuhi target. Ini memaksimalkan precision (mengurangi peringatan
    palsu) dengan syarat keselamatan tetap terpenuhi.
    """
    feasible = [r for r in rows if r["recall"] >= recall_target]
    if feasible:
        return max(feasible, key=lambda r: r["threshold"])
    best = max(rows, key=lambda r: r["recall"])
    best = dict(best)
    best["_note"] = (f"Target recall {recall_target:.2f} TIDAK tercapai pada "
                     f"threshold mana pun. Recall maksimum "
                     f"{best['recall']:.3f}. Model perlu diperbaiki, bukan "
                     f"cuma diambangi.")
    return best


def pick_info_threshold(rows: list[dict], beta: float = 0.5) -> dict:
    """
    Untuk kelas informasional: maksimalkan F-beta dengan beta < 1,
    yang memberi bobot lebih besar ke precision.

    Alasannya bukan soal keselamatan tapi soal kebisingan narasi:
    setiap false positive berarti satu narasi TTS yang tidak perlu,
    dan narasi yang terlalu ramai justru menenggelamkan peringatan penting.
    """
    best, best_score = None, -1.0
    b2 = beta ** 2
    for r in rows:
        p, rc = r["precision"], r["recall"]
        if p + rc == 0:
            continue
        fbeta = (1 + b2) * p * rc / (b2 * p + rc)
        if fbeta > best_score:
            best_score, best = fbeta, dict(r, fbeta=fbeta)
    return best if best is not None else (rows[0] if rows else {})


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Tuning confidence threshold per kelas (recall-first)"
    )
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "valid", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou-thresh", type=float, default=0.45,
                    help="IoU untuk mencocokkan prediksi ke ground truth")
    ap.add_argument("--hazard-recall-target", type=float, default=0.90)
    ap.add_argument("--info-beta", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"Dependensi kurang: {e}")
        sys.exit(1)

    data_yaml = Path(args.data)
    cfg, names, base = load_config(data_yaml)

    split_rel = cfg.get(args.split) or cfg.get("val") or "images/valid"
    img_dir = base / split_rel
    if not img_dir.exists():
        print(f"Folder split tidak ada: {img_dir}")
        sys.exit(1)
    lbl_dir = Path(str(img_dir).replace("/images/", "/labels/"))

    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMG_EXTENSIONS)
    if args.max_images:
        images = images[:args.max_images]

    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 66)
    print("  TUNING THRESHOLD PER KELAS")
    print("=" * 66)
    print(f"  Bobot   : {args.weights}")
    print(f"  Split   : {args.split} ({len(images)} gambar)")
    print(f"  Device  : {device}")
    print(f"  Target recall kelas bahaya: {args.hazard_recall_target}")

    model = YOLO(args.weights)

    print("\nMengumpulkan prediksi (sekali jalan, conf=0.01)...")
    detections, gt_counts = collect_predictions(
        model, images, lbl_dir, args.imgsz, device, args.iou_thresh)

    thresholds = np.arange(0.05, 0.96, 0.01)

    print("\n" + "=" * 66)
    print("  HASIL PER KELAS")
    print("=" * 66)

    result = {}
    for cid in sorted(names):
        name = names[cid]
        n_gt = gt_counts.get(cid, 0)
        is_hazard = name in HAZARD_CLASSES

        if n_gt == 0:
            print(f"\n  {name}: tidak ada ground truth di split ini, dilewati.")
            result[name] = {"threshold": 0.25, "note": "tidak ada GT"}
            continue

        rows = sweep_class(detections.get(cid, []), n_gt, thresholds)
        if not rows:
            continue

        chosen = (pick_hazard_threshold(rows, args.hazard_recall_target)
                  if is_hazard else pick_info_threshold(rows, args.info_beta))

        kind = "BAHAYA (recall-first)" if is_hazard else "INFO (precision-first)"
        print(f"\n  {name}  [{kind}]")
        print(f"    ground truth : {n_gt}")
        print(f"    threshold    : {chosen['threshold']:.2f}")
        print(f"    precision    : {chosen['precision']:.3f}")
        print(f"    recall       : {chosen['recall']:.3f}")
        print(f"    TP/FP/FN     : {chosen['tp']}/{chosen['fp']}/{chosen['fn']}")
        if "_note" in chosen:
            print(f"    CATATAN: {chosen['_note']}")

        # Tunjukkan trade-off di sekitar pilihan
        print(f"    {'thr':>6} {'prec':>7} {'recall':>7}")
        for t in (0.15, 0.25, 0.35, 0.50, 0.70):
            idx = int(np.argmin(np.abs(thresholds - t)))
            r = rows[idx]
            mark = " <--" if abs(r["threshold"] - chosen["threshold"]) < 0.005 else ""
            print(f"    {r['threshold']:>6.2f} {r['precision']:>7.3f} "
                  f"{r['recall']:>7.3f}{mark}")

        result[name] = {
            "class_id": cid,
            "threshold": round(float(chosen["threshold"]), 3),
            "precision": round(float(chosen["precision"]), 4),
            "recall": round(float(chosen["recall"]), 4),
            "is_hazard": is_hazard,
            "n_ground_truth": n_gt,
        }
        if "_note" in chosen:
            result[name]["warning"] = chosen["_note"]

    # ── Ringkasan ──
    print("\n" + "=" * 66)
    print("  RINGKASAN")
    print("=" * 66)
    print(f"  {'Kelas':<14} {'threshold':>10} {'precision':>10} {'recall':>9}")
    print(f"  {'-' * 46}")
    for name, r in result.items():
        if "threshold" not in r or "precision" not in r:
            continue
        mark = " *" if r.get("is_hazard") else ""
        print(f"  {name:<14} {r['threshold']:>10.2f} "
              f"{r['precision']:>10.3f} {r['recall']:>9.3f}{mark}")

    failed = [n for n, r in result.items() if "warning" in r]
    if failed:
        print(f"\n  Kelas yang tidak mencapai target recall: {failed}")
        print("  Threshold tidak bisa memperbaiki model yang memang belum "
              "bisa melihat objeknya. Yang dibutuhkan adalah data tambahan "
              "untuk kelas itu, khususnya pada kondisi sulit.")

    out_path = Path(args.output) if args.output else \
        Path(args.weights).parent / "class_thresholds.json"

    payload = {
        "thresholds": {n: r.get("threshold", 0.25) for n, r in result.items()},
        "detail": result,
        "config": {
            "hazard_recall_target": args.hazard_recall_target,
            "info_beta": args.info_beta,
            "iou_thresh": args.iou_thresh,
            "split": args.split,
            "n_images": len(images),
        },
        "usage_note": (
            "Jalankan model pada conf rendah (misal 0.10) lalu saring "
            "sendiri per kelas memakai threshold di atas. Kalau memakai "
            "satu conf global, kamu terpaksa memilih antara melewatkan "
            "lubang atau membanjiri pengguna dengan narasi orang lewat."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n  Tersimpan: {out_path}")
    print("\n  Cara pakai di backend / Flutter:")
    print("    results = model.predict(img, conf=0.10)   # ambang rendah")
    print("    for det in results:")
    print("        if det.conf < thresholds[det.class_name]: continue")
    print("        # proses deteksi")


if __name__ == "__main__":
    main()
