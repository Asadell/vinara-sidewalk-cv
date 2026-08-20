#!/usr/bin/env python3
"""
06_train_pidnet.py  (REVISI)
============================
Training PIDNet-S segmentasi jalur 3 zona.

RINGKASAN PERUBAHAN DARI VERSI LAMA

  Loss
    - OHEM CrossEntropy menggantikan CE biasa. 90%+ piksel itu gampang
      (tengah jalan, tengah trotoar) dan mendominasi gradien; OHEM
      memfokuskan training ke piksel sulit di tepi zona, yang justru
      menentukan kualitas navigasi.
    - Bobot kelas DIHITUNG dari distribusi piksel aktual, bukan tebakan
      [1.0, 1.0, 3.0]
    - Target boundary diperbaiki: versi lama memakai `(masks == 2)` yang
      itu mask kelas hazard, BUKAN boundary. Akibatnya cabang D cuma jadi
      duplikat cabang segmentasi dan manfaat arsitektur tiga-cabang hilang.

  Optimisasi
    - AMP (mixed precision) - sekitar 1.5-2x lebih cepat di GPU modern,
      hemat VRAM, jadi batch bisa lebih besar
    - PolynomialLR (power 0.9) - standar de facto untuk segmentasi,
      menggantikan CosineAnnealingLR
    - Warmup, gradient clipping, EMA
    - `torch.backends.cudnn.benchmark` untuk input berukuran tetap

  Evaluasi
    - mIoU PER ZONA (kiri/tengah/kanan) plus akurasi keputusan zona,
      karena app membacakan keputusan zona, bukan mask piksel
    - Safety error dipisah: non_walkable->walkable dan hazard->walkable
    - Checkpoint terbaik dipilih dengan skor komposit yang memperhitungkan
      keselamatan, bukan mIoU semata

  Kebenaran
    - `run_epoch` tidak lagi menerima optimizer saat validasi
    - Confusion matrix dihitung di GPU (versi lama memanggil .cpu() tiap
      batch, memaksa sinkronisasi dan memperlambat training)
    - Resume training yang benar (optimizer + scheduler + scaler state)

Usage:
    python scripts/06_train_pidnet.py \
        --dataset-root ../dataset_master_seg \
        --epochs 120 --batch-size 12 --img-size 512 \
        --base-ch 32 --use-ibn --copy-paste 0.4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pidnet.dataset import (  # noqa: E402
    IGNORE_INDEX,
    SidewalkSegDataset,
    save_dataset_preview,
)
from src.pidnet.loss import BoundaryAwareLoss, compute_class_weights  # noqa: E402
from src.pidnet.metrics import (  # noqa: E402
    CLASS_NAMES,
    SegmentationMetrics,
    count_class_pixels,
)
from src.pidnet.model import PIDNetS, count_parameters  # noqa: E402

NUM_CLASSES = 3


# ─── Scheduler ─────────────────────────────────────────────────────────────────

class PolyLRWithWarmup:
    """
    Polynomial decay dengan warmup linear.

        lr = base_lr * (1 - iter/max_iter) ** power

    Ini schedule standar untuk semantic segmentation (dipakai DeepLab,
    PSPNet, PIDNet, dan hampir semua metode di Cityscapes). Dibanding
    cosine, poly menahan LR relatif tinggi lebih lama lalu turun tajam
    di akhir, yang cocok untuk tugas dense prediction.

    Diimplementasi manual (bukan torch.optim.lr_scheduler.PolynomialLR)
    supaya warmup dan per-iterasi stepping-nya eksplisit dan bisa
    di-resume dengan benar.
    """

    def __init__(self, optimizer, base_lr: float, max_iters: int,
                 power: float = 0.9, warmup_iters: int = 500,
                 warmup_ratio: float = 0.1, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_iters = max(1, max_iters)
        self.power = power
        self.warmup_iters = max(0, warmup_iters)
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr
        self.last_iter = 0

    def get_lr(self, it: int) -> float:
        if it < self.warmup_iters:
            alpha = it / max(1, self.warmup_iters)
            factor = self.warmup_ratio * (1 - alpha) + alpha
            return self.base_lr * factor
        progress = (it - self.warmup_iters) / max(
            1, self.max_iters - self.warmup_iters)
        progress = min(1.0, max(0.0, progress))
        return max(self.min_lr,
                   self.base_lr * ((1 - progress) ** self.power))

    def step(self, it: int | None = None) -> float:
        if it is None:
            it = self.last_iter + 1
        self.last_iter = it
        lr = self.get_lr(it)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def state_dict(self):
        return {"last_iter": self.last_iter}

    def load_state_dict(self, sd):
        self.last_iter = sd.get("last_iter", 0)


# ─── EMA ───────────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Exponential moving average bobot model.

    Untuk segmentasi, EMA hampir selalu memberi mIoU sedikit lebih tinggi
    dan yang lebih penting: kurva validasi jauh lebih halus, sehingga
    pemilihan checkpoint terbaik tidak terjebak pada spike acak.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.buffers = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if not v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().float(), alpha=1 - self.decay)
            else:
                self.buffers[k] = v.detach().clone()

    def state_dict(self):
        out = {k: v.clone() for k, v in self.shadow.items()}
        out.update({k: v.clone() for k, v in self.buffers.items()})
        return out

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.state_dict(), strict=False)


# ─── Epoch ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, epoch, total_epochs, global_step, ema=None,
                    aux_weight=0.4, grad_clip=5.0, log_every=20):
    model.train()
    running = {"loss": 0.0, "sem": 0.0, "bd": 0.0, "bas": 0.0}
    n_batches = 0
    t0 = time.time()

    use_amp = scaler is not None and scaler.is_enabled()

    for i, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        lr = scheduler.step(global_step)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            logits, boundary, aux = model(imgs, return_aux=True)
            loss, parts = criterion(logits, boundary, masks)
            if aux is not None and aux_weight > 0:
                aux_loss = nn.functional.cross_entropy(
                    aux, masks, ignore_index=IGNORE_INDEX)
                loss = loss + aux_weight * aux_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        for k in running:
            running[k] += parts.get(k, 0.0)
        n_batches += 1
        global_step += 1

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            ips = (i + 1) * imgs.size(0) / max(elapsed, 1e-6)
            print(f"    [{epoch}/{total_epochs}] "
                  f"batch {i + 1}/{len(loader)}  "
                  f"loss={running['loss'] / n_batches:.4f}  "
                  f"lr={lr:.2e}  {ips:.1f} img/s")

    avg = {k: v / max(1, n_batches) for k, v in running.items()}
    return avg, global_step


@torch.no_grad()
def validate(model, loader, criterion, device, aux_weight=0.4):
    model.eval()
    metrics = SegmentationMetrics(NUM_CLASSES, n_zones=3,
                                  ignore_index=IGNORE_INDEX, device=device)
    total_loss = 0.0
    n_batches = 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits, boundary, _ = model(imgs, return_aux=True)
        loss, _ = criterion(logits, boundary, masks)

        total_loss += float(loss)
        n_batches += 1
        metrics.update(logits.argmax(dim=1), masks)

    return total_loss / max(1, n_batches), metrics.compute(), metrics


def composite_score(result: dict) -> float:
    """
    Skor komposit untuk memilih checkpoint terbaik.

    KENAPA BUKAN mIoU SAJA
    Model dengan mIoU 0.85 yang sering menyebut jalan raya sebagai
    trotoar lebih berbahaya daripada model mIoU 0.80 yang konservatif.
    Memilih checkpoint hanya berdasar mIoU berarti kamu bisa mendeploy
    model yang secara statistik lebih baik tapi secara praktis lebih
    berisiko.

    Bobotnya:
      0.45 * mIoU
      0.30 * akurasi keputusan zona   (yang benar-benar dipakai app)
      0.15 * (1 - safety error)       (jalan raya dibaca trotoar)
      0.10 * (1 - hazard->walkable)   (lubang dibilang aman)
    """
    miou = result["miou"]
    zone_acc = result["zone_decision_accuracy"]
    safety = result["safety_error_pct"] / 100.0
    hz = result["hazard_as_walkable_pct"]
    hz = 0.0 if np.isnan(hz) else hz / 100.0

    return (0.45 * miou
            + 0.30 * zone_acc
            + 0.15 * (1.0 - safety)
            + 0.10 * (1.0 - hz))


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Training PIDNet-S segmentasi jalur")
    ap.add_argument("--dataset-root", required=True,
                    help="Folder dataset_master_seg hasil script 04")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "pidnet"))
    ap.add_argument("--name", default=None)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--power", type=float, default=0.9)
    ap.add_argument("--warmup-iters", type=int, default=500)
    ap.add_argument("--grad-clip", type=float, default=5.0)

    ap.add_argument("--base-ch", type=int, default=32,
                    help="Lebar model. 16=sangat ringan, 24=seimbang, 32=akurat")
    ap.add_argument("--use-ibn", action="store_true", default=True)
    ap.add_argument("--no-ibn", dest="use_ibn", action="store_false")
    ap.add_argument("--ibn-ratio", type=float, default=0.5)
    ap.add_argument("--no-deep-supervision", action="store_true")
    ap.add_argument("--aux-weight", type=float, default=0.4)

    ap.add_argument("--aug-strength", choices=["light", "medium", "heavy"],
                    default="medium")
    ap.add_argument("--copy-paste", type=float, default=0.4,
                    help="Probabilitas copy-paste hazard; 0 = matikan")
    ap.add_argument("--ohem-thresh", type=float, default=0.9)
    ap.add_argument("--min-kept-frac", type=float, default=0.08,
                    help="Porsi piksel minimum yang dipertahankan OHEM")
    ap.add_argument("--weight-method", default="median_freq",
                    choices=["median_freq", "inverse", "log"])
    ap.add_argument("--w-boundary", type=float, default=20.0)

    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None, help="Path checkpoint last.pth")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if not args.name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"pidnet_s_c{args.base_ch}_e{args.epochs}_{ts}"

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"Dataset tidak ditemukan: {dataset_root}")
        sys.exit(1)

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  GUIDIO - Training PIDNet-S Segmentasi Jalur 3 Zona")
    print("=" * 66)
    print(f"  Dataset  : {dataset_root}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {args.epochs}  batch: {args.batch_size}  "
          f"img: {args.img_size}")
    print(f"  base_ch  : {args.base_ch}   IBN: {args.use_ibn}")
    print(f"  Output   : {out_dir}")

    # ── Dataset ──
    print("\nMenyiapkan dataset...")
    train_ds = SidewalkSegDataset(
        str(dataset_root), "train", args.img_size, augment=True,
        aug_strength=args.aug_strength, copy_paste_p=args.copy_paste,
    )
    # Nama split validasi bisa "valid" atau "val" tergantung script konversi
    val_split = "valid" if (dataset_root / "images" / "valid").exists() else "val"
    val_ds = SidewalkSegDataset(
        str(dataset_root), val_split, args.img_size, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0,
    )
    print(f"  Train: {len(train_ds)} gambar | Valid: {len(val_ds)} gambar")

    if not args.no_preview:
        try:
            save_dataset_preview(train_ds, str(out_dir / "dataset_preview.jpg"),
                                 n=5)
        except Exception as e:
            print(f"  Preview gagal: {e}")

    # ── Bobot kelas dari data ──
    print("\nMenghitung distribusi piksel untuk bobot kelas...")
    stat_loader = DataLoader(train_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers)
    max_b = max(1, min(len(stat_loader), 60))
    pixel_counts = count_class_pixels(stat_loader, NUM_CLASSES,
                                      IGNORE_INDEX, max_batches=max_b)
    total_px = pixel_counts.sum()
    print(f"  {'Kelas':<16} {'piksel':>14} {'porsi':>8}")
    print(f"  {'-' * 40}")
    for i, name in enumerate(CLASS_NAMES):
        pct = 100.0 * pixel_counts[i] / max(1, total_px)
        print(f"  {name:<16} {int(pixel_counts[i]):>14,} {pct:>7.2f}%")

    class_w = compute_class_weights(pixel_counts, method=args.weight_method)
    print(f"\n  Bobot kelas ({args.weight_method}): "
          f"{dict(zip(CLASS_NAMES, np.round(class_w, 3)))}")
    print("  Versi lama memakai [1.0, 1.0, 3.0] yang ditebak manual. "
          "Angka di atas dihitung dari distribusi piksel sebenarnya.")

    # ── Model ──
    model = PIDNetS(
        num_classes=NUM_CLASSES, base_ch=args.base_ch,
        use_ibn=args.use_ibn, ibn_ratio=args.ibn_ratio,
        deep_supervision=not args.no_deep_supervision,
    ).to(device)
    n_params, n_m = count_parameters(model)
    print(f"\n  Parameter model: {n_params:,} ({n_m:.2f} M)")

    # ── Loss ──
    px_per_batch = args.batch_size * args.img_size * args.img_size
    min_kept = int(px_per_batch * args.min_kept_frac)
    criterion = BoundaryAwareLoss(
        num_classes=NUM_CLASSES,
        class_weight=torch.tensor(class_w, device=device),
        ohem_thresh=args.ohem_thresh,
        min_kept=min_kept,
        w_boundary=args.w_boundary,
    ).to(device)
    print(f"  OHEM: thresh={args.ohem_thresh}, min_kept={min_kept:,} piksel")

    # ── Optimizer & scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    iters_per_epoch = max(1, len(train_loader))
    max_iters = iters_per_epoch * args.epochs
    scheduler = PolyLRWithWarmup(optimizer, args.lr, max_iters,
                                 power=args.power,
                                 warmup_iters=args.warmup_iters)

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"  AMP: {use_amp}   EMA: {not args.no_ema}")

    ema = None if args.no_ema else ModelEMA(model, args.ema_decay)

    # ── Resume ──
    start_epoch = 1
    global_step = 0
    best_score = 0.0
    history = []

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler"])
            if ema is not None and "ema" in ckpt:
                ema.shadow = {k: v.to(device)
                              for k, v in ckpt["ema"].items()
                              if v.dtype.is_floating_point}
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)
            best_score = ckpt.get("best_score", 0.0)
            history = ckpt.get("history", [])
            print(f"\n  Resume dari epoch {start_epoch}")
        else:
            print(f"\n  Checkpoint resume tidak ada: {ckpt_path}")

    # ── Loop ──
    print("\n" + "=" * 66)
    print("  MULAI TRAINING")
    print("=" * 66)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_stats, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, args.epochs, global_step, ema,
            aux_weight=args.aux_weight, grad_clip=args.grad_clip,
        )

        # Validasi memakai bobot EMA kalau tersedia
        if ema is not None:
            backup = {k: v.detach().clone()
                      for k, v in model.state_dict().items()}
            ema.apply_to(model)

        val_loss, result, metrics = validate(model, val_loader, criterion,
                                             device, args.aux_weight)

        if ema is not None:
            model.load_state_dict(backup)

        score = composite_score(result)
        dt = time.time() - t0

        print(f"\n  Epoch {epoch}/{args.epochs}  ({dt:.0f}s)")
        print(f"    train loss={train_stats['loss']:.4f} "
              f"(sem={train_stats['sem']:.3f} bd={train_stats['bd']:.3f})")
        print(f"    valid loss={val_loss:.4f}  "
              f"mIoU={result['miou']:.4f}  skor={score:.4f}")
        print(f"    safety_err={result['safety_error_pct']:.2f}%  "
              f"zona_acc={result['zone_decision_accuracy'] * 100:.1f}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_loss,
            "miou": result["miou"],
            "score": score,
            "safety_error_pct": result["safety_error_pct"],
            "zone_decision_accuracy": result["zone_decision_accuracy"],
        })

        state = {
            "model": (ema.state_dict() if ema is not None
                      else model.state_dict()),
            "raw_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_score": best_score,
            "history": history,
            "config": vars(args),
            "class_weights": class_w.tolist(),
        }
        if ema is not None:
            state["ema"] = ema.state_dict()
        torch.save(state, out_dir / "last.pth")

        if score > best_score:
            best_score = score
            state["best_score"] = best_score
            torch.save(state, out_dir / "best.pth")
            print(f"    -> checkpoint terbaik disimpan (skor={score:.4f})")
            print(metrics.format_report(result))

    # ── Evaluasi akhir ──
    print("\n" + "=" * 66)
    print("  EVALUASI AKHIR (checkpoint terbaik)")
    print("=" * 66)

    best_ckpt = torch.load(out_dir / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model"], strict=False)
    _, final_result, final_metrics = validate(model, val_loader, criterion,
                                              device, args.aux_weight)
    print(final_metrics.format_report(final_result))

    # ── Gate rilis ──
    print("\n" + "=" * 66)
    print("  GATE RILIS")
    print("=" * 66)
    hz_recall = final_result["hazard_recall_pct"]
    gates = [
        ("mIoU keseluruhan", final_result["miou"], 0.70, "min"),
        ("Safety error", final_result["safety_error_pct"], 5.0, "max"),
        ("Akurasi keputusan zona",
         final_result["zone_decision_accuracy"] * 100, 85.0, "min"),
        ("Keputusan tidak aman",
         final_result["zone_unsafe_decision_rate"] * 100, 3.0, "max"),
    ]
    if not np.isnan(hz_recall):
        gates.insert(1, ("Hazard recall", hz_recall, 85.0, "min"))

    all_pass = True
    for label, value, threshold, direction in gates:
        ok = value >= threshold if direction == "min" else value <= threshold
        all_pass = all_pass and ok
        sym = ">=" if direction == "min" else "<="
        print(f"  [{'LOLOS' if ok else 'GAGAL'}] {label:<24} "
              f"{value:>7.2f}  (target {sym} {threshold})")

    if not all_pass:
        print("\n  Belum layak rilis. Urutan yang biasanya paling berdampak:")
        print("    1. Tambah data pada kondisi yang paling lemah "
              "(cek per-zona di atas untuk tahu sisi mana)")
        print("    2. Naikkan --aug-strength ke heavy")
        print("    3. Naikkan --copy-paste kalau hazard recall yang rendah")
        print("    4. Naikkan --base-ch kalau mIoU mentok di semua kondisi")

    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "result": final_result,
            "release_gate_passed": all_pass,
            "history": history,
            "config": vars(args),
            "n_params": n_params,
        }, f, indent=2)

    print("\n" + "=" * 66)
    print("  SELESAI")
    print("=" * 66)
    print(f"  Checkpoint terbaik : {out_dir / 'best.pth'}")
    print(f"  Metrik             : {out_dir / 'final_metrics.json'}")
    print("\n  Langkah berikutnya:")
    print(f"    python scripts/07_export_onnx.py --checkpoint "
          f"{out_dir / 'best.pth'} --img-size {args.img_size}")


if __name__ == "__main__":
    main()
