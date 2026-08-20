"""
src/pidnet/metrics.py
=====================
Metrik evaluasi segmentasi jalur untuk navigasi tunanetra.

KENAPA mIoU SAJA TIDAK CUKUP
----------------------------
mIoU memperlakukan semua kesalahan sama rata. Untuk navigasi tunanetra,
kesalahan itu SANGAT tidak simetris:

  - Memprediksi jalan raya sebagai trotoar (non_walkable -> walkable):
    pengguna melangkah ke jalan raya. Ini bisa fatal.
  - Memprediksi trotoar sebagai jalan raya (walkable -> non_walkable):
    pengguna berhenti atau memutar tanpa perlu. Menyebalkan, tidak fatal.

Model dengan mIoU 0.85 bisa jadi jauh lebih berbahaya daripada model
dengan mIoU 0.80 kalau semua kesalahannya jatuh di arah pertama.

Modul ini menghitung:
  1. mIoU standar + IoU per kelas
  2. Safety error: % piksel non_walkable yang salah jadi walkable
  3. Hazard recall: % piksel hazard yang berhasil terdeteksi
  4. mIoU PER ZONA (kiri / tengah / kanan)
  5. Zone decision accuracy: apakah keputusan tingkat-zona benar

KENAPA PER ZONA
---------------
App Vinara tidak membacakan mask piksel ke pengguna. Yang dibacakan itu
kesimpulan tingkat zona: "jalur aman di depan", "geser ke kanan".
Jadi yang benar-benar penting adalah apakah KEPUTUSAN per zona benar,
bukan apakah setiap piksel sempurna.

Model bisa punya mIoU bagus tapi keputusan zona kacau kalau kesalahannya
menumpuk di satu sisi. Sebaliknya, model dengan mIoU sedang bisa
menghasilkan navigasi yang mulus kalau kesalahannya tersebar merata.
Metrik ini mengukur yang benar-benar dipakai.
"""

from __future__ import annotations

import numpy as np
import torch

CLASS_NAMES = ["non_walkable", "walkable", "hazard"]
SEG_NON_WALKABLE, SEG_WALKABLE, SEG_HAZARD = 0, 1, 2

ZONE_NAMES = ["kiri", "tengah", "kanan"]


# ─── Confusion matrix ──────────────────────────────────────────────────────────

def fast_confusion(pred: torch.Tensor, target: torch.Tensor,
                   num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """
    Confusion matrix via bincount. Dihitung di device yang sama dengan
    tensor input (biasanya GPU), jadi tidak ada transfer CPU per batch.

    Versi lama memanggil `.cpu()` tiap batch, yang memaksa sinkronisasi
    GPU dan memperlambat training secara signifikan pada dataset besar.
    """
    mask = (target >= 0) & (target < num_classes) & (target != ignore_index)
    if not mask.any():
        return torch.zeros(num_classes, num_classes,
                           dtype=torch.int64, device=pred.device)
    idx = num_classes * target[mask].long() + pred[mask].long()
    return torch.bincount(
        idx, minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)


def iou_from_confusion(conf: torch.Tensor) -> torch.Tensor:
    """IoU per kelas dari confusion matrix."""
    conf = conf.float()
    inter = torch.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    return inter / union.clamp(min=1.0)


# ─── Metrik keselamatan ────────────────────────────────────────────────────────

def safety_error(conf: torch.Tensor) -> float:
    """
    % piksel non_walkable yang salah diprediksi sebagai walkable.
    Ini kesalahan paling berbahaya: pengguna disuruh melangkah ke
    area yang sebenarnya bukan trotoar.
    """
    total = conf[SEG_NON_WALKABLE].sum().float()
    if total == 0:
        return 0.0
    wrong = conf[SEG_NON_WALKABLE, SEG_WALKABLE].float()
    return float(wrong / total * 100.0)


def hazard_recall(conf: torch.Tensor) -> float:
    """
    % piksel hazard yang berhasil terdeteksi sebagai hazard.
    Recall lebih penting daripada precision untuk kelas ini: melewatkan
    lubang jauh lebih buruk daripada memperingatkan lubang yang ternyata
    cuma bayangan.
    """
    total = conf[SEG_HAZARD].sum().float()
    if total == 0:
        return float("nan")
    return float(conf[SEG_HAZARD, SEG_HAZARD].float() / total * 100.0)


def hazard_missed_as_walkable(conf: torch.Tensor) -> float:
    """
    % piksel hazard yang justru diprediksi sebagai walkable.
    Ini kombinasi terburuk: ada lubang, model bilang aman dilalui.
    """
    total = conf[SEG_HAZARD].sum().float()
    if total == 0:
        return float("nan")
    return float(conf[SEG_HAZARD, SEG_WALKABLE].float() / total * 100.0)


# ─── Metrik per zona ───────────────────────────────────────────────────────────

def split_zones(tensor: torch.Tensor, n_zones: int = 3) -> list[torch.Tensor]:
    """
    Bagi tensor (B, H, W) jadi n_zones potongan kolom: kiri, tengah, kanan.
    """
    w = tensor.shape[-1]
    edges = [int(round(i * w / n_zones)) for i in range(n_zones + 1)]
    return [tensor[..., edges[i]:edges[i + 1]] for i in range(n_zones)]


def zone_decision(mask_zone: torch.Tensor,
                  walkable_ratio_thresh: float = 0.35,
                  hazard_ratio_thresh: float = 0.05) -> int:
    """
    Ubah mask satu zona jadi satu keputusan navigasi.

    Return:
        0 = tidak bisa dilalui
        1 = bisa dilalui
        2 = ada bahaya (prioritas tertinggi)

    Ambang hazard sengaja kecil (5%): sedikit piksel hazard saja sudah
    cukup untuk memicu peringatan. Lebih baik memperingatkan lubang kecil
    daripada melewatkannya.
    """
    total = mask_zone.numel()
    if total == 0:
        return 0
    hazard_ratio = float((mask_zone == SEG_HAZARD).sum()) / total
    if hazard_ratio >= hazard_ratio_thresh:
        return 2
    walk_ratio = float((mask_zone == SEG_WALKABLE).sum()) / total
    return 1 if walk_ratio >= walkable_ratio_thresh else 0


class SegmentationMetrics:
    """
    Akumulator metrik lintas batch.

    Pakai:
        m = SegmentationMetrics(num_classes=3, device=device)
        for imgs, masks in loader:
            preds = model(imgs).argmax(1)
            m.update(preds, masks)
        result = m.compute()
    """

    def __init__(self, num_classes: int = 3, n_zones: int = 3,
                 ignore_index: int = 255,
                 device: torch.device | str = "cpu"):
        self.num_classes = num_classes
        self.n_zones = n_zones
        self.ignore_index = ignore_index
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        nc = self.num_classes
        self.conf = torch.zeros(nc, nc, dtype=torch.int64, device=self.device)
        self.zone_conf = [
            torch.zeros(nc, nc, dtype=torch.int64, device=self.device)
            for _ in range(self.n_zones)
        ]
        # Confusion matrix keputusan zona (3 keputusan x 3 keputusan)
        self.decision_conf = np.zeros((3, 3), dtype=np.int64)
        self.n_samples = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds:   (B, H, W) int - hasil argmax
            targets: (B, H, W) int - ground truth
        """
        preds = preds.to(self.device)
        targets = targets.to(self.device)

        self.conf += fast_confusion(preds, targets, self.num_classes,
                                    self.ignore_index)

        pred_zones = split_zones(preds, self.n_zones)
        targ_zones = split_zones(targets, self.n_zones)
        for i in range(self.n_zones):
            self.zone_conf[i] += fast_confusion(
                pred_zones[i], targ_zones[i],
                self.num_classes, self.ignore_index,
            )

        # Keputusan tingkat zona, per gambar
        for b in range(preds.shape[0]):
            for i in range(self.n_zones):
                d_pred = zone_decision(pred_zones[i][b])
                d_true = zone_decision(targ_zones[i][b])
                self.decision_conf[d_true, d_pred] += 1
        self.n_samples += preds.shape[0]

    def compute(self) -> dict:
        iou = iou_from_confusion(self.conf)

        zone_results = {}
        for i, name in enumerate(ZONE_NAMES[:self.n_zones]):
            z_iou = iou_from_confusion(self.zone_conf[i])
            zone_results[name] = {
                "miou": float(z_iou.mean()),
                "iou_per_class": {
                    CLASS_NAMES[c]: float(z_iou[c])
                    for c in range(self.num_classes)
                },
                "safety_error_pct": safety_error(self.zone_conf[i]),
            }

        dc = self.decision_conf
        total_dec = dc.sum()
        dec_acc = float(np.trace(dc) / total_dec) if total_dec else 0.0

        # Keputusan berbahaya: sebenarnya tidak bisa dilalui / bahaya,
        # tapi model bilang bisa dilalui
        unsafe = int(dc[0, 1] + dc[2, 1])
        unsafe_rate = float(unsafe / total_dec) if total_dec else 0.0

        return {
            "miou": float(iou.mean()),
            "iou_per_class": {
                CLASS_NAMES[c]: float(iou[c]) for c in range(self.num_classes)
            },
            "safety_error_pct": safety_error(self.conf),
            "hazard_recall_pct": hazard_recall(self.conf),
            "hazard_as_walkable_pct": hazard_missed_as_walkable(self.conf),
            "zones": zone_results,
            "zone_decision_accuracy": dec_acc,
            "zone_unsafe_decision_rate": unsafe_rate,
            "zone_decision_confusion": dc.tolist(),
            "n_samples": self.n_samples,
        }

    def format_report(self, result: dict | None = None) -> str:
        """Format hasil jadi teks yang enak dibaca di terminal."""
        r = result if result is not None else self.compute()
        lines = []
        lines.append(f"  mIoU keseluruhan : {r['miou']:.4f}")
        for name, val in r["iou_per_class"].items():
            lines.append(f"    {name:<14}: {val:.4f}")

        lines.append("")
        lines.append(f"  Safety error     : {r['safety_error_pct']:.2f}%  "
                     f"(non_walkable dibaca walkable; target <= 5%)")
        hr = r["hazard_recall_pct"]
        lines.append(f"  Hazard recall    : "
                     f"{'n/a' if np.isnan(hr) else f'{hr:.2f}%'}  "
                     f"(target >= 85%)")
        hw = r["hazard_as_walkable_pct"]
        lines.append(f"  Hazard->walkable : "
                     f"{'n/a' if np.isnan(hw) else f'{hw:.2f}%'}  "
                     f"(paling berbahaya; target <= 3%)")

        lines.append("")
        lines.append(f"  {'Zona':<8} {'mIoU':>8} {'safety_err':>12}")
        lines.append(f"  {'-' * 30}")
        for name, z in r["zones"].items():
            lines.append(f"  {name:<8} {z['miou']:>8.4f} "
                         f"{z['safety_error_pct']:>11.2f}%")

        lines.append("")
        lines.append(f"  Akurasi keputusan zona : "
                     f"{r['zone_decision_accuracy'] * 100:.2f}%")
        lines.append(f"  Keputusan tidak aman   : "
                     f"{r['zone_unsafe_decision_rate'] * 100:.2f}%  "
                     f"(zona bahaya/terlarang dibilang bisa dilalui)")

        dc = np.array(r["zone_decision_confusion"])
        lines.append("")
        lines.append("  Confusion keputusan zona (baris=asli, kolom=prediksi)")
        lines.append(f"  {'':>14}{'tdk_bisa':>10}{'bisa':>10}{'bahaya':>10}")
        for i, nm in enumerate(["tdk_bisa", "bisa", "bahaya"]):
            lines.append(f"  {nm:>14}{dc[i, 0]:>10}{dc[i, 1]:>10}{dc[i, 2]:>10}")

        return "\n".join(lines)


# ─── Statistik distribusi piksel ───────────────────────────────────────────────

@torch.no_grad()
def count_class_pixels(loader, num_classes: int = 3,
                       ignore_index: int = 255,
                       max_batches: int | None = None) -> np.ndarray:
    """
    Hitung jumlah piksel per kelas di seluruh dataset.
    Dipakai untuk menghitung bobot kelas berbasis data,
    bukan tebakan seperti [1.0, 1.0, 3.0].
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for i, batch in enumerate(loader):
        masks = batch[1]
        valid = masks[masks != ignore_index]
        bc = torch.bincount(valid.flatten().long(), minlength=num_classes)
        counts += bc.cpu().numpy()[:num_classes]
        if max_batches is not None and i + 1 >= max_batches:
            break
    return counts
