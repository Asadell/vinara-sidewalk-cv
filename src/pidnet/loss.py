"""
src/pidnet/loss.py
==================
Loss function untuk PIDNet-S.

KENAPA BUKAN CrossEntropyLoss BIASA
-----------------------------------
Versi lama memakai:

    class_weights = torch.tensor([1.0, 1.0, 3.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

Ada dua masalah dengan pendekatan itu:

1. Bobot statis 3x untuk hazard adalah tebakan. Kalau hazard cuma 0.5%
   dari piksel, 3x masih terlalu kecil. Kalau 15%, 3x bikin model
   over-predict hazard dan spam peringatan palsu ke pengguna. Bobotnya
   seharusnya dihitung dari distribusi dataset yang sebenarnya.

2. CrossEntropy rata-rata di SEMUA piksel. Padahal 90%+ piksel itu
   gampang (tengah jalan raya, tengah trotoar) dan sudah benar sejak
   epoch ke-5. Piksel gampang yang jumlahnya masif itu mendominasi
   gradien, sementara piksel sulit (tepi trotoar, batas hazard) yang
   justru menentukan kualitas navigasi malah tenggelam.

OHEM (Online Hard Example Mining) memecahkan nomor 2: hanya piksel
dengan loss di atas ambang yang dihitung, dengan jaminan minimal
sekian piksel supaya training tetap stabil. Ini yang dipakai paper
PIDNet asli dan mayoritas metode segmentasi real-time.

Untuk nomor 1, `compute_class_weights()` menghitung bobot dari
frekuensi piksel aktual memakai skema median-frequency balancing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Bobot kelas dari data ─────────────────────────────────────────────────────

def compute_class_weights(pixel_counts: np.ndarray,
                          method: str = "median_freq",
                          clip: tuple[float, float] = (0.25, 8.0)) -> np.ndarray:
    """
    Hitung bobot kelas dari jumlah piksel aktual.

    Args:
        pixel_counts: array (num_classes,) jumlah piksel per kelas
        method:
            "median_freq" -> bobot = median(freq) / freq  (Eigen & Fergus)
                             Stabil, tidak meledak untuk kelas sangat langka.
            "inverse"     -> bobot = total / (nc * count)
                             Lebih agresif, rawan tidak stabil.
            "log"         -> bobot = 1 / log(1.02 + freq)
                             Paling lembut.
        clip: batas bawah & atas bobot, supaya tidak ada kelas yang
              mendominasi gradien secara ekstrem

    Returns:
        array (num_classes,) float32
    """
    counts = np.asarray(pixel_counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()

    if method == "median_freq":
        w = np.median(freq) / freq
    elif method == "inverse":
        w = counts.sum() / (len(counts) * counts)
    elif method == "log":
        w = 1.0 / np.log(1.02 + freq)
    else:
        raise ValueError(f"method tidak dikenal: {method}")

    w = np.clip(w, clip[0], clip[1])
    # Normalisasi supaya rata-rata bobot = 1 (menjaga skala loss tetap sebanding
    # antar konfigurasi, jadi learning rate tidak perlu ikut disetel ulang)
    w = w / w.mean()
    return w.astype(np.float32)


# ─── OHEM Cross Entropy ────────────────────────────────────────────────────────

class OhemCrossEntropy(nn.Module):
    """
    Cross entropy dengan Online Hard Example Mining.

    Cara kerja:
      1. Hitung loss per piksel (tanpa reduction)
      2. Buang piksel yang sudah diprediksi benar dengan confidence
         di atas `thresh`
      3. Kalau piksel tersisa kurang dari `min_kept`, ambil `min_kept`
         piksel dengan loss tertinggi (supaya gradien tidak kosong)
      4. Rata-ratakan loss dari piksel terpilih

    Args:
        thresh: ambang probabilitas kelas benar. Piksel dengan
                p_true > thresh dianggap "gampang" dan dibuang.
                0.9 = agresif, 0.7 = moderat.
        min_kept: jumlah piksel minimum yang selalu dipertahankan per batch
        weight: bobot per kelas (tensor)
        ignore_index: label yang diabaikan (misal 255 untuk void)
    """

    def __init__(self, thresh: float = 0.9, min_kept: int = 100_000,
                 weight: torch.Tensor | None = None,
                 ignore_index: int = 255):
        super().__init__()
        self.thresh = thresh
        self.min_kept = max(1, min_kept)
        self.ignore_index = ignore_index
        self.register_buffer(
            "class_weight",
            weight if weight is not None else torch.empty(0),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.class_weight if self.class_weight.numel() > 0 else None

        pixel_loss = F.cross_entropy(
            logits, target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W)

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            valid = target != self.ignore_index
            safe_target = torch.where(valid, target,
                                      torch.zeros_like(target))
            p_true = probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
            # Piksel sulit = probabilitas kelas benar masih rendah
            hard = valid & (p_true < self.thresh)

        flat_loss = pixel_loss[valid]
        if flat_loss.numel() == 0:
            return logits.sum() * 0.0

        hard_loss = pixel_loss[hard]
        if hard_loss.numel() >= self.min_kept:
            return hard_loss.mean()

        # Tidak cukup piksel sulit: ambil top-k loss tertinggi
        k = min(self.min_kept, flat_loss.numel())
        topk, _ = torch.topk(flat_loss, k)
        return topk.mean()


# ─── Boundary loss ─────────────────────────────────────────────────────────────

def compute_boundary_target(mask: torch.Tensor,
                            ignore_index: int = 255) -> torch.Tensor:
    """
    Bikin target boundary dari mask segmentasi: piksel yang bertetangga
    dengan kelas berbeda ditandai 1.

    KENAPA INI PERBAIKAN PENTING
    ----------------------------
    Versi lama memakai:

        bnd_target = (masks == 2).float().unsqueeze(1)

    Itu bukan boundary, itu mask kelas hazard. Cabang D di PIDNet
    dirancang untuk belajar BATAS ANTAR ZONA, bukan satu kelas tertentu.
    Melatih cabang D untuk memprediksi seluruh area hazard membuatnya
    jadi duplikat cabang segmentasi utama, dan manfaat arsitektur
    tiga-cabangnya hilang.

    Yang benar: boundary = tepi antara zona apa pun. Tepi trotoar dengan
    jalan raya sama pentingnya dengan tepi hazard, karena di situlah
    pengguna bisa tersandung turun ke jalan.
    """
    m = mask.unsqueeze(1).float()

    # Max pool dan min pool 3x3: kalau keduanya beda, berarti ada tepi
    max_pool = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    min_pool = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)

    boundary = (max_pool != min_pool).float()

    # Abaikan area void
    if ignore_index is not None:
        valid = (mask != ignore_index).unsqueeze(1).float()
        boundary = boundary * valid

    return boundary  # (B, 1, H, W)


class BoundaryAwareLoss(nn.Module):
    """
    Loss gabungan PIDNet:
        L = L_ohem_semantic + w_bd * L_boundary + w_bas * L_boundary_aware_seg

    Komponen ketiga (boundary-aware segmentation loss) memberi bobot ekstra
    pada loss semantik DI PIKSEL YANG DEKAT BATAS. Ini yang bikin tepi
    trotoar tajam alih-alih kabur, dan tepi tajam itu yang menentukan
    apakah narasi "trotoar melebar ke kanan" akurat atau tidak.
    """

    def __init__(self, num_classes: int = 3,
                 class_weight: torch.Tensor | None = None,
                 ohem_thresh: float = 0.9,
                 min_kept: int = 100_000,
                 w_boundary: float = 20.0,
                 w_boundary_aware: float = 1.0,
                 boundary_pos_weight: float = 4.0,
                 ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.w_boundary = w_boundary
        self.w_boundary_aware = w_boundary_aware
        self.boundary_pos_weight = boundary_pos_weight

        self.semantic = OhemCrossEntropy(
            thresh=ohem_thresh, min_kept=min_kept,
            weight=class_weight, ignore_index=ignore_index,
        )
        self.register_buffer(
            "cw",
            class_weight if class_weight is not None else torch.empty(0),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, aux_boundary: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            (total_loss, dict komponen untuk logging)
        """
        # 1. Loss semantik utama (OHEM)
        loss_sem = self.semantic(logits, target)

        # 2. Loss boundary (cabang D)
        bd_target = compute_boundary_target(target, self.ignore_index)
        pos_w = torch.tensor(self.boundary_pos_weight, device=logits.device)
        loss_bd = F.binary_cross_entropy_with_logits(
            aux_boundary, bd_target, pos_weight=pos_w
        )

        # 3. Boundary-aware segmentation loss
        loss_bas = torch.zeros((), device=logits.device)
        if self.w_boundary_aware > 0:
            with torch.no_grad():
                bd_mask = (torch.sigmoid(aux_boundary) > 0.5).squeeze(1)
                valid = target != self.ignore_index
                sel = bd_mask & valid

            if sel.any():
                weight = self.cw if self.cw.numel() > 0 else None
                px = F.cross_entropy(
                    logits, target, weight=weight,
                    ignore_index=self.ignore_index, reduction="none",
                )
                loss_bas = px[sel].mean()

        total = (loss_sem
                 + self.w_boundary * loss_bd
                 + self.w_boundary_aware * loss_bas)

        return total, {
            "loss": float(total.detach()),
            "sem": float(loss_sem.detach()),
            "bd": float(loss_bd.detach()),
            "bas": float(loss_bas.detach()),
        }
