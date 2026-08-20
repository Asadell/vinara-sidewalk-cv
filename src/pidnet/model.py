"""
src/pidnet/model.py
===================
PIDNet-S dengan Instance-Batch Normalization (IBN) untuk domain
generalization.

PERUBAHAN UTAMA DARI VERSI LAMA
-------------------------------

1. IBN BLOCKS (perubahan paling berdampak, biayanya nol)
   Instance Normalization menormalkan tiap gambar secara independen,
   sehingga menghapus informasi "gaya": kecerahan global, suhu warna,
   kontras. Yang tersisa adalah struktur bentuk. Batch Normalization
   sebaliknya menjaga informasi diskriminatif.

   IBN-Net (Pan et al., ECCV 2018) menggabungkan keduanya: IN di layer
   dangkal (tempat variasi gaya paling dominan), BN di layer dalam.
   Hasilnya model jauh lebih tahan terhadap perubahan domain, dan
   yang penting: TIDAK ADA tambahan biaya inference sama sekali,
   karena IN dan BN sama-sama cuma normalisasi affine.

   Buat kasus GUIDIO, "domain" itu berarti: siang vs malam, kering vs
   hujan, paving block vs beton vs aspal, HP bagus vs HP murah.
   Semuanya variasi gaya, bukan variasi bentuk. Persis yang IBN tangani.

2. STEM YANG LEBIH KUAT
   Versi lama: dua conv stride-2 berturut-turut langsung dari 3 channel.
   Itu membuang terlalu banyak informasi spasial sebelum sempat
   mengekstrak fitur apa pun. Tepi trotoar yang tipis hilang di situ.
   Versi baru menaikkan channel dulu sebelum downsample kedua.

3. CABANG I LEBIH DALAM + PAPPM
   Konteks luas penting untuk membedakan "trotoar retak" (aman) dari
   "lubang" (bahaya), karena keduanya mirip kalau dilihat lokal saja.
   Ditambahkan modul pyramid pooling ringan (PAPPM-style) di ujung
   cabang I.

4. BAG FUSION
   PIDNet asli memakai modul Bag (Boundary-attention-guided) untuk
   menggabungkan cabang P dan I dengan panduan cabang D. Versi lama
   cuma concat lalu conv, yang membuang fungsi cabang D. Sekarang
   fusion-nya dipandu boundary sungguhan.

5. DEEP SUPERVISION
   Head tambahan di cabang I saat training (dibuang saat inference)
   untuk mempercepat konvergensi.

CATATAN JUJUR
-------------
Ini implementasi terinspirasi PIDNet, bukan port persis dari repo
resmi. Kalau kamu butuh reproduksi angka paper, pakai repo resmi
(github.com/XuJiacong/PIDNet) plus bobot pretrained ImageNet-nya.
Implementasi ini dirancang supaya bisa dilatih dari nol dengan dataset
seukuran punyamu (~2.000 gambar) dan tetap ringan untuk on-device.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1


# ─── Blok dasar ────────────────────────────────────────────────────────────────

class IBNorm(nn.Module):
    """
    Instance-Batch Normalization.

    Channel dibagi dua: sebagian lewat InstanceNorm (buang info gaya),
    sisanya lewat BatchNorm (jaga info diskriminatif).

    Args:
        ratio: porsi channel yang pakai InstanceNorm. 0.5 adalah nilai
               dari paper IBN-Net dan bekerja baik di sebagian besar kasus.
               Naikkan ke 0.7 kalau domain shift sangat parah dan kamu
               rela kehilangan sedikit akurasi in-domain.
    """

    def __init__(self, channels: int, ratio: float = 0.5):
        super().__init__()
        self.half = int(channels * ratio)
        self.rest = channels - self.half
        if self.half > 0:
            self.instance = nn.InstanceNorm2d(self.half, affine=True)
        if self.rest > 0:
            self.batch = nn.BatchNorm2d(self.rest, momentum=BN_MOMENTUM)

    def forward(self, x):
        if self.half == 0:
            return self.batch(x)
        if self.rest == 0:
            return self.instance(x)
        split = torch.split(x, [self.half, self.rest], dim=1)
        return torch.cat([self.instance(split[0].contiguous()),
                          self.batch(split[1].contiguous())], dim=1)


def norm_layer(channels: int, use_ibn: bool, ibn_ratio: float = 0.5):
    return IBNorm(channels, ibn_ratio) if use_ibn else \
        nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)


def conv_norm_relu(in_ch, out_ch, k=3, s=1, p=1, use_ibn=False,
                   ibn_ratio=0.5, relu=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
              norm_layer(out_ch, use_ibn, ibn_ratio)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    """Residual block ala ResNet, dengan opsi IBN."""

    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, use_ibn=False,
                 ibn_ratio=0.5, no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.norm1 = norm_layer(out_ch, use_ibn, ibn_ratio)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.no_relu = no_relu

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + identity
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck block untuk ujung cabang I (lebih hemat parameter)."""

    expansion = 2

    def __init__(self, in_ch, mid_ch, stride=1, no_relu=True):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.no_relu = no_relu

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + identity
        return out if self.no_relu else self.relu(out)


# ─── Modul konteks ─────────────────────────────────────────────────────────────

class PyramidPooling(nn.Module):
    """
    Pyramid pooling ringan (semangat PAPPM).

    Menggabungkan konteks pada beberapa skala termasuk global average
    pooling. Ini yang memberi model kemampuan menjawab pertanyaan
    "apakah bentuk gelap ini lubang atau cuma bayangan?", karena
    jawabannya bergantung pada konteks seluruh scene, bukan tekstur lokal.

    Catatan ONNX/TFLite: AdaptiveAvgPool2d dengan output_size=1 diekspor
    dengan baik. Ukuran pooling lain memakai AvgPool2d biasa dengan
    kernel dihitung dari ukuran input supaya graph-nya statis dan aman
    saat konversi ke TFLite.
    """

    def __init__(self, in_ch, mid_ch, out_ch, scales=(2, 4, 8)):
        super().__init__()
        self.scales = scales

        self.scale_convs = nn.ModuleList([
            conv_norm_relu(in_ch, mid_ch, k=1, p=0) for _ in scales
        ])
        self.global_conv = conv_norm_relu(in_ch, mid_ch, k=1, p=0)
        self.local_conv = conv_norm_relu(in_ch, mid_ch, k=1, p=0)

        n_branch = len(scales) + 2
        self.fuse = conv_norm_relu(mid_ch * n_branch, out_ch, k=1, p=0)

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [self.local_conv(x)]

        for scale, conv in zip(self.scales, self.scale_convs):
            kh = max(1, h // scale)
            kw = max(1, w // scale)
            pooled = F.avg_pool2d(x, kernel_size=(kh, kw),
                                  stride=(kh, kw), ceil_mode=True)
            pooled = conv(pooled)
            feats.append(F.interpolate(pooled, size=(h, w), mode="bilinear",
                                       align_corners=False))

        g = F.adaptive_avg_pool2d(x, 1)
        g = self.global_conv(g)
        feats.append(g.expand(-1, -1, h, w))

        return self.fuse(torch.cat(feats, dim=1))


class BagFusion(nn.Module):
    """
    Boundary-attention-guided fusion.

    Cabang D memprediksi di mana batas antar-zona berada. Di dekat batas,
    detail resolusi tinggi (cabang P) lebih dipercaya. Jauh dari batas,
    konteks resolusi rendah (cabang I) lebih dipercaya.

    Rumusnya: out = sigma(D) * P + (1 - sigma(D)) * I

    Ini yang membuat cabang D punya peran nyata. Di versi lama, ketiga
    cabang cuma di-concat lalu di-conv, jadi cabang D tidak pernah
    benar-benar "memandu" apa pun.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv_norm_relu(in_ch, out_ch, k=3, p=1)

    def forward(self, p, i, d):
        attn = torch.sigmoid(d)
        fused = attn * p + (1.0 - attn) * i
        return self.conv(fused)


# ─── Model utama ───────────────────────────────────────────────────────────────

class PIDNetS(nn.Module):
    """
    PIDNet-S untuk segmentasi jalur 3 zona.

    Input : (B, 3, H, W), dinormalisasi dengan mean/std ImageNet
    Output:
        training  -> (logits, boundary, aux_logits)
        inference -> logits saja, (B, num_classes, H, W)

    Args:
        num_classes: jumlah kelas semantik (3: non_walkable/walkable/hazard)
        base_ch: lebar dasar. 32 = ringan (~2 juta parameter),
                 64 = lebih akurat tapi ~4x parameter
        use_ibn: aktifkan IBN untuk domain generalization
        ibn_ratio: porsi channel InstanceNorm di layer dangkal
    """

    def __init__(self, num_classes: int = 3, base_ch: int = 32,
                 use_ibn: bool = True, ibn_ratio: float = 0.5,
                 deep_supervision: bool = True):
        super().__init__()
        c = base_ch
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # ── Stem: /4 ──
        # IBN dipakai di sini karena layer dangkal yang paling banyak
        # menangkap variasi gaya (cahaya, warna, kontras).
        self.stem = nn.Sequential(
            conv_norm_relu(3, c, k=3, s=2, p=1, use_ibn=use_ibn,
                           ibn_ratio=ibn_ratio),
            conv_norm_relu(c, c, k=3, s=1, p=1, use_ibn=use_ibn,
                           ibn_ratio=ibn_ratio),
            conv_norm_relu(c, c * 2, k=3, s=2, p=1, use_ibn=use_ibn,
                           ibn_ratio=ibn_ratio),
            BasicBlock(c * 2, c * 2, use_ibn=use_ibn, ibn_ratio=ibn_ratio),
        )

        # ── Cabang P (detail, tetap /8) ──
        self.p_down = conv_norm_relu(c * 2, c * 2, k=3, s=2, p=1,
                                     use_ibn=use_ibn, ibn_ratio=ibn_ratio)
        self.p_branch = nn.Sequential(
            BasicBlock(c * 2, c * 2, use_ibn=use_ibn, ibn_ratio=ibn_ratio),
            BasicBlock(c * 2, c * 2),
        )

        # ── Cabang I (konteks, turun ke /32) ──
        self.i_stage1 = nn.Sequential(
            BasicBlock(c * 2, c * 4, stride=2),   # /8
            BasicBlock(c * 4, c * 4),
        )
        self.i_stage2 = nn.Sequential(
            BasicBlock(c * 4, c * 8, stride=2),   # /16
            BasicBlock(c * 8, c * 8),
        )
        self.i_stage3 = nn.Sequential(
            Bottleneck(c * 8, c * 8, stride=2),   # /32
        )
        self.pappm = PyramidPooling(c * 16, c * 4, c * 2)
        self.i_proj = conv_norm_relu(c * 2, c * 2, k=1, p=0)

        # ── Cabang D (boundary) ──
        self.d_branch = nn.Sequential(
            BasicBlock(c * 2, c * 2, use_ibn=use_ibn, ibn_ratio=ibn_ratio),
            conv_norm_relu(c * 2, c * 2),
        )
        self.d_head = nn.Sequential(
            conv_norm_relu(c * 2, c, k=3, p=1),
            nn.Conv2d(c, 1, 1),
        )

        # ── Fusion & head ──
        self.bag = BagFusion(c * 2, c * 4)
        self.cls_head = nn.Sequential(
            conv_norm_relu(c * 4, c * 4, k=3, p=1),
            nn.Conv2d(c * 4, num_classes, 1),
        )

        if deep_supervision:
            self.aux_head = nn.Sequential(
                conv_norm_relu(c * 8, c * 2, k=3, p=1),
                nn.Conv2d(c * 2, num_classes, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_aux: bool = False):
        h, w = x.shape[2:]

        s = self.stem(x)                       # /4,  c*2

        # Cabang P & D berbagi downsample yang sama, lalu bercabang.
        # Menghitung self.p_down(s) dua kali (seperti draft awal) berarti
        # menjalankan conv yang sama dua kali untuk hasil identik: buang
        # waktu inference tanpa manfaat apa pun.
        shared = self.p_down(s)                # /8,  c*2
        p = self.p_branch(shared)              # /8,  c*2
        d = self.d_branch(shared)              # /8,  c*2
        boundary_logit = self.d_head(d)        # /8,  1

        # Cabang I turun jauh untuk konteks
        i1 = self.i_stage1(s)                  # /8,  c*4
        i2 = self.i_stage2(i1)                 # /16, c*8
        i3 = self.i_stage3(i2)                 # /32, c*16
        ctx = self.pappm(i3)                   # /32, c*2
        ctx = F.interpolate(ctx, size=p.shape[2:], mode="bilinear",
                            align_corners=False)
        ctx = self.i_proj(ctx)                 # /8,  c*2

        # Fusion dipandu boundary
        fused = self.bag(p, ctx, d)            # /8,  c*4
        logits = self.cls_head(fused)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear",
                               align_corners=False)

        if not return_aux:
            return logits

        boundary = F.interpolate(boundary_logit, size=(h, w),
                                 mode="bilinear", align_corners=False)

        if self.deep_supervision:
            aux = self.aux_head(i2)
            aux = F.interpolate(aux, size=(h, w), mode="bilinear",
                                align_corners=False)
            return logits, boundary, aux

        return logits, boundary, None


# ─── Utilitas ──────────────────────────────────────────────────────────────────

class InferenceWrapper(nn.Module):
    """
    Bungkus model supaya SELALU mengembalikan satu tensor.

    Ini wajib untuk ekspor ONNX/TFLite: tracer tidak bisa menangani
    `return_aux` yang bercabang, dan kalau dipaksa, hasilnya graph yang
    salah tanpa error apa pun.
    """

    def __init__(self, model: nn.Module, apply_softmax: bool = False):
        super().__init__()
        self.model = model
        self.apply_softmax = apply_softmax

    def forward(self, x):
        out = self.model(x, return_aux=False)
        if self.apply_softmax:
            out = torch.softmax(out, dim=1)
        return out


def count_parameters(model: nn.Module) -> tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    return n, n / 1e6


if __name__ == "__main__":
    for ibn in (False, True):
        m = PIDNetS(num_classes=3, base_ch=32, use_ibn=ibn)
        m.eval()
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = m(x)
            logits, bnd, aux = m(x, return_aux=True)
        n, mb = count_parameters(m)
        print(f"use_ibn={ibn}: out={tuple(out.shape)} "
              f"bnd={tuple(bnd.shape)} "
              f"aux={tuple(aux.shape) if aux is not None else None} "
              f"params={mb:.2f}M")
