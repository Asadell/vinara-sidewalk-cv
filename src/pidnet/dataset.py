"""
src/pidnet/dataset.py
=====================
Dataset loader segmentasi jalur dengan augmentasi outdoor tersinkron.

TEMUAN PENTING DARI KODE LAMA
-----------------------------
Docstring `06_train_pidnet.py` versi lama menyatakan:

    "Synchronized Albumentations (Flip, Lighting Jitter, Motion Blur)
     untuk memperbanyak variasi kondisi trotoar Indonesia"

Tapi kode `dataset.py` sebenarnya cuma melakukan ini:

    if self.augment and np.random.rand() < 0.5:
        img  = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

Cuma horizontal flip. Tidak ada lighting jitter, tidak ada motion blur,
tidak ada Albumentations sama sekali. Dokumentasi dan kode tidak sinkron.
Ini layak dicek ulang, karena kalau kamu mengevaluasi hasil training
sambil mengira augmentasinya sudah kaya, kesimpulanmu soal "kenapa model
jeblok saat hujan" bakal salah arah.

Modul ini mengimplementasikan yang dijanjikan docstring itu, plus:

  - Resize yang MENJAGA ASPECT RATIO (versi lama squash ke persegi,
    yang mendistorsi geometri trotoar; padahal bentuk trapesium
    perspektif trotoar itu justru sinyal utama untuk zona kiri/tengah/kanan)
  - Random scale + crop (standar untuk segmentation, bukan resize penuh)
  - Augmentasi cuaca, bayangan, malam yang tersinkron antara gambar & mask
  - Copy-paste hazard untuk menambah kelas minoritas
  - ignore_index untuk piksel padding, supaya area padding tidak
    dihitung sebagai kelas apa pun saat loss & metrik

CATATAN SINKRONISASI
--------------------
Transform geometris HARUS diterapkan ke gambar DAN mask bersamaan,
dengan interpolasi berbeda: bilinear untuk gambar, NEAREST untuk mask.
Kalau mask ikut bilinear, label 0 dan 2 bisa menghasilkan 1 di tepi,
yang artinya "jalan raya bertemu hazard" jadi "trotoar". Persis
kesalahan yang paling berbahaya. Albumentations menangani ini otomatis
lewat argumen `mask=`, asal transform-nya memang mendukung mask.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    A = None
    HAS_ALBUMENTATIONS = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IGNORE_INDEX = 255
SEG_NON_WALKABLE, SEG_WALKABLE, SEG_HAZARD = 0, 1, 2

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─── Augmentasi ────────────────────────────────────────────────────────────────

def build_seg_train_transform(img_h: int, img_w: int,
                              strength: str = "medium",
                              scale_range=(0.5, 1.6),
                              resize_mode: str = "stretch"):
    """
    Pipeline augmentasi training untuk segmentasi.

    Urutan sengaja: geometri dulu (di resolusi asli, supaya distorsinya
    natural), baru fotometrik, baru degradasi optik. Kalau blur diterapkan
    sebelum resize, kekuatan blur-nya jadi salah skala.

    BENTUK OUTPUT SEKARANG (img_h, img_w), BUKAN PERSEGI.
    Alasannya ada di catatan panjang di build_seg_eval_transform().
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations belum terinstall.")

    presets = {
        "light":  dict(weather=0.20, shadow=0.25, night=0.10, optic=0.25),
        "medium": dict(weather=0.35, shadow=0.40, night=0.20, optic=0.40),
        "heavy":  dict(weather=0.50, shadow=0.55, night=0.30, optic=0.55),
    }
    p = presets[strength]

    return A.Compose([
        # ── Geometri (tersinkron gambar + mask) ───────────────────────────
        A.HorizontalFlip(p=0.5),
        # JANGAN VerticalFlip: langit di bawah dan trotoar di atas tidak
        # pernah terjadi, dan itu merusak asumsi perspektif model.
        A.Affine(
            scale=scale_range,
            rotate=(-8, 8),          # goyangan langkah kaki, bukan rotasi besar
            translate_percent={"x": (-0.08, 0.08), "y": (-0.06, 0.06)},
            shear={"x": (-4, 4), "y": (-2, 2)},
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=IGNORE_INDEX,
            p=0.8,
        ),
        A.Perspective(
            scale=(0.02, 0.07),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=IGNORE_INDEX,
            p=0.3,
        ),
        # Bentuk akhir dibuat SAMA dengan yang dipakai app.
        #   stretch   : resize paksa ke (img_h, img_w), aspect ratio berubah.
        #               Ini yang dilakukan NavFrameConverter di Flutter.
        #   letterbox : jaga aspect ratio lalu pad. Lebih "benar" secara
        #               geometris, tapi TIDAK cocok dengan app saat ini.
        *(
            [A.Resize(height=img_h, width=img_w,
                      interpolation=cv2.INTER_LINEAR,
                      mask_interpolation=cv2.INTER_NEAREST)]
            if resize_mode == "stretch" else
            [A.LongestMaxSize(max_size_hw=(img_h, img_w),
                              interpolation=cv2.INTER_LINEAR,
                              mask_interpolation=cv2.INTER_NEAREST),
             A.PadIfNeeded(min_height=img_h, min_width=img_w,
                           border_mode=cv2.BORDER_CONSTANT,
                           fill=0, fill_mask=IGNORE_INDEX),
             A.RandomCrop(height=img_h, width=img_w)]
        ),

        # ── Fotometrik ────────────────────────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=(-0.32, 0.28),
                                   contrast_limit=(-0.28, 0.28), p=0.6),
        A.RandomGamma(gamma_limit=(60, 150), p=0.4),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25,
                             val_shift_limit=20, p=0.4),
        A.RandomToneCurve(scale=0.20, p=0.25),

        # ── Bayangan & cuaca ──────────────────────────────────────────────
        A.RandomShadow(shadow_roi=(0, 0.3, 1, 1),
                       num_shadows_limit=(1, 3),
                       shadow_dimension=5, p=p["shadow"]),
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.85, drop_width=1,
                         blur_value=3, p=1.0),
            A.RandomFog(p=1.0),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.4), src_radius=160, p=1.0),
        ], p=p["weather"]),

        # ── Malam ─────────────────────────────────────────────────────────
        A.Compose([
            A.RandomGamma(gamma_limit=(180, 320), p=1.0),
            A.ISONoise(color_shift=(0.02, 0.06), intensity=(0.3, 0.8), p=0.8),
        ], p=p["night"]),

        # ── Degradasi optik ───────────────────────────────────────────────
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 11), p=1.0),
            A.Defocus(radius=(1, 4), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=p["optic"]),
        A.GaussNoise(std_range=(0.02, 0.09), p=p["optic"] * 0.6),
        A.ImageCompression(quality_range=(35, 92), p=0.3),
    ], p=1.0, seed=None)


def build_seg_eval_transform(img_h: int, img_w: int,
                             resize_mode: str = "stretch"):
    """
    Val/test: bentuk ulang ke (img_h, img_w). Tidak ada augmentasi.

    KENAPA DEFAULTNYA "stretch" DAN BUKAN LAGI PERSEGI
    ---------------------------------------------------
    Docstring versi sebelumnya menulis "Ini harus SAMA PERSIS dengan
    preprocessing di sisi Flutter". Niatnya benar, tapi waktu diperiksa
    ternyata TIDAK sama, dan bedanya ada dua lapis sekaligus:

      training lama : LongestMaxSize + PadIfNeeded  -> PERSEGI 512x512,
                      aspect ratio dijaga, sisanya bar padding
      app (Flutter) : lib/services/nav_frame_converter.dart melakukan
                          sxUpright = tx * uprightW / _pidW
                          syUpright = ty * uprightH / _pidH
                      yaitu resize PAKSA ke 640x384, tanpa padding sama
                      sekali, aspect ratio berubah

    Jadi model dilatih melihat trotoar beraspek asli di tengah kanvas
    persegi berbingkai hitam, lalu di lapangan diberi trotoar yang
    gepeng memenuhi bingkai 640x384. Itu domain shift geometris yang
    tidak akan pernah muncul di angka mIoU validasi.

    Default sekarang mengikuti app. Kalau nanti sisi Dart diubah jadi
    letterbox, pindahkan flag --resize-mode ke "letterbox" supaya
    keduanya tetap sinkron.

    Args:
        img_h, img_w: tinggi & lebar output (app memakai 384 x 640)
        resize_mode: "stretch" (ikut app) atau "letterbox"
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations belum terinstall.")
    if resize_mode == "stretch":
        return A.Compose([
            A.Resize(height=img_h, width=img_w,
                     interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    return A.Compose([
        # max_size_hw, BUKAN max_size. Dengan max_size=max(h, w), gambar
        # 480x640 yang ditarget 384x640 tidak berubah sama sekali (sisi
        # terpanjangnya sudah 640), lalu PadIfNeeded juga diam karena
        # 480 >= 384. Hasilnya 480x640: bentuk salah, tanpa error apa pun.
        A.LongestMaxSize(max_size_hw=(img_h, img_w),
                         interpolation=cv2.INTER_LINEAR,
                         mask_interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=img_h, min_width=img_w,
                      position="center",
                      border_mode=cv2.BORDER_CONSTANT,
                      fill=0, fill_mask=IGNORE_INDEX),
        A.CenterCrop(height=img_h, width=img_w),
    ], p=1.0)


# ─── Copy-paste hazard ─────────────────────────────────────────────────────────

class HazardCopyPaste:
    """
    Tempel region hazard dari gambar lain ke gambar sekarang.

    Kelas hazard (lubang, tangga) biasanya cuma 1-3% dari piksel dataset.
    Augmentasi fotometrik tidak menambah jumlah CONTOH hazard, cuma
    memvariasikan yang sudah ada. Copy-paste menambah contoh sungguhan
    dalam konteks baru, jadi jauh lebih efektif untuk kelas minoritas.

    Implementasi sengaja konservatif:
      - Hanya menempel di area yang di ground truth-nya walkable atau
        non_walkable (tidak menimpa hazard yang sudah ada)
      - Blending tepi supaya tidak ada garis potong tajam yang bisa
        dipelajari model sebagai artefak
      - Skala disesuaikan supaya ukurannya masuk akal
    """

    def __init__(self, source_pool: list[tuple[np.ndarray, np.ndarray]],
                 p: float = 0.4, max_paste: int = 2,
                 scale_range=(0.6, 1.4)):
        self.pool = source_pool
        self.p = p
        self.max_paste = max_paste
        self.scale_range = scale_range

    def __call__(self, img: np.ndarray, mask: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if not self.pool or rng.random() > self.p:
            return img, mask

        h, w = mask.shape[:2]
        n = int(rng.integers(1, self.max_paste + 1))

        for _ in range(n):
            src_img, src_mask = self.pool[int(rng.integers(0, len(self.pool)))]
            haz = (src_mask == SEG_HAZARD).astype(np.uint8)
            if haz.sum() < 64:
                continue

            ys, xs = np.where(haz > 0)
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            patch_img = src_img[y1:y2, x1:x2]
            patch_msk = haz[y1:y2, x1:x2]
            if patch_img.size == 0:
                continue

            scale = float(rng.uniform(*self.scale_range))
            ph = max(8, min(h - 2, int(patch_img.shape[0] * scale)))
            pw = max(8, min(w - 2, int(patch_img.shape[1] * scale)))
            patch_img = cv2.resize(patch_img, (pw, ph),
                                   interpolation=cv2.INTER_LINEAR)
            patch_msk = cv2.resize(patch_msk, (pw, ph),
                                   interpolation=cv2.INTER_NEAREST)

            # Tempel di bagian bawah-tengah frame (tempat hazard biasanya
            # muncul dari sudut pandang pejalan kaki)
            oy = int(rng.integers(int(h * 0.35), max(int(h * 0.35) + 1, h - ph)))
            ox = int(rng.integers(0, max(1, w - pw)))

            region_mask = mask[oy:oy + ph, ox:ox + pw]
            if region_mask.shape[:2] != patch_msk.shape[:2]:
                continue

            # Jangan timpa hazard yang sudah ada atau area ignore
            allowed = (region_mask != SEG_HAZARD) & (region_mask != IGNORE_INDEX)
            paste = (patch_msk > 0) & allowed
            if paste.sum() < 32:
                continue

            # Blending tepi lembut supaya tidak ada garis potong tajam
            alpha = cv2.GaussianBlur(paste.astype(np.float32), (0, 0),
                                     sigmaX=1.5)[..., None]
            region_img = img[oy:oy + ph, ox:ox + pw].astype(np.float32)
            img[oy:oy + ph, ox:ox + pw] = (
                region_img * (1 - alpha) + patch_img.astype(np.float32) * alpha
            ).astype(np.uint8)

            region_mask[paste] = SEG_HAZARD

        return img, mask


# ─── Dataset ───────────────────────────────────────────────────────────────────

class SidewalkSegDataset(Dataset):
    """
    Dataset segmentasi jalur.

    Struktur folder yang diharapkan:
        <root>/images/{train,valid,test}/*.jpg
        <root>/masks/{train,valid,test}/*.png   (nilai piksel 0/1/2)

    Args:
        root: folder dataset_master_seg
        split: "train" | "valid" | "test"
        img_size: (tinggi, lebar) output. int juga diterima -> persegi.
        resize_mode: "stretch" (cocok dengan app) atau "letterbox"
        augment: aktifkan augmentasi (otomatis mati kecuali split=train)
        aug_strength: "light" | "medium" | "heavy"
        copy_paste_p: probabilitas copy-paste hazard (0 = matikan)
        cache_pool_size: jumlah gambar hazard yang di-cache untuk copy-paste
    """

    def __init__(self, root: str, split: str = "train",
                 img_size: int | tuple[int, int] = (384, 640),
                 augment: bool = False, aug_strength: str = "medium",
                 copy_paste_p: float = 0.0, cache_pool_size: int = 120,
                 scale_range=(0.5, 1.6), resize_mode: str = "stretch"):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_h, self.img_w = int(img_size[0]), int(img_size[1])
        self.img_size = (self.img_h, self.img_w)
        self.resize_mode = resize_mode
        self.augment = augment and split == "train"

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Folder gambar tidak ada: {self.img_dir}\n"
                f"Jalankan scripts/04_convert_coco_segmentation_to_mask.py dulu."
            )

        self.samples = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.samples:
            raise FileNotFoundError(f"Tidak ada gambar di {self.img_dir}")

        # Verifikasi mask ada untuk tiap gambar
        missing = [p.name for p in self.samples[:50]
                   if not self._mask_path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"Mask tidak ditemukan untuk: {missing[:5]}\n"
                f"Dicari di: {self.mask_dir}"
            )

        if self.augment:
            self.transform = build_seg_train_transform(
                self.img_h, self.img_w, aug_strength, scale_range,
                resize_mode)
        else:
            self.transform = build_seg_eval_transform(
                self.img_h, self.img_w, resize_mode)

        self.copy_paste = None
        if self.augment and copy_paste_p > 0:
            pool = self._build_hazard_pool(cache_pool_size)
            if pool:
                self.copy_paste = HazardCopyPaste(pool, p=copy_paste_p)
                print(f"  [dataset] copy-paste hazard aktif "
                      f"({len(pool)} gambar sumber, p={copy_paste_p})")
            else:
                print("  [dataset] tidak ada gambar dengan piksel hazard; "
                      "copy-paste dilewati.")

    def _mask_path(self, img_path: Path) -> Path:
        return self.mask_dir / f"{img_path.stem}.png"

    def _build_hazard_pool(self, limit: int):
        """Kumpulkan gambar yang punya cukup piksel hazard untuk dijadikan sumber."""
        pool = []
        for p in self.samples:
            if len(pool) >= limit:
                break
            mp = self._mask_path(p)
            if not mp.exists():
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if (mask == SEG_HAZARD).sum() < 400:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Simpan versi kecil supaya hemat RAM
            h, w = mask.shape[:2]
            if max(h, w) > 640:
                s = 640 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)),
                                 interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (int(w * s), int(h * s)),
                                  interpolation=cv2.INTER_NEAREST)
            pool.append((img, mask))
        return pool

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        mask_path = self._mask_path(img_path)

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Gagal baca gambar: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Gagal baca mask: {mask_path}")

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        if self.copy_paste is not None:
            rng = np.random.default_rng()
            img, mask = self.copy_paste(img, mask, rng)

        out = self.transform(image=img, mask=mask)
        img, mask = out["image"], out["mask"]

        img_np = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask.astype(np.int64)).long()

        return img_t, mask_t


# ─── Preview ───────────────────────────────────────────────────────────────────

ZONE_COLORS = np.array([
    [80, 80, 80],     # 0 non_walkable - abu
    [40, 200, 90],    # 1 walkable     - hijau
    [230, 60, 60],    # 2 hazard       - merah
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Ubah mask label jadi gambar berwarna untuk inspeksi visual."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls in range(len(ZONE_COLORS)):
        out[mask == cls] = ZONE_COLORS[cls]
    out[mask == IGNORE_INDEX] = (0, 0, 0)
    return out


def save_dataset_preview(dataset: SidewalkSegDataset, out_path: str,
                         n: int = 6) -> None:
    """
    Simpan grid gambar + overlay mask hasil augmentasi.

    Jalankan ini SEBELUM training panjang. Kalau mask dan gambar tidak
    tersinkron (misal gambar ke-flip tapi mask tidak), itu langsung
    kelihatan di sini, sementara di angka loss cuma terlihat sebagai
    "training kok tidak konvergen".
    """
    rows = []
    for i in range(min(n, len(dataset))):
        img_t, mask_t = dataset[i]
        img = img_t.numpy().transpose(1, 2, 0)
        img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        mask = mask_t.numpy().astype(np.uint8)
        color = colorize_mask(mask)
        overlay = cv2.addWeighted(img, 0.6, color, 0.4, 0)

        rows.append(np.hstack([img, color, overlay]))

    grid = np.vstack(rows)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Preview dataset tersimpan: {out_path}")
    print("  Kolom: gambar | mask | overlay. "
          "Pastikan mask benar-benar menempel pada objek yang tepat.")
