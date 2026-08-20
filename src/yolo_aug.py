"""
src/yolo_aug.py
===============
Augmentasi outdoor untuk YOLO11n: bayangan, hujan, kabut, malam, silau,
motion blur, dan degradasi sensor HP.

MASALAH YANG DIPECAHKAN
-----------------------
Versi lama cuma mengandalkan augmentasi bawaan Ultralytics:

    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=10.0, ...

HSV jitter itu menggeser warna & kecerahan SECARA SERAGAM di seluruh
gambar. Masalah nyata di lapangan tidak seperti itu:

  - Bayangan pohon jam 4 sore -> gelap LOKAL berbentuk tidak beraturan,
    tepinya tajam. Lubang yang tertutup bayangan praktis hilang kontras.
  - Trotoar basah setelah hujan -> refleksi terang LOKAL, specular
    highlight yang bisa menyerupai tepi lubang.
  - Malam dengan lampu jalan -> gelap global TAPI ada kerucut terang di
    bawah tiap lampu, plus noise sensor tinggi, plus pergeseran suhu
    warna ke oranye.
  - Jalan sambil menggenggam HP -> motion blur berarah, bukan blur merata.

Tidak satu pun bisa disimulasikan oleh HSV jitter. Makanya modul ini ada.

CARA KERJA
----------
Ultralytics punya kelas `ultralytics.data.augment.Albumentations` yang
dipanggil di dalam pipeline `v8_transforms`. Modul ini menambal
(`monkeypatch`) konstruktor kelas itu supaya memakai daftar transform
kita.

Yang di-inject SENGAJA HANYA transform image-only (fotometrik &
degradasi). Alasannya penting:

  - Transform image-only tidak mengubah posisi objek, jadi bounding box
    tidak perlu disesuaikan. Nol risiko label rusak.
  - Transform geometris (mosaic, scale, rotate, perspective, translate)
    SUDAH ditangani Ultralytics dengan benar termasuk update bbox-nya.
    Menduplikasi itu di Albumentations justru rawan bikin bbox meleset.

Jadi pembagian tugasnya: Ultralytics urus geometri, modul ini urus cahaya
dan degradasi.

PERINGATAN VERSI
----------------
Monkeypatch bergantung pada struktur internal Ultralytics yang bisa
berubah antar versi. Modul ini memverifikasi dulu bahwa atribut yang
dibutuhkan ada; kalau tidak cocok, patch dibatalkan dengan pesan jelas
dan training tetap jalan (pakai augmentasi bawaan). Lebih baik kehilangan
augmentasi tambahan daripada diam-diam merusak label.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    A = None
    HAS_ALBUMENTATIONS = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Transform kustom: simulasi malam & permukaan basah
# ═══════════════════════════════════════════════════════════════════════════════

class NightSimulation(A.ImageOnlyTransform if HAS_ALBUMENTATIONS else object):
    """
    Simulasi malam hari dengan lampu jalan.

    Tiga komponen yang harus ada bareng supaya realistis:
      1. Gamma darkening global (bukan sekadar kurangi brightness linear;
         mata & sensor merespons cahaya secara non-linear)
      2. Kerucut cahaya lampu jalan (beberapa blob terang lembut)
      3. Noise sensor tinggi + pergeseran suhu warna ke oranye
         (lampu natrium jalanan Indonesia rata-rata sangat oranye)

    Kalau cuma digelapkan tanpa noise, model belajar "malam = gelap bersih"
    yang tidak pernah terjadi di HP mid-low.
    """

    def __init__(self, gamma_range=(1.8, 3.6), n_lights=(0, 3),
                 noise_sigma=(6, 20), warm_shift=(5, 25), p=0.5):
        if HAS_ALBUMENTATIONS:
            super().__init__(p=p)
        self.gamma_range = gamma_range
        self.n_lights = n_lights
        self.noise_sigma = noise_sigma
        self.warm_shift = warm_shift

    def apply(self, img, **params):
        rng = np.random.default_rng()
        h, w = img.shape[:2]
        out = img.astype(np.float32) / 255.0

        # 1. Gamma darkening
        gamma = float(rng.uniform(*self.gamma_range))
        out = np.power(out, gamma)

        # 2. Kerucut lampu jalan
        n = int(rng.integers(self.n_lights[0], self.n_lights[1] + 1))
        if n > 0:
            glow = np.zeros((h, w), dtype=np.float32)
            for _ in range(n):
                cx = float(rng.uniform(0, w))
                cy = float(rng.uniform(0, h * 0.6))  # lampu di bagian atas
                radius = float(rng.uniform(w * 0.15, w * 0.55))
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                d2 = (xx - cx) ** 2 + (yy - cy) ** 2
                glow += np.exp(-d2 / (2 * radius ** 2))
            glow = np.clip(glow, 0, 1.5)
            strength = float(rng.uniform(0.25, 0.7))
            out = out + glow[..., None] * strength * (1.0 - out)

        out = np.clip(out, 0, 1) * 255.0

        # 3. Pergeseran suhu warna ke oranye (BGR: naikkan R, turunkan B)
        shift = float(rng.uniform(*self.warm_shift))
        out[..., 2] = np.clip(out[..., 2] + shift, 0, 255)        # R
        out[..., 0] = np.clip(out[..., 0] - shift * 0.6, 0, 255)  # B

        # 4. Noise sensor (naik drastis di ISO tinggi)
        sigma = float(rng.uniform(*self.noise_sigma))
        out = out + rng.normal(0, sigma, out.shape)

        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("gamma_range", "n_lights", "noise_sigma", "warm_shift")


class WetSurface(A.ImageOnlyTransform if HAS_ALBUMENTATIONS else object):
    """
    Simulasi permukaan basah / reflektif setelah hujan.

    Ini penting karena genangan air punya sifat visual yang membingungkan:
    warnanya gelap seperti lubang, tapi punya specular highlight terang.
    Model yang tidak pernah lihat ini akan sering salah, ke dua arah:
    genangan disangka lubang (false positive) atau lubang berisi air
    disangka genangan biasa (false negative, yang lebih berbahaya).
    """

    def __init__(self, n_puddles=(1, 4), darkness=(0.25, 0.55),
                 specular=(0.3, 0.8), p=0.35):
        if HAS_ALBUMENTATIONS:
            super().__init__(p=p)
        self.n_puddles = n_puddles
        self.darkness = darkness
        self.specular = specular

    def apply(self, img, **params):
        rng = np.random.default_rng()
        h, w = img.shape[:2]
        out = img.astype(np.float32)

        n = int(rng.integers(self.n_puddles[0], self.n_puddles[1] + 1))
        for _ in range(n):
            # Genangan cenderung di bagian bawah frame (dekat kaki)
            cy = int(rng.uniform(h * 0.45, h))
            cx = int(rng.uniform(0, w))
            ax = int(rng.uniform(w * 0.08, w * 0.35))
            ay = int(rng.uniform(h * 0.03, h * 0.15))
            angle = float(rng.uniform(0, 180))

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
            mask_f = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, ax / 8.0))
            mask_f = (mask_f.astype(np.float32) / 255.0)[..., None]

            # Gelapkan area genangan
            dark = float(rng.uniform(*self.darkness))
            out = out * (1.0 - mask_f * dark)

            # Tambah specular highlight (pantulan langit)
            spec = np.zeros((h, w), dtype=np.float32)
            sx = cx + int(rng.uniform(-ax * 0.4, ax * 0.4))
            sy = cy + int(rng.uniform(-ay * 0.4, ay * 0.4))
            cv2.ellipse(spec, (sx, sy), (max(2, ax // 3), max(1, ay // 3)),
                        angle, 0, 360, 1.0, -1)
            spec = cv2.GaussianBlur(spec, (0, 0), sigmaX=max(2.0, ax / 10.0))
            strength = float(rng.uniform(*self.specular))
            out = out + (spec * mask_f[..., 0] * strength * 255.0)[..., None]

        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("n_puddles", "darkness", "specular")


class HardShadow(A.ImageOnlyTransform if HAS_ALBUMENTATIONS else object):
    """
    Bayangan bertepi tajam dari pohon, tiang, atau gedung.

    `A.RandomShadow` bawaan Albumentations menghasilkan poligon bayangan
    yang cukup lembut dan seragam. Bayangan sore hari di Indonesia jauh
    lebih kontras dan tepinya tegas, sering berpola belang-belang karena
    daun. Transform ini menambah dua mode itu.
    """

    def __init__(self, strength=(0.35, 0.7), mode_dappled_p=0.4, p=0.4):
        if HAS_ALBUMENTATIONS:
            super().__init__(p=p)
        self.strength = strength
        self.mode_dappled_p = mode_dappled_p

    def apply(self, img, **params):
        rng = np.random.default_rng()
        h, w = img.shape[:2]
        out = img.astype(np.float32)

        if rng.random() < self.mode_dappled_p:
            # Bayangan belang-belang daun: noise low-frequency di-threshold
            noise = rng.random((h // 8 + 1, w // 8 + 1)).astype(np.float32)
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=w / 60.0)
            thresh = float(rng.uniform(0.42, 0.58))
            mask = (noise < thresh).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, w / 250.0))
        else:
            # Bayangan poligon bertepi tajam
            n_pts = int(rng.integers(3, 6))
            pts = np.stack([
                rng.integers(-w // 4, w + w // 4, size=n_pts),
                rng.integers(-h // 4, h + h // 4, size=n_pts),
            ], axis=1).astype(np.int32)
            mask_u8 = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_u8, [pts], 255)
            mask = mask_u8.astype(np.float32) / 255.0
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, w / 400.0))

        s = float(rng.uniform(*self.strength))
        out = out * (1.0 - mask[..., None] * s)

        # Bayangan juga menurunkan saturasi sedikit (cahaya ambient lebih biru)
        blue_shift = mask[..., None] * s * 12.0
        out[..., 0] = np.clip(out[..., 0] + blue_shift[..., 0], 0, 255)

        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("strength", "mode_dappled_p")


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def build_outdoor_transforms(strength: str = "medium"):
    """
    Daftar transform image-only untuk navigasi outdoor.

    strength:
        light  -> dataset sudah beragam kondisi
        medium -> DEFAULT
        heavy  -> dataset kamu mayoritas foto siang cerah
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations belum terinstall.")

    presets = {
        "light":  dict(weather=0.25, shadow=0.30, night=0.15, optic=0.25),
        "medium": dict(weather=0.40, shadow=0.45, night=0.25, optic=0.40),
        "heavy":  dict(weather=0.55, shadow=0.60, night=0.35, optic=0.55),
    }
    if strength not in presets:
        raise ValueError(f"strength harus salah satu dari {list(presets)}")
    p = presets[strength]

    return [
        # ── Cahaya & bayangan ─────────────────────────────────────────────
        HardShadow(strength=(0.35, 0.70), p=p["shadow"]),
        A.RandomShadow(shadow_roi=(0, 0.35, 1, 1),
                       num_shadows_limit=(1, 3),
                       shadow_dimension=5, p=p["shadow"] * 0.6),
        A.RandomBrightnessContrast(brightness_limit=(-0.30, 0.28),
                                   contrast_limit=(-0.28, 0.28), p=0.45),
        A.RandomGamma(gamma_limit=(65, 145), p=0.35),
        A.RandomToneCurve(scale=0.20, p=0.25),

        # ── Malam ─────────────────────────────────────────────────────────
        NightSimulation(p=p["night"]),

        # ── Cuaca & permukaan ─────────────────────────────────────────────
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.85, drop_width=1,
                         blur_value=3, p=1.0),
            A.RandomFog(p=1.0),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.45),
                             src_radius=180, p=1.0),
        ], p=p["weather"]),
        WetSurface(p=p["weather"] * 0.7),

        # ── Degradasi optik HP mid-low ────────────────────────────────────
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 11), p=1.0),
            A.Defocus(radius=(1, 4), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=p["optic"]),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.10), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=p["optic"] * 0.7),
        A.ImageCompression(quality_range=(35, 90), p=0.35),
        A.Downscale(scale_range=(0.45, 0.85),
                    interpolation_pair={"downscale": cv2.INTER_AREA,
                                        "upscale": cv2.INTER_LINEAR},
                    p=0.20),
    ]


def patch_ultralytics_albumentations(strength: str = "medium",
                                     verbose: bool = True) -> bool:
    """
    Tambal `ultralytics.data.augment.Albumentations` supaya memakai
    transform outdoor kita.

    Return True kalau patch berhasil, False kalau dilewati.
    Training tetap bisa jalan kalau return False (pakai augmentasi bawaan).
    """
    if not HAS_ALBUMENTATIONS:
        if verbose:
            print("  [aug] albumentations tidak terinstall, patch dilewati.")
        return False

    try:
        from ultralytics.data import augment as ul_augment
    except ImportError:
        if verbose:
            print("  [aug] ultralytics.data.augment tidak ditemukan, "
                  "patch dilewati.")
        return False

    cls = getattr(ul_augment, "Albumentations", None)
    if cls is None:
        if verbose:
            print("  [aug] kelas Albumentations tidak ada di versi Ultralytics "
                  "ini. Patch dilewati.")
        return False

    # Verifikasi bahwa jalur non-spatial memang didukung. Kalau atribut
    # `contains_spatial` tidak dikenali oleh __call__, patch bisa berbahaya.
    import inspect
    try:
        src = inspect.getsource(cls.__call__)
    except (OSError, TypeError):
        src = ""
    if "contains_spatial" not in src:
        if verbose:
            print("  [aug] struktur internal Ultralytics tidak dikenali "
                  "(tidak ada `contains_spatial`). Patch DIBATALKAN demi "
                  "keamanan label. Training lanjut dengan augmentasi bawaan.")
        return False

    transforms = build_outdoor_transforms(strength)

    def patched_init(self, p: float = 1.0):
        self.p = p
        self.contains_spatial = False  # semua transform kita image-only
        self.transform = A.Compose(transforms)

    cls.__init__ = patched_init

    if verbose:
        print(f"  [aug] Augmentasi outdoor aktif (strength={strength}, "
              f"{len(transforms)} transform image-only).")
        print("  [aug] Geometri (mosaic/scale/rotate/perspective) tetap "
              "ditangani Ultralytics.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Preview
# ═══════════════════════════════════════════════════════════════════════════════

def save_augmentation_preview(image_path: str, out_path: str,
                              strength: str = "medium",
                              n_samples: int = 12,
                              grid_cols: int = 4) -> None:
    """
    Simpan grid contoh hasil augmentasi.

    SELALU jalankan ini sebelum training panjang. Augmentasi yang terlalu
    ekstrem atau tidak realistis bisa MENURUNKAN akurasi, dan kamu tidak
    akan tahu penyebabnya kalau cuma melihat angka loss. Lihat gambarnya.
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations belum terinstall.")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Tidak bisa baca gambar: {image_path}")

    tf = A.Compose(build_outdoor_transforms(strength))

    tiles = [img.copy()]
    for _ in range(n_samples - 1):
        tiles.append(tf(image=img)["image"])

    th, tw = 240, int(240 * img.shape[1] / img.shape[0])
    tiles = [cv2.resize(t, (tw, th)) for t in tiles]

    rows = []
    for i in range(0, len(tiles), grid_cols):
        row = tiles[i:i + grid_cols]
        while len(row) < grid_cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)

    cv2.putText(grid, "ASLI", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), grid)
    print(f"  Preview augmentasi tersimpan: {out_path}")
    print("  Periksa: apakah objek masih bisa dikenali mata manusia? "
          "Kalau kamu sendiri tidak bisa lihat lubangnya, model juga tidak.")
