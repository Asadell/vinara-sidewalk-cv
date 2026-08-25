#!/usr/bin/env python3
"""
07_export_tflite.py  (REVISI)
=============================
Ekspor PIDNet-S ke TFLite untuk inference on-device di Flutter.

MASALAH BESAR DI VERSI LAMA
---------------------------
Versi lama memakai:

    from onnx_tf.backend import prepare
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_model_dir)

`onnx-tf` sudah TIDAK DIRAWAT sejak 2023 dan tidak kompatibel dengan
TensorFlow 2.16+ (Keras 3). Selain itu, kalaupun jalan, `onnx-tf`
menyisipkan operator Transpose di mana-mana untuk mengonversi NCHW
(konvensi PyTorch) ke NHWC (konvensi TFLite). Model hasilnya bisa
2-5x lebih lambat dari yang seharusnya, dan itu fatal untuk target
2 FPS di HP mid-low.

Menariknya, docstring versi lama sendiri sudah menulis "onnx-tf sudah
deprecated, jadi pakai pipeline tf.lite.TFLiteConverter.from_saved_model"
tapi kodenya tetap meng-import onnx_tf. Jadi niatnya sudah benar,
implementasinya yang belum menyusul.

SOLUSI
------
Pakai `onnx2tf` (github.com/PINTO0309/onnx2tf), yang dirancang khusus
untuk mengatasi masalah Transpose itu: dia melakukan optimasi tata
letak sehingga model hasilnya benar-benar NHWC-native tanpa transpose
berlebih.

Alur: ONNX -> onnx2tf -> SavedModel -> TFLite (FP16 / INT8)

Kalau `onnx2tf` tidak tersedia, script menawarkan alternatif yang
dijelaskan di bagian akhir output, bukan gagal diam-diam.

Usage:
    # Dari checkpoint langsung (ekspor ONNX dulu secara otomatis)
    python scripts/07_export_tflite.py \
        --checkpoint runs/pidnet/<run>/best.pth --img-size 512

    # Dari ONNX yang sudah ada
    python scripts/07_export_tflite.py --onnx runs/pidnet/<run>/pidnet_s.onnx

    # INT8 dengan kalibrasi dataset asli
    python scripts/07_export_tflite.py \
        --checkpoint runs/pidnet/<run>/best.pth \
        --int8 --calib-data ../dataset_master_seg --calib-samples 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Kalibrasi ─────────────────────────────────────────────────────────────────

def _hw(img_size) -> tuple[int, int]:
    """Terima int, [S], atau [H, W]; kembalikan (H, W)."""
    if isinstance(img_size, int):
        return img_size, img_size
    v = list(img_size)
    return (v[0], v[0]) if len(v) == 1 else (v[0], v[1])


def build_calibration_array(data_root: Path, img_size,
                            n_samples: int,
                            resize_mode: str = "stretch") -> np.ndarray | None:
    """
    Bangun array kalibrasi INT8 dari gambar dataset asli.

    PENTING: preprocessing di sini harus SAMA PERSIS dengan
    src/pidnet/dataset.py DAN dengan sisi Flutter. Kalau kalibrasi melihat
    distribusi yang berbeda dari yang dilihat model saat training, rentang
    quantization-nya meleset dan akurasi INT8 anjlok tanpa sebab terlihat.

    Versi sebelumnya SELALU letterbox ke persegi, padahal
    nav_frame_converter.dart di app melakukan resize paksa ke 640x384
    tanpa padding sama sekali. Sekarang mode-nya mengikuti training.

    onnx2tf mengharapkan data kalibrasi dalam tata letak NHWC.
    """
    try:
        import cv2
    except ImportError:
        print("  opencv tidak terinstall, kalibrasi dilewati.")
        return None

    img_dir = data_root / "images" / "train"
    if not img_dir.exists():
        print(f"  Folder kalibrasi tidak ada: {img_dir}")
        return None

    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        print(f"  Tidak ada gambar di {img_dir}")
        return None

    rng = np.random.default_rng(0)
    rng.shuffle(paths)
    paths = paths[:n_samples]

    batch = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ih, iw = _hw(img_size)
        if resize_mode == "stretch":
            canvas = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_LINEAR)
        else:
            h, w = img.shape[:2]
            scale = min(ih / h, iw / w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((ih, iw, 3), dtype=np.uint8)
            canvas[(ih - nh) // 2:(ih - nh) // 2 + nh,
                   (iw - nw) // 2:(iw - nw) // 2 + nw] = resized

        norm = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        batch.append(norm)

    if not batch:
        return None

    arr = np.stack(batch, axis=0).astype(np.float32)  # NHWC
    print(f"  Data kalibrasi: {arr.shape} "
          f"(rentang [{arr.min():.2f}, {arr.max():.2f}])")
    return arr


# ─── Konversi ──────────────────────────────────────────────────────────────────

def check_onnx2tf() -> bool:
    try:
        import onnx2tf  # noqa: F401
        return True
    except ImportError:
        return False


def convert_with_onnx2tf(onnx_path: Path, out_dir: Path,
                         calib_array: np.ndarray | None,
                         int8: bool, fp16: bool) -> dict:
    """Jalankan onnx2tf. Return dict path hasil."""
    import onnx2tf

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_path = None
    if int8 and calib_array is not None:
        calib_path = out_dir / "calib_data.npy"
        np.save(calib_path, calib_array)

    kwargs = dict(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(out_dir),
        output_signaturedefs=True,
        non_verbose=True,
    )

    if int8 and calib_path is not None:
        kwargs["custom_input_op_name_np_data_path"] = [
            ["input", str(calib_path), 0.0, 1.0]
        ]
    else:
        kwargs["output_integer_quantized_tflite"] = False

    print("  Menjalankan onnx2tf...")
    onnx2tf.convert(**kwargs)

    results = {}
    for f in sorted(out_dir.glob("*.tflite")):
        size_mb = f.stat().st_size / 1024 / 1024
        results[f.name] = {"path": str(f), "size_mb": round(size_mb, 2)}
        print(f"    {f.name:<48} {size_mb:>7.2f} MB")

    return results


def validate_tflite(tflite_path: Path, img_size: int,
                    onnx_path: Path | None = None) -> dict:
    """
    Bandingkan output TFLite vs ONNX pada input yang sama.

    Sama seperti di 07_export_onnx.py: ekspor yang "tidak error" belum
    tentu benar. Konversi NCHW->NHWC adalah tempat paling sering
    terjadinya kesalahan diam-diam.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return {"validated": False, "reason": "tensorflow tidak terinstall"}

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    info = {
        "input_shape": [int(x) for x in inp["shape"]],
        "input_dtype": str(np.dtype(inp["dtype"])),
        "output_shape": [int(x) for x in out["shape"]],
        "output_dtype": str(np.dtype(out["dtype"])),
        "input_quant": [float(inp["quantization"][0]),
                        int(inp["quantization"][1])],
        "output_quant": [float(out["quantization"][0]),
                         int(out["quantization"][1])],
    }
    print(f"    input : {info['input_dtype']} {info['input_shape']}")
    print(f"    output: {info['output_dtype']} {info['output_shape']}")

    # Bandingkan dengan ONNX kalau tersedia
    if onnx_path is None or not onnx_path.exists():
        return {"validated": True, **info, "compared_to_onnx": False}

    try:
        import onnxruntime as ort
    except ImportError:
        return {"validated": True, **info, "compared_to_onnx": False}

    rng = np.random.default_rng(1)
    x_nhwc = rng.standard_normal(
        (1, img_size, img_size, 3)).astype(np.float32)
    x_nchw = x_nhwc.transpose(0, 3, 1, 2).copy()

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["output"], {"input": x_nchw})[0]  # NCHW

    in_dtype = np.dtype(inp["dtype"])
    if in_dtype in (np.int8, np.uint8):
        scale, zero = inp["quantization"]
        x_feed = np.round(x_nhwc / scale + zero).astype(in_dtype)
    else:
        x_feed = x_nhwc.astype(np.float32)

    interp.set_tensor(inp["index"], x_feed)
    interp.invoke()
    tfl_out = interp.get_tensor(out["index"])

    out_dtype = np.dtype(out["dtype"])
    if out_dtype in (np.int8, np.uint8):
        scale, zero = out["quantization"]
        tfl_out = (tfl_out.astype(np.float32) - zero) * scale

    # TFLite output kemungkinan NHWC; samakan ke NCHW
    if tfl_out.ndim == 4 and tfl_out.shape[-1] == onnx_out.shape[1]:
        tfl_out = tfl_out.transpose(0, 3, 1, 2)

    if tfl_out.shape != onnx_out.shape:
        print(f"    Bentuk tidak cocok: tflite={tfl_out.shape} "
              f"onnx={onnx_out.shape}")
        return {"validated": False, **info, "compared_to_onnx": True,
                "reason": "shape mismatch"}

    diff = float(np.abs(tfl_out - onnx_out).max())
    agree = float((tfl_out.argmax(1) == onnx_out.argmax(1)).mean())
    print(f"    selisih maks vs ONNX  : {diff:.4f}")
    print(f"    kecocokan argmax      : {agree * 100:.2f}%")

    ok = agree > 0.98
    if not ok:
        print("    PERINGATAN: kecocokan argmax rendah. Untuk FP16 ini "
              "tidak normal. Untuk INT8, cek apakah data kalibrasi "
              "representatif.")

    return {"validated": ok, **info, "compared_to_onnx": True,
            "max_abs_diff": diff, "argmax_agreement": agree}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ekspor PIDNet-S ke TFLite")
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint .pth (ONNX diekspor otomatis)")
    ap.add_argument("--onnx", default=None,
                    help="Pakai file ONNX yang sudah ada")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--img-size", type=int, nargs="+", default=[384, 640],
                    metavar=("H", "W"),
                    help="Tinggi dan lebar. Satu angka = persegi. "
                         "Default 384 640 mengikuti app.")
    ap.add_argument("--resize-mode", choices=["stretch", "letterbox"],
                    default="stretch",
                    help="HARUS sama dengan yang dipakai saat training.")
    ap.add_argument("--int8", action="store_true",
                    help="Hasilkan varian INT8 (butuh --calib-data)")
    ap.add_argument("--calib-data", default=None,
                    help="Folder dataset_master_seg untuk kalibrasi INT8")
    ap.add_argument("--calib-samples", type=int, default=200)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint and not args.onnx:
        print("Wajib pakai --checkpoint atau --onnx.")
        sys.exit(1)

    print("=" * 66)
    print("  EKSPOR PIDNet-S -> TFLite")
    print("=" * 66)

    # ── Siapkan ONNX ──
    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.exists():
            print(f"ONNX tidak ada: {onnx_path}")
            sys.exit(1)
    else:
        ckpt = Path(args.checkpoint)
        onnx_path = ckpt.parent / "pidnet_s.onnx"
        if not onnx_path.exists():
            print(f"  ONNX belum ada, mengekspor dari checkpoint dulu...")
            cmd = [
                sys.executable, str(ROOT / "scripts" / "07_export_onnx.py"),
                "--checkpoint", str(ckpt),
                "--output", str(onnx_path),
                "--img-size", *[str(v) for v in args.img_size],
                "--simplify",
            ]
            r = subprocess.run(cmd)
            if r.returncode != 0 or not onnx_path.exists():
                print("  Ekspor ONNX gagal. Perbaiki itu dulu.")
                sys.exit(1)
        else:
            print(f"  Memakai ONNX yang sudah ada: {onnx_path}")

    out_dir = Path(args.out_dir) if args.out_dir else \
        onnx_path.parent / "tflite"

    # ── Cek onnx2tf ──
    if not check_onnx2tf():
        print("\n" + "=" * 66)
        print("  onnx2tf TIDAK TERINSTALL")
        print("=" * 66)
        print("  Install dengan:")
        print("    pip install onnx2tf onnx onnx-graphsurgeon "
              "sng4onnx onnxsim tensorflow")
        print()
        print("  Kenapa onnx2tf dan bukan onnx-tf (yang dipakai versi lama):")
        print("    - onnx-tf sudah tidak dirawat sejak 2023 dan tidak")
        print("      kompatibel dengan TensorFlow 2.16+")
        print("    - onnx-tf menyisipkan operator Transpose di mana-mana")
        print("      untuk konversi NCHW->NHWC, yang bisa membuat model")
        print("      2-5x lebih lambat. Fatal untuk target 2 FPS di HP.")
        print()
        print("  Alternatif kalau onnx2tf bermasalah di lingkunganmu:")
        print("    1. ai-edge-torch (Google, PyTorch -> TFLite langsung,")
        print("       melewati ONNX sepenuhnya)")
        print("    2. Jalankan ONNX langsung di Flutter lewat")
        print("       onnxruntime, tanpa TFLite sama sekali. Ini sejalan")
        print("       dengan rencana migrasi ke ONNX Runtime yang sudah")
        print("       kamu sebutkan, dan menghilangkan seluruh kelas")
        print("       masalah konversi ini.")
        sys.exit(1)

    # ── Kalibrasi ──
    calib = None
    if args.int8:
        if not args.calib_data:
            print("\n  --int8 butuh --calib-data. Tanpa kalibrasi dari data "
                  "asli, INT8 akan jauh lebih tidak akurat.")
            sys.exit(1)
        print("\nMenyiapkan data kalibrasi INT8...")
        calib = build_calibration_array(Path(args.calib_data),
                                        args.img_size, args.calib_samples,
                                        args.resize_mode)
        if calib is None:
            print("  Kalibrasi gagal disiapkan.")
            sys.exit(1)

    # ── Konversi ──
    print(f"\nMengonversi ke TFLite (output: {out_dir})...")
    try:
        results = convert_with_onnx2tf(onnx_path, out_dir, calib,
                                       args.int8, fp16=True)
    except Exception as e:
        print(f"  Konversi gagal: {e}")
        print("\n  Saran: coba tanpa --int8 dulu untuk mengisolasi masalah, "
              "atau pertimbangkan onnxruntime di Flutter sebagai gantinya.")
        sys.exit(1)

    if not results:
        print("  Tidak ada file .tflite yang dihasilkan.")
        sys.exit(1)

    # ── Validasi ──
    report = {"onnx": str(onnx_path), "img_size": args.img_size,
              "outputs": results}

    if not args.no_validate:
        print("\nValidasi hasil konversi...")
        for name, meta in results.items():
            print(f"\n  {name}")
            meta["validation"] = validate_tflite(
                Path(meta["path"]), args.img_size, onnx_path)

    with open(out_dir / "export_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Rekomendasi ──
    print("\n" + "=" * 66)
    print("  REKOMENDASI DEPLOY")
    print("=" * 66)
    fp16 = [n for n in results if "float16" in n]
    int8 = [n for n in results if "integer_quant" in n or "int8" in n]
    f32 = [n for n in results if "float32" in n]

    if fp16:
        print(f"  Mobile (seimbang) : {fp16[0]}")
        print("    FP16 memangkas ukuran setengah dengan kehilangan akurasi "
              "yang hampir nol. Ini pilihan default yang aman.")
    if int8:
        print(f"  Mobile (tercepat) : {int8[0]}")
        print("    INT8 paling cepat, tapi WAJIB dicek akurasinya dulu. "
              "Untuk segmentasi, INT8 kadang merusak tepi zona, dan tepi "
              "itu justru yang menentukan kualitas navigasi.")
    if f32:
        print(f"  Referensi         : {f32[0]}")

    print(f"\n  Semua file: {out_dir.resolve()}")
    print("\n  Kontrak preprocessing di Flutter (WAJIB sama dengan training):")
    print("    1. BGR/YUV kamera -> RGB")
    _ih, _iw = _hw(args.img_size)
    if args.resize_mode == "stretch":
        print(f"    2. Resize PAKSA ke {_ih}x{_iw} (HxW), tanpa padding")
        print(f"    3. (tidak ada langkah padding)")
    else:
        print(f"    2. Resize jaga aspect ratio agar muat {_ih}x{_iw}")
        print(f"    3. Pad ke {_ih}x{_iw} di TENGAH, nilai 0")
    print("    4. /255, lalu (x - [0.485,0.456,0.406]) / [0.229,0.224,0.225]")
    print("    5. Tata letak NHWC (TFLite), float32")


if __name__ == "__main__":
    main()
