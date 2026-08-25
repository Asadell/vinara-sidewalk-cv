#!/usr/bin/env python3
"""
07_export_onnx.py  (REVISI)
===========================
Ekspor PIDNet-S ke ONNX untuk backend FastAPI.

PERBAIKAN DARI VERSI LAMA

  1. VERIFIKASI NUMERIK, BUKAN CUMA "TIDAK ERROR"
     Versi lama menganggap ekspor berhasil kalau tidak melempar exception.
     Padahal ONNX bisa diekspor dengan sukses tapi menghasilkan angka
     yang berbeda dari PyTorch, misalnya karena operator interpolate
     yang di-trace dengan ukuran statis padahal input berubah. Script ini
     membandingkan output PyTorch vs onnxruntime dan menolak kalau
     selisihnya di atas toleransi.

  2. opset 17, bukan 12
     opset 12 sudah sangat tua. Operator resize/interpolate punya
     dukungan jauh lebih baik di opset 16+.

  3. DYNAMIC AXES YANG BENAR
     Versi lama cuma menandai batch sebagai dinamis. Kalau backend
     mengirim gambar dengan resolusi berbeda, graph-nya pecah. Sekarang
     tinggi & lebar juga bisa dinamis (opsional, karena dynamic shape
     memperlambat sebagian runtime).

  4. SIMPLIFIKASI GRAPH
     Kalau `onnxsim` tersedia, graph disederhanakan. Ini biasanya
     memangkas 10-30% node dan mempercepat inference, sekaligus
     memperbesar peluang konversi TFLite berhasil.

  5. BENCHMARK
     Mengukur latency onnxruntime supaya kamu tahu angka nyata sebelum
     deploy, bukan menebak.

Usage:
    python scripts/07_export_onnx.py \
        --checkpoint runs/pidnet/<run>/best.pth \
        --img-size 512 --simplify --benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pidnet.model import InferenceWrapper, PIDNetS, count_parameters  # noqa: E402

NUM_CLASSES = 3


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Muat model beserta konfigurasi arsitekturnya dari checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    base_ch = cfg.get("base_ch", 32)
    use_ibn = cfg.get("use_ibn", True)
    ibn_ratio = cfg.get("ibn_ratio", 0.5)

    print(f"  Konfigurasi dari checkpoint: base_ch={base_ch}, "
          f"use_ibn={use_ibn}, ibn_ratio={ibn_ratio}")

    model = PIDNetS(
        num_classes=NUM_CLASSES, base_ch=base_ch,
        use_ibn=use_ibn, ibn_ratio=ibn_ratio,
        deep_supervision=False,   # head aux tidak dipakai saat inference
    )

    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        real_missing = [k for k in missing if not k.startswith("aux_head")]
        if real_missing:
            print(f"  PERINGATAN: bobot hilang: {real_missing[:5]}")
    if unexpected:
        real_unexpected = [k for k in unexpected if not k.startswith("aux_head")]
        if real_unexpected:
            print(f"  PERINGATAN: bobot tak terpakai: {real_unexpected[:5]}")

    model.to(device).eval()
    return model, ckpt


def _hw(img_size) -> tuple[int, int]:
    """Terima int, [S], atau [H, W]; kembalikan (H, W)."""
    if isinstance(img_size, int):
        return img_size, img_size
    v = list(img_size)
    return (v[0], v[0]) if len(v) == 1 else (v[0], v[1])


def export(model, img_size, out_path: Path, opset: int = 17,
           dynamic_hw: bool = False, apply_softmax: bool = False) -> None:
    wrapper = InferenceWrapper(model, apply_softmax=apply_softmax).eval()
    ih, iw = _hw(img_size)
    dummy = torch.randn(1, 3, ih, iw)

    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    if dynamic_hw:
        dynamic_axes["input"].update({2: "height", 3: "width"})
        dynamic_axes["output"].update({2: "height", 3: "width"})

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(out_path),
            input_names=["input"], output_names=["output"],
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  ONNX tersimpan: {out_path} ({size_mb:.2f} MB)")


def simplify(onnx_path: Path) -> bool:
    try:
        import onnx
        from onnxsim import simplify as onnxsim_simplify
    except ImportError:
        print("  onnxsim tidak terinstall, simplifikasi dilewati. "
              "(pip install onnxsim)")
        return False

    try:
        model_onnx = onnx.load(str(onnx_path))
        n_before = len(model_onnx.graph.node)
        simplified, ok = onnxsim_simplify(model_onnx)
        if not ok:
            print("  Simplifikasi gagal validasi, memakai graph asli.")
            return False
        onnx.save(simplified, str(onnx_path))
        n_after = len(simplified.graph.node)
        print(f"  Graph disederhanakan: {n_before} -> {n_after} node "
              f"({100 * (1 - n_after / max(1, n_before)):.0f}% lebih sedikit)")
        return True
    except Exception as e:
        print(f"  Simplifikasi error: {e}")
        return False


def verify(model, onnx_path: Path, img_size,
           apply_softmax: bool, tol: float = 1e-3) -> dict:
    """
    Bandingkan output PyTorch vs onnxruntime.

    Ini langkah yang paling sering dilewati orang dan paling sering
    jadi sumber bug misterius "kenapa akurasi di server beda dari
    waktu training".
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime tidak terinstall, verifikasi dilewati. "
              "(pip install onnxruntime)")
        return {"verified": False, "reason": "onnxruntime tidak ada"}

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    max_diff_all = 0.0
    argmax_agree_all = []

    for trial in range(3):
        ih, iw = _hw(img_size)
        x = rng.standard_normal((1, 3, ih, iw)).astype(np.float32)

        with torch.no_grad():
            wrapper = InferenceWrapper(model, apply_softmax=apply_softmax).eval()
            torch_out = wrapper(torch.from_numpy(x)).numpy()

        onnx_out = sess.run(["output"], {"input": x})[0]

        if torch_out.shape != onnx_out.shape:
            print(f"  GAGAL: bentuk output beda. "
                  f"torch={torch_out.shape} onnx={onnx_out.shape}")
            return {"verified": False, "reason": "shape mismatch"}

        diff = float(np.abs(torch_out - onnx_out).max())
        max_diff_all = max(max_diff_all, diff)

        # Yang paling penting: apakah keputusan argmax-nya sama?
        agree = float((torch_out.argmax(1) == onnx_out.argmax(1)).mean())
        argmax_agree_all.append(agree)

    mean_agree = float(np.mean(argmax_agree_all))
    print(f"  Selisih numerik maksimum : {max_diff_all:.3e}")
    print(f"  Kecocokan argmax piksel  : {mean_agree * 100:.4f}%")

    ok = max_diff_all < tol and mean_agree > 0.9999
    if ok:
        print("  Verifikasi LOLOS: ONNX identik dengan PyTorch.")
    else:
        print("  Verifikasi GAGAL. Jangan deploy ONNX ini.")
        print("  Penyebab umum: operator interpolate ter-trace dengan "
              "ukuran statis, atau opset terlalu rendah. Coba --opset 17 "
              "dan pastikan --dynamic-hw dimatikan kalau tidak dibutuhkan.")

    return {
        "verified": ok,
        "max_abs_diff": max_diff_all,
        "argmax_agreement": mean_agree,
    }


def benchmark(onnx_path: Path, img_size, n_runs: int = 20) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {}

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    ih, iw = _hw(img_size)
    x = np.random.randn(1, 3, ih, iw).astype(np.float32)

    for _ in range(3):
        sess.run(["output"], {"input": x})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(["output"], {"input": x})
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000
    result = {
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "fps": float(1000.0 / times.mean()),
    }
    print(f"  Latency CPU (onnxruntime): "
          f"mean={result['mean_ms']:.1f}ms  "
          f"p90={result['p90_ms']:.1f}ms  "
          f"({result['fps']:.1f} FPS)")
    print("  Catatan: ini mesin tempat script dijalankan, BUKAN HP target. "
          "HP Android mid-low biasanya 3-6x lebih lambat dari CPU server.")
    return result


def main():
    ap = argparse.ArgumentParser(description="Ekspor PIDNet-S ke ONNX")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default=None)
    # Default (384, 640) mengikuti app: nav_frame_converter.dart menyiapkan
    # tensor PIDNet 640x384. Ekspor pada bentuk lain berarti model di HP
    # menerima geometri yang tidak pernah dilihatnya saat training.
    ap.add_argument("--img-size", type=int, nargs="+", default=[384, 640],
                    metavar=("H", "W"),
                    help="Tinggi dan lebar. Satu angka = persegi. "
                         "Default 384 640 mengikuti app.")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic-hw", action="store_true",
                    help="Izinkan tinggi/lebar dinamis (lebih fleksibel, "
                         "tapi sebagian runtime jadi lebih lambat)")
    ap.add_argument("--softmax", action="store_true",
                    help="Tempel softmax di graph (backend tidak perlu "
                         "menghitung sendiri)")
    ap.add_argument("--simplify", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint tidak ada: {ckpt_path}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else \
        ckpt_path.parent / "pidnet_s.onnx"

    device = torch.device("cpu")
    print("=" * 66)
    print("  EKSPOR PIDNet-S -> ONNX")
    print("=" * 66)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output     : {out_path}")
    print(f"  img_size   : {_hw(args.img_size)[0]}x{_hw(args.img_size)[1]} "
          f"(HxW)   opset: {args.opset}")

    model, ckpt = load_model_from_checkpoint(ckpt_path, device)
    n_params, n_m = count_parameters(model)
    print(f"  Parameter  : {n_params:,} ({n_m:.2f} M)")

    print("\nMengekspor...")
    export(model, args.img_size, out_path, args.opset,
           args.dynamic_hw, args.softmax)

    if args.simplify:
        print("\nMenyederhanakan graph...")
        simplify(out_path)

    report = {"checkpoint": str(ckpt_path), "onnx": str(out_path),
              "img_size": list(_hw(args.img_size)), "opset": args.opset,
              "softmax_in_graph": args.softmax,
              "dynamic_hw": args.dynamic_hw}

    if not args.no_verify:
        print("\nVerifikasi numerik PyTorch vs ONNX...")
        report["verification"] = verify(model, out_path, args.img_size,
                                        args.softmax)

    if args.benchmark:
        print("\nBenchmark...")
        report["benchmark"] = benchmark(out_path, args.img_size)

    report_path = out_path.with_suffix(".export.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 66)
    print(f"  Selesai. Laporan: {report_path}")
    print("=" * 66)
    print("\n  Kontrak preprocessing untuk backend:")
    print("    1. BGR -> RGB")
    print("    2. Resize menjaga aspect ratio ke sisi terpanjang "
          f"{args.img_size}, lalu pad ke {args.img_size}x{args.img_size} "
          "di TENGAH dengan nilai 0")
    print("    3. Bagi 255, lalu normalisasi mean=[0.485,0.456,0.406] "
          "std=[0.229,0.224,0.225]")
    print("    4. Transpose ke NCHW, dtype float32")
    print("\n  Ini harus SAMA PERSIS dengan src/pidnet/dataset.py, "
          "kalau tidak akurasi produksi akan beda dari akurasi validasi.")


if __name__ == "__main__":
    main()
