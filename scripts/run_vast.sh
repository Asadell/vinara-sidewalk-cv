#!/usr/bin/env bash
# ==============================================================================
# GUIDIO - Complete GPU Training Pipeline (Revised)
# YOLO11n (Obstacle Detection) + PIDNet-S (Sidewalk Segmentation)
#
#   bash scripts/run_vast.sh
#   EPOCHS_YOLO=150 EPOCHS_PID=120 bash scripts/run_vast.sh
#   DATASET_ROOT=~/guido/dataset_sidewalk bash scripts/run_vast.sh
#
# CATATAN: versi sebelumnya punya beberapa cacat yang membuatnya diam-diam
# tidak mengerjakan apa yang tertulis:
#   1. `08_copy_paste_minority.py --data ...` -> script itu tidak punya --data.
#      `--data` ambigu antara --dataset dan --data-yaml, jadi argparse menolak.
#      Errornya ditelan `2>/dev/null || echo skip`, jadi copy-paste TIDAK
#      PERNAH jalan tanpa ada yang sadar.
#   2. Hasil copy-paste tidak pernah dipakai. Step training tetap menunjuk
#      configs/custom_navigasi.yaml (dataset asli), jadi walaupun step 3
#      berhasil, hasilnya dibuang.
#   3. Merge script dicari di ../datasets/merge_all.py dan di path absolut
#      $HOME/guido/... milik satu orang, padahal repo ini SUDAH punya
#      scripts/03_merge_all_datasets.py sendiri.
#   4. PIDNet dilatih --base-ch 32 --img-size 512, padahal README repo ini
#      sendiri merekomendasikan 16/384 untuk target 2 FPS di HP mid-low.
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f "/venv/main/bin/python" ]; then
    PYTHON_BIN="/venv/main/bin/python"
elif [ -f "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# ── Parameter ─────────────────────────────────────────────────────────────────
EPOCHS_YOLO="${EPOCHS_YOLO:-${1:-150}}"
EPOCHS_PID="${EPOCHS_PID:-${2:-120}}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets}"
YOLO_SRC="${YOLO_SRC:-$DATASET_ROOT/dataset_master_yolo}"
YOLO_CP="${YOLO_CP:-$DATASET_ROOT/dataset_master_yolo_cp}"
SEG_ROOT="${SEG_ROOT:-$DATASET_ROOT/dataset_master_seg}"

# PIDNet: default mengikuti KONTRAK APP, bukan angka bulat yang enak dilihat.
# lib/services/nav_frame_converter.dart menyiapkan tensor 640x384 dengan
# resize PAKSA (tanpa padding). Training harus memakai bentuk yang sama.
PID_H="${PID_H:-384}"
PID_W="${PID_W:-640}"
PID_BASE_CH="${PID_BASE_CH:-16}"      # README: 16 = titik awal aman untuk 2 FPS
PID_BATCH="${PID_BATCH:-12}"
RESIZE_MODE="${RESIZE_MODE:-stretch}"

YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
YOLO_BATCH="${YOLO_BATCH:-16}"
AUG_STRENGTH="${AUG_STRENGTH:-medium}"
MINORITY="${MINORITY:-tangga got_terbuka}"
CP_MULTIPLIER="${CP_MULTIPLIER:-2.5}"

echo "======================================================================"
echo "  GUIDIO CV Training - Revised Pipeline"
echo "  Repo      : $REPO_DIR"
echo "  Python    : $PYTHON_BIN"
echo "  YOLO      : $YOLO_SRC  (imgsz $YOLO_IMGSZ, $EPOCHS_YOLO epoch)"
echo "  PIDNet    : $SEG_ROOT  (${PID_H}x${PID_W} $RESIZE_MODE,"
echo "              base_ch $PID_BASE_CH, $EPOCHS_PID epoch)"
echo "======================================================================"

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo -e "\n[PRE-FLIGHT] Cek dependencies..."
"$PYTHON_BIN" - <<'PY' || "$PYTHON_BIN" -m pip install -r requirements.txt -q
import albumentations, cv2, torch, torchvision, ultralytics  # noqa: F401
PY
"$PYTHON_BIN" - <<'PY'
import albumentations as A, torch, torchvision, ultralytics
print(f"  torch {torch.__version__} | torchvision {torchvision.__version__}")
print(f"  ultralytics {ultralytics.__version__} | albumentations {A.__version__}")
print(f"  CUDA: {torch.cuda.is_available()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
PY

# ── Step 1: dataset ───────────────────────────────────────────────────────────
echo -e "\n[1/6] Cek dataset..."
if [ ! -d "$YOLO_SRC" ] && [ -f "$REPO_DIR/../datasets/download_all_vinara_datasets.py" ]; then
    "$PYTHON_BIN" "$REPO_DIR/../datasets/download_all_vinara_datasets.py"
fi

# ── Step 2: merge ─────────────────────────────────────────────────────────────
if [ ! -d "$YOLO_SRC/images/train" ]; then
    echo -e "\n[2/6] Merge dataset -> $YOLO_SRC ..."
    # Repo ini punya merge script-nya sendiri. Pakai itu, jangan cari-cari
    # ke path absolut milik mesin orang lain.
    "$PYTHON_BIN" scripts/03_merge_all_datasets.py \
        --base-dir "$DATASET_ROOT" --out-dir "$YOLO_SRC"
else
    echo -e "\n[2/6] $YOLO_SRC sudah ada - skip merge"
fi

# ── Step 3: copy-paste kelas minoritas ────────────────────────────────────────
# Errornya SENGAJA tidak ditelan lagi. Kalau step ini gagal, kita mau tahu.
echo -e "\n[3/6] Copy-paste kelas minoritas ($MINORITY)..."
# shellcheck disable=SC2086
"$PYTHON_BIN" scripts/08_copy_paste_minority.py \
    --dataset "$YOLO_SRC" \
    --output "$YOLO_CP" \
    --data-yaml configs/custom_navigasi.yaml \
    --minority-classes $MINORITY \
    --target-multiplier "$CP_MULTIPLIER" \
    --preview

# Dataset yang dipakai training = hasil copy-paste kalau ada, kalau tidak asli.
if [ -f "$YOLO_CP/data.yaml" ]; then
    TRAIN_DATA="$YOLO_CP/data.yaml"
elif [ -d "$YOLO_CP/images/train" ]; then
    TRAIN_DATA="$YOLO_CP/data.yaml"
else
    echo "  Copy-paste tidak menghasilkan dataset, pakai config asli."
    TRAIN_DATA="configs/custom_navigasi.yaml"
fi
echo "  Dataset training YOLO: $TRAIN_DATA"

# ── Step 4: YOLO11n ───────────────────────────────────────────────────────────
echo -e "\n[4/6] Training YOLO11n ($EPOCHS_YOLO epoch)..."
"$PYTHON_BIN" scripts/05_train_yolo.py \
    --data "$TRAIN_DATA" \
    --model yolo11n.pt \
    --epochs "$EPOCHS_YOLO" \
    --imgsz "$YOLO_IMGSZ" \
    --batch "$YOLO_BATCH" \
    --aug-strength "$AUG_STRENGTH" \
    --oversample-minority \
    --export-tflite

# ── Step 5: PIDNet-S ──────────────────────────────────────────────────────────
echo -e "\n[5/6] Training PIDNet-S ($EPOCHS_PID epoch)..."
"$PYTHON_BIN" scripts/06_train_pidnet.py \
    --dataset-root "$SEG_ROOT" \
    --epochs "$EPOCHS_PID" \
    --batch-size "$PID_BATCH" \
    --img-size "$PID_H" "$PID_W" \
    --resize-mode "$RESIZE_MODE" \
    --base-ch "$PID_BASE_CH" \
    --use-ibn \
    --pretrained \
    --copy-paste 0.4

# ── Step 6: export + threshold ────────────────────────────────────────────────
echo -e "\n[6/6] Export model..."

BEST_PTH="$(ls -t runs/pidnet/*/best.pth 2>/dev/null | head -1 || true)"
if [ -n "$BEST_PTH" ] && [ -f "$BEST_PTH" ]; then
    "$PYTHON_BIN" scripts/07_export_onnx.py \
        --checkpoint "$BEST_PTH" \
        --img-size "$PID_H" "$PID_W" --simplify --benchmark
    "$PYTHON_BIN" scripts/07_export_tflite.py \
        --checkpoint "$BEST_PTH" \
        --img-size "$PID_H" "$PID_W" \
        --resize-mode "$RESIZE_MODE" \
        --int8 --calib-data "$SEG_ROOT" --calib-samples 200
else
    echo "  best.pth tidak ditemukan - skip export PIDNet"
fi

BEST_PT="$(ls -t runs/yolo/*/weights/best.pt 2>/dev/null | head -1 || true)"
if [ -n "$BEST_PT" ] && [ -f "$BEST_PT" ]; then
    "$PYTHON_BIN" scripts/10_tune_thresholds.py \
        --weights "$BEST_PT" \
        --data "$TRAIN_DATA" \
        --hazard-recall-target 0.90 \
        --output runs/class_thresholds.json
else
    echo "  best.pt tidak ditemukan - skip tuning threshold"
fi

# ── Ringkasan ─────────────────────────────────────────────────────────────────
echo -e "\n======================================================================"
echo "  SELESAI."
echo ""
echo "  YOLO11n  : runs/yolo/*/weights/best_int8.tflite"
echo "             runs/class_thresholds.json"
echo "  PIDNet-S : runs/pidnet/*/pidnet_s*.onnx"
echo "             runs/pidnet/*/pidnet_s*_fp16.tflite"
echo ""
echo "  KONTRAK PREPROCESSING PIDNet (harus identik di Flutter):"
echo "    input      ${PID_H}x${PID_W} (HxW) RGB"
echo "    resize     $RESIZE_MODE"
echo "    normalisasi (x/255 - ImageNet mean) / ImageNet std"
echo ""
echo "  Deploy:"
echo "    *.tflite -> project/guidio_app/assets/models/"
echo "    *.onnx   -> project/backend/models/"
echo ""
echo "  BELUM OTOMATIS: app saat ini TIDAK membaca class_thresholds.json."
echo "  Threshold masih hardcoded di yolo_nav_int8_service.dart (0.25) dan"
echo "  yolo_navigasi_service.dart (0.30). Selama itu belum disambungkan,"
echo "  hasil 10_tune_thresholds.py tidak sampai ke pengguna."
echo "======================================================================"
