#!/usr/bin/env bash
# ==============================================================================
# GUIDIO — Complete GPU Training Pipeline (Revised)
# YOLO11n (Obstacle Detection) + PIDNet-S (Sidewalk Segmentation)
#
# Usage di VPS setelah setup credentials:
#   bash scripts/run_vast.sh
#   bash scripts/run_vast.sh 150 120   # custom epochs: YOLO=150, PIDNet=120
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ── Python: deteksi venv atau fallback ke python3 ─────────────────────────────
PYTHON_BIN=""
if [ -f "/venv/main/bin/python" ]; then
    PYTHON_BIN="/venv/main/bin/python"
elif [ -f "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo "  GUIDIO CV Training — Revised Pipeline"
echo "  Repo     : $REPO_DIR"
echo "  Python   : $PYTHON_BIN"
echo "======================================================================"

# ── Cek & install dependencies ────────────────────────────────────────────────
echo -e "\n[PRE-FLIGHT] Checking dependencies..."
"$PYTHON_BIN" -c "import ultralytics, torch, albumentations" 2>/dev/null || {
    echo "  Installing from requirements.txt ..."
    "$PYTHON_BIN" -m pip install -r requirements.txt -q
}
echo "  ✓ Dependencies OK"

# ── Parameter epoch ───────────────────────────────────────────────────────────
EPOCHS_YOLO="${1:-150}"
EPOCHS_PID="${2:-120}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets}"

# ── Step 1: Download dataset (jika folder belum ada) ──────────────────────────
if [ ! -d "$DATASET_DIR/pothole" ]; then
    echo -e "\n[1/6] Downloading datasets ke $DATASET_DIR ..."
    "$PYTHON_BIN" ../datasets/download_all_vinara_datasets.py
else
    echo -e "\n[1/6] Dataset sudah ada di $DATASET_DIR — skip download"
fi

# ── Step 2: Merge & konversi ke dataset_master_yolo ──────────────────────────
echo -e "\n[2/6] Merging YOLO datasets -> dataset_master_yolo ..."
MERGE_SCRIPT=""
for candidate in \
    "$REPO_DIR/../datasets/merge_all.py" \
    "$HOME/guido/dataset_sidewalk/guido_cv_training/scripts/03_merge_all_datasets.py"; do
    [ -f "$candidate" ] && MERGE_SCRIPT="$candidate" && break
done
if [ -n "$MERGE_SCRIPT" ]; then
    "$PYTHON_BIN" "$MERGE_SCRIPT" --base-dir "$DATASET_DIR"
else
    echo "  ⚠ Merge script tidak ditemukan — asumsikan dataset_master_yolo sudah tersedia"
fi

# ── Step 3: Copy-paste augmentasi untuk kelas minoritas ───────────────────────
echo -e "\n[3/6] Copy-paste augmentation (tangga, got_terbuka)..."
"$PYTHON_BIN" scripts/08_copy_paste_minority.py \
    --data configs/custom_navigasi.yaml 2>/dev/null || \
    echo "  ⚠ Skip (dataset_master_yolo mungkin belum ada)"

# ── Step 4: Training YOLO11n ──────────────────────────────────────────────────
echo -e "\n[4/6] Training YOLO11n ($EPOCHS_YOLO epochs)..."
"$PYTHON_BIN" scripts/05_train_yolo.py \
    --data configs/custom_navigasi.yaml \
    --model yolo11n.pt \
    --epochs "$EPOCHS_YOLO" \
    --imgsz 640 \
    --batch 16 \
    --aug-strength medium \
    --oversample-minority \
    --export-tflite

# ── Step 5: Training PIDNet-S ─────────────────────────────────────────────────
echo -e "\n[5/6] Training PIDNet-S ($EPOCHS_PID epochs)..."
"$PYTHON_BIN" scripts/06_train_pidnet.py \
    --dataset-root "$DATASET_DIR/dataset_master_seg" \
    --epochs "$EPOCHS_PID" \
    --batch-size 12 \
    --img-size 512 \
    --base-ch 32 \
    --use-ibn \
    --copy-paste 0.4

# ── Step 6: Export semua model ────────────────────────────────────────────────
echo -e "\n[6/6] Exporting models..."

BEST_PTH=$(ls -t runs/pidnet/*/best.pth 2>/dev/null | head -1 || echo "")
if [ -f "$BEST_PTH" ]; then
    "$PYTHON_BIN" scripts/07_export_onnx.py \
        --checkpoint "$BEST_PTH" --img-size 512 --simplify --benchmark
    "$PYTHON_BIN" scripts/07_export_tflite.py \
        --checkpoint "$BEST_PTH" --img-size 512
else
    echo "  ⚠ best.pth tidak ditemukan — skip PIDNet export"
fi

BEST_PT=$(ls -t runs/yolo/*/weights/best.pt 2>/dev/null | head -1 || echo "")
if [ -f "$BEST_PT" ]; then
    "$PYTHON_BIN" scripts/10_tune_thresholds.py \
        --weights "$BEST_PT" \
        --data configs/custom_navigasi.yaml \
        --hazard-recall-target 0.90
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "\n======================================================================"
echo "  [✓] SELESAI! Output ada di:"
echo ""
echo "  YOLO11n (mobile):"
echo "    runs/yolo/*/weights/best_int8.tflite"
echo "    class_thresholds.json"
echo ""
echo "  PIDNet-S (backend):"
echo "    runs/pidnet/*/pidnet_s_navigasi.onnx"
echo "    runs/pidnet/*/pidnet_s_navigasi_fp16.tflite"
echo ""
echo "  Langkah selanjutnya:"
echo "    1. Salin *.tflite ke project/guidio_app/assets/models/"
echo "    2. Salin *.onnx ke project/backend/models/"
echo "======================================================================"
