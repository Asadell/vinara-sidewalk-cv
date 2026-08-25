# GUIDIO CV Training - Pipeline Revisi (Navigasi & Deteksi Rintangan Tunanetra)

Pipeline training resmi untuk dua model vision utama **GUIDIO** (`project/guidio_app`):

1. **YOLO11n (Obstacle BBox Detector - 6 Kelas)**: Deteksi bahaya dan rintangan fisik di trotoar/jalan untuk pengguna tunanetra.
2. **PIDNet-S (Sidewalk Semantic Segmenter - 3 Zona)**: Segmentasi real-time jalur trotoar aman, bahaya, dan area pejalan kaki.

Direvisi dengan fokus pada **kondisi dunia nyata yang sulit**: bayangan pohon, permukaan basah, malam hari, background ramai, dan ketidakseimbangan kelas (class imbalance).

---

## 🎯 6 Kelas Target Navigasi Terpadu (YOLO11n)

Semua label mentah dari 20+ dataset otomatis dipetakan oleh `scripts/03_merge_all_datasets.py` menjadi 6 kelas standar berikut:

| Class ID | Nama Kelas Target | Raw Keywords yang Diselaraskan | Fungsi untuk Pengguna Tunanetra |
| :---: | :--- | :--- | :--- |
| **0** | `lubang` | `pothole`, `potholes`, `hole`, `crack`, `road_hole`, `d40`, `d00`, `damage`, `pothole_water` | Mencegah tersandung / terperosok ke lubang jalan |
| **1** | `got_terbuka` | `manhole`, `drain`, `drain_hole`, `sewer_cover`, `got_terbuka`, `open_manhole` | Peringatan selokan / manhole terbuka |
| **2** | `tangga` | `stair`, `stairs`, `stairs_up`, `stairs_down`, `stairsup`, `stairsdown`, `step`, `steps`, `stairway`, `escalera` | Deteksi undakan / tangga naik & turun |
| **3** | `orang` | `person`, `persona`, `pedestrian` | Menghindari benturan dengan pejalan kaki lain |
| **4** | `motor` | `vehicle`, `car`, `motorcycle`, `motorbike`, `bus`, `truck` | Menghindari kendaraan melintas / parkir di trotoar |
| **5** | `tiang` | `pole`, `poles`, `tiang`, `obstacle`, `obstacles`, `bollard`, `bollards`, `post`, `lamp_post`, `street_pole`, `utility_pole`, `electric_pole`, `hydrant`, `equip_lamp`, `pole_hydro`, `pole_signal` | **Peringatan benturan badan/kepala** dengan tiang listrik, lamp post, & bollard |

---

## 🚀 Quick Start — 1-Click GPU Execution

Jika kamu menyewa GPU remote (Vast.ai, RunPod, Google Cloud, atau VPS custom), jalankan 1 perintah ini untuk otomatis mendownload dataset, mem-merge label, melatih model, dan mengekspor file `.tflite`:

```bash
git clone https://github.com/Asadell/vinara-sidewalk-cv.git && cd vinara-sidewalk-cv
export ROBOFLOW_API_KEY='FOdZd5fsYRPdf0n5SEEX'
bash scripts/run_vast.sh
```

---

## 📂 Struktur Repository

```
guido_cv_training_revised/
├── README.md
├── requirements.txt
├── configs/
│   └── custom_navigasi.yaml         # Konfigurasi data & 6 kelas target YOLO
├── src/
│   ├── yolo_aug.py                  # Augmentasi outdoor + patch Ultralytics
│   └── pidnet/
│       ├── model.py                 # PIDNet-S + IBN + Bag fusion + PAPPM
│       ├── dataset.py               # Augmentasi tersinkron + copy-paste hazard
│       ├── loss.py                  # OHEM CE + boundary-aware loss
│       └── metrics.py               # mIoU per zona + metrik keselamatan
└── scripts/
    ├── download_all_vinara_datasets.py # Auto-downloader 20+ dataset + fallback (v1..v10)
    ├── 03_merge_all_datasets.py     # Merger label otomatis & hardlink dataset zero-bloat
    ├── 05_train_yolo.py             # Training YOLO11n dengan outdoor augmentation
    ├── 06_train_pidnet.py           # Training PIDNet-S
    ├── 07_export_onnx.py            # Ekspor ONNX + verifikasi numerik
    ├── 07_export_tflite.py          # Ekspor TFLite (INT8) via onnx2tf
    ├── 08_copy_paste_minority.py    # Copy-paste kelas minoritas (tangga, got_terbuka, tiang)
    ├── 09_distill_yolo.py           # Pseudo-labeling / distillation
    └── 10_tune_thresholds.py        # Threshold per kelas (hazard-recall first)
```

---

## 🛠️ Langkah-Langkah Manual

### Step 1 — Download & Merge Dataset
```bash
# Set API Key Roboflow
export ROBOFLOW_API_KEY='your_key_here'

# Download 14 Sidewalk + 2 Pole + Pothole + Stairs datasets
python scripts/download_all_vinara_datasets.py

# Merge semua dataset menjadi 6 kelas terpadu
python scripts/03_merge_all_datasets.py
```

### Step 2 — Copy-Paste Kelas Minoritas (Tangga, Got, Tiang)
```bash
python scripts/08_copy_paste_minority.py \
    --dataset ~/datasets/dataset_master_yolo \
    --output ~/datasets/dataset_master_yolo_cp \
    --minority-classes tangga got_terbuka tiang \
    --target-multiplier 2.5 --preview
```

### Step 3 — Training YOLO11n
```bash
python scripts/05_train_yolo.py \
    --data ~/datasets/dataset_master_yolo_cp/data.yaml \
    --epochs 150 --aug-strength medium --oversample-minority
```

### Step 4 — Tuning Threshold per Kelas
```bash
python scripts/10_tune_thresholds.py \
    --weights runs/yolo/<run>/weights/best.pt \
    --data configs/custom_navigasi.yaml \
    --hazard-recall-target 0.90
```

### Step 5 — Ekspor ke TFLite INT8 (Mobile Flutter)
```bash
python scripts/07_export_tflite.py \
    --weights runs/yolo/<run>/weights/best.pt \
    --img-size 640
```

---

## 📊 Metrik Keselamatan Navigasi Tunanetra

Untuk aplikasi keselamatan tunanetra, kesalahan **tidak simetris**:
- **Lubang / Tiang tidak terdeteksi (False Negative)** → Pengguna terperosok / menabrak tiang (**Fatal**).
- **Bayangan disangka lubang (False Positive)** → Pengguna melambat sejenak (**Aman**).

Karena itu, pipeline ini memprioritaskan:
1. **Hazard Recall >= 90%** (memastikan >90% rintangan berbahaya terdeteksi).
2. **Threshold Per Kelas**: Nilai threshold untuk kelas bahaya (`lubang`, `got_terbuka`, `tiang`) diset lebih sensitif dibandingkan kelas informasional.
