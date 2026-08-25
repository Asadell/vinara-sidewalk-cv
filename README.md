# GUIDIO CV Training - Pipeline Revisi

Pipeline training untuk dua model vision Vinara/GUIDIO:

1. **YOLO11n** - deteksi bahaya jalanan, 6 kelas
2. **PIDNet-S** - segmentasi jalur trotoar 3 zona

Direvisi dengan fokus pada **kondisi dunia nyata yang sulit**: bayangan pohon,
permukaan basah, malam hari, background ramai, dan ketidakseimbangan kelas.

---


---

## 🎯 6 Kelas Target Navigasi Terpadu (YOLO11n)

Semua label mentah dari 20+ dataset otomatis dipetakan oleh `scripts/03_merge_all_datasets.py` menjadi 6 kelas standar berikut:

| Class ID | Nama Kelas Target | Raw Keywords yang Diselaraskan | Fungsi untuk Pengguna Tunanetra |
| :---: | :--- | :--- | :--- |
| **0** | `lubang` | `pothole`, `potholes`, `hole`, `crack`, `road_hole`, `d40`, `d00`, `damage`, `pothole_water`, `drain_hole` | Mencegah tersandung / terperosok ke lubang jalan |
| **1** | `got_terbuka` | `manhole`, `drain`, `drain_hole`, `sewer_cover`, `got_terbuka`, `open_manhole` | Peringatan selokan / manhole terbuka |
| **2** | `tangga` | `stair`, `stairs`, `stairs_up`, `stairs_down`, `stairsup`, `stairsdown`, `step`, `steps`, `stairway`, `escalera` | Deteksi undakan / tangga naik & turun |
| **3** | `orang` | `person`, `persona`, `pedestrian` | Menghindari benturan dengan pejalan kaki lain |
| **4** | `motor` | `vehicle`, `car`, `motorcycle`, `motorbike`, `bus`, `truck` | Menghindari kendaraan melintas / parkir di trotoar |
| **5** | `tiang` | `pole`, `poles`, `tiang`, `obstacle`, `obstacles`, `bollard`, `bollards`, `post`, `lamp_post`, `street_pole`, `utility_pole`, `electric_pole`, `hydrant`, `equip_lamp`, `pole_hydro`, `pole_signal` | **Peringatan benturan badan/kepala** dengan tiang listrik, lamp post, & bollard |

---

## 📦 Daftar 20+ Dataset Terintegrasi

Script `scripts/download_all_vinara_datasets.py` otomatis mengunduh & menangani fallback versi (v1..v10) untuk dataset berikut:

### 1. Dataset Pole / Tiang (2 Dataset)
- `poleproject/pole-detection-v2-bvqug` (Roboflow)
- `akshay-anand-bfabb/pole-data` (Roboflow)

### 2. Dataset Sidewalk & Road (14 Dataset)
- `alharth-alhaj-hussein-1hig7/sidewalk-road`
- `project-xagxj/road_segment_v2`
- `project-xagxj/road_person_recognition`
- `project-xagxj/road_person_recognition_v2`
- `sidewalk/sidewalk-segmentation`
- `sidewalk/sidewalks-seg`
- `school-stpl7/sidewalk-dz4ug`
- `senior-design-scl0l/sidewalk_semantics_segmentation`
- `yolo-s6mwf/sidewalk-6imhx`
- `fieldlinedetection/sidewalk_test`
- `elvis-cmqng/sidewalk-and-stair-train-image`
- `capstone-project-nhlns/sidewalk-detection-ykwpf`
- `do-hvtgm/test_sidewalk_1`
- `projects-5k1o6/sidewalk-dlu6l`

### 3. Dataset Pothole / Lubang (4 Dataset)
- `anggadwisunarto/potholes-detection-yolov8` (Kaggle)
- `intel-unnati-training-program/pothole-detection-bqu6s` (Roboflow)
- `projects-hjaax/pothole-detection-using-yolov5` (Roboflow)
- `roaddamage-ak8w6/road-damage-uyvns` (Roboflow)

### 4. Dataset Tangga / Stairs (5 Dataset)
- `stair-eyhvv/stairs-detection-6cq2a` (Roboflow)
- `tesisusbbog/stairs-data` (Roboflow)
- `group10textdetect/stair-detect` (Roboflow)
- `dataclusterlabs/stairs-image-dataset` (Kaggle)
- `samuelayman/stairs` (Kaggle)

---

## 🚀 Quick Start — 1-Click GPU Execution

Jika kamu menyewa GPU remote (Vast.ai, RunPod, Google Cloud, atau VPS custom), jalankan 1 perintah ini untuk otomatis mendownload dataset, mem-merge label, melatih model, dan mengekspor file `.tflite`:

```bash
git clone https://github.com/Asadell/vinara-sidewalk-cv.git && cd vinara-sidewalk-cv
export ROBOFLOW_API_KEY='FOdZd5fsYRPdf0n5SEEX'
bash scripts/run_vast.sh
```

## Temuan dari kode lama yang perlu kamu tahu duluan

### 1. Docstring dan kode tidak sinkron di augmentasi PIDNet

`06_train_pidnet.py` versi lama mengklaim:

> "Synchronized Albumentations (Flip, Lighting Jitter, Motion Blur) untuk
> memperbanyak variasi kondisi trotoar Indonesia secara otomatis di setiap epoch"

Tapi `src/pidnet/dataset.py` sebenarnya cuma melakukan ini:

```python
if self.augment and np.random.rand() < 0.5:
    img  = img.transpose(Image.FLIP_LEFT_RIGHT)
    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
```

Horizontal flip saja. Tidak ada Albumentations, lighting jitter, maupun motion
blur. Ini penting karena kalau kamu menganalisis "kenapa segmentasi jeblok saat
hujan" sambil mengira augmentasinya sudah kaya, kesimpulanmu akan salah arah.

### 2. Target boundary bukan boundary

```python
bnd_target = (masks == 2).float().unsqueeze(1)
```

Itu mask kelas hazard, bukan batas antar zona. Cabang D di PIDNet dirancang
belajar **tepi antar zona**. Melatihnya memprediksi seluruh area hazard membuat
cabang D jadi duplikat cabang segmentasi utama, dan seluruh manfaat arsitektur
tiga-cabang hilang. Tepi trotoar dengan jalan raya sama pentingnya dengan tepi
hazard, karena di situlah pengguna bisa tersandung turun ke jalan.

### 3. Konversi TFLite memakai paket mati

Versi lama meng-import `onnx_tf`, yang sudah tidak dirawat sejak 2023 dan tidak
kompatibel dengan TensorFlow 2.16+. Lucunya docstring-nya sendiri sudah menulis
*"onnx-tf sudah deprecated, jadi pakai pipeline tf.lite.TFLiteConverter"* tapi
kodenya tetap meng-import onnx_tf. Niatnya benar, implementasinya belum menyusul.

Masalah tambahan: `onnx-tf` menyisipkan operator Transpose di mana-mana untuk
konversi NCHW ke NHWC, yang bisa membuat model 2-5x lebih lambat. Fatal untuk
target 2 FPS di HP mid-low.

### 4. Bobot kelas ditebak manual

`[1.0, 1.0, 3.0]` untuk hazard adalah tebakan. Kalau hazard cuma 0,5% dari
piksel, 3x masih terlalu kecil. Kalau 15%, 3x bikin model over-predict hazard
dan spam peringatan palsu. Sekarang dihitung dari distribusi piksel aktual.

### 5. `device=0` di-hardcode

Training langsung crash di mesin tanpa GPU, tanpa pesan yang jelas.

### 6. Path absolut mesin developer di config

`/home/asadel/kuliah/lomba/smstr6/...` pecah begitu dijalankan di Colab, Kaggle,
Vast.ai, atau laptop anggota tim lain.

---

## Struktur

```
guido_cv_training_revised/
├── README.md
├── requirements.txt
├── configs/
│   └── custom_navigasi.yaml
├── src/
│   ├── yolo_aug.py               # augmentasi outdoor + patch Ultralytics
│   └── pidnet/
│       ├── model.py              # PIDNet-S + IBN + Bag fusion + PAPPM
│       ├── dataset.py            # augmentasi tersinkron + copy-paste hazard
│       ├── loss.py               # OHEM CE + boundary-aware
│       └── metrics.py            # mIoU per zona + metrik keselamatan
└── scripts/
    ├── 05_train_yolo.py          # training YOLO11n
    ├── 06_train_pidnet.py        # training PIDNet-S
    ├── 07_export_onnx.py         # ekspor ONNX + verifikasi numerik
    ├── 07_export_tflite.py       # ekspor TFLite via onnx2tf
    ├── 08_copy_paste_minority.py # copy-paste kelas minoritas (bbox)
    ├── 09_distill_yolo.py        # pseudo-labeling / distillation
    └── 10_tune_thresholds.py     # threshold per kelas, recall-first
```

Script 01-04 (konversi dataset) tidak diubah, tetap pakai versi lamamu.

---

## Setup

```bash
# GPU: install torch sesuai CUDA dari pytorch.org DULU
pip install -r requirements.txt
```

`albumentations>=2.0` wajib. Tanpa itu augmentasi outdoor tidak aktif sama
sekali, dan itu justru inti dari revisi ini.

---

## Alur YOLO11n

### Step 1 - Cek distribusi & preview augmentasi

```bash
python scripts/05_train_yolo.py --data configs/custom_navigasi.yaml --epochs 1
```

Script mencetak distribusi kelas dan menyimpan `augmentation_preview.jpg`.
**Lihat gambarnya.** Kalau kamu sendiri tidak bisa melihat lubangnya di hasil
augmentasi, model juga tidak bisa, dan augmentasi itu justru merugikan.

### Step 2 - Copy-paste kelas minoritas (kalau rasio > 8x)

```bash
python scripts/08_copy_paste_minority.py \
    --dataset ../dataset_master_yolo \
    --output ../dataset_master_yolo_cp \
    --minority-classes tangga tiang got_terbuka \
    --target-multiplier 2.5 --preview
```

Ultralytics punya parameter `copy_paste`, tapi itu hanya bekerja untuk task
segmentasi karena butuh mask poligon. Dataset kamu format bbox, jadi parameter
itu tidak berpengaruh apa-apa. Script ini mengisi celah tersebut.

Periksa `copy_paste_preview.jpg`. Kalau tempelannya kelihatan seperti stiker,
model akan belajar mendeteksi stiker. Turunkan multiplier.

### Step 3 - Training

```bash
python scripts/05_train_yolo.py \
    --data ../dataset_master_yolo_cp/data.yaml \
    --epochs 150 --aug-strength medium --oversample-minority
```

Untuk baseline pembanding (penting, supaya tahu augmentasinya benar-benar
membantu atau tidak):

```bash
python scripts/05_train_yolo.py --no-outdoor-aug --name baseline
```

### Step 4 - Tuning threshold per kelas

```bash
python scripts/10_tune_thresholds.py \
    --weights runs/yolo/<run>/weights/best.pt \
    --data configs/custom_navigasi.yaml \
    --hazard-recall-target 0.90
```

Ini menghasilkan `class_thresholds.json`.

### Step 5 (opsional) - Manfaatkan foto tak berlabel

```bash
# Latih teacher besar dulu
python scripts/05_train_yolo.py --model yolo11l.pt --name teacher --batch 8

# Pseudo-label foto jalanan mentah
python scripts/09_distill_yolo.py --mode pseudo \
    --teacher runs/yolo/teacher/weights/best.pt \
    --unlabeled-dir ~/foto_jalanan_mentah \
    --data configs/custom_navigasi.yaml \
    --output-dataset ../dataset_master_yolo_pseudo
```

---

## Alur PIDNet-S

```bash
python scripts/06_train_pidnet.py \
    --dataset-root ../dataset_master_seg \
    --epochs 120 --batch-size 12 --img-size 512 \
    --base-ch 32 --use-ibn --copy-paste 0.4

python scripts/07_export_onnx.py \
    --checkpoint runs/pidnet/<run>/best.pth \
    --img-size 512 --simplify --benchmark

python scripts/07_export_tflite.py \
    --checkpoint runs/pidnet/<run>/best.pth --img-size 512
```

### Memilih `--base-ch`

Diukur di container ini (CPU, 4 thread, PyTorch eager, bukan HP target):

| base_ch | Parameter | 384px | 512px |
|---|---|---|---|
| 16 | 1,18 M | 45 ms | 67 ms |
| 24 | 2,65 M | 75 ms | 128 ms |
| 32 | 4,71 M | 108 ms | 193 ms |

HP Android mid-low umumnya 3-6x lebih lambat dari CPU server. Untuk target
2 FPS (500 ms), **`--base-ch 16 --img-size 384` adalah titik awal paling aman**,
naikkan kalau ternyata masih lapang. Angka pastinya harus kamu ukur di HP
target, bukan diekstrapolasi dari tabel ini.

---

## Yang berubah, per file

### `05_train_yolo.py`

| Aspek | Lama | Baru |
|---|---|---|
| Augmentasi cahaya | HSV jitter seragam | Bayangan tajam & belang daun, malam (gamma + lampu jalan + noise + warm shift), hujan, kabut, silau, permukaan basah |
| Motion blur | Tidak ada | MotionBlur berarah, defocus, downscale |
| Copy-paste | `copy_paste=0` (tidak berfungsi untuk bbox) | Script 08 terpisah |
| Class imbalance | Tidak ditangani | Analisis + oversampling berbasis symlink |
| Device | Hardcode `device=0` | Auto-detect |
| LR schedule | Default | AdamW + cosine + warmup |
| `close_mosaic` | Implisit | Eksplisit (default 15) |
| `erasing` | 0.1 | 0.25 |
| `scale` | 0.3 | 0.45 |
| Evaluasi | mAP agregat | Recall per kelas, pemisahan kelas bahaya vs info, gate rilis |
| Mapping kelas | Asumsi `ap_class_index` lengkap | Mapping eksplisit + peringatan kelas kosong di val |

### `06_train_pidnet.py`

| Aspek | Lama | Baru |
|---|---|---|
| Loss | CrossEntropy biasa | OHEM CE + boundary + boundary-aware seg |
| Bobot kelas | Tebakan `[1,1,3]` | Median-frequency dari distribusi piksel aktual |
| Target boundary | `(masks == 2)` (salah) | Tepi antar zona sungguhan |
| AMP | Tidak ada | Ada |
| Scheduler | CosineAnnealingLR | PolynomialLR (power 0.9) + warmup |
| EMA | Tidak ada | Ada |
| Gradient clipping | Tidak ada | Ada |
| Metrik | mIoU + safety error | + mIoU per zona, akurasi keputusan zona, hazard recall, hazard->walkable |
| Pemilihan checkpoint | mIoU saja | Skor komposit yang memperhitungkan keselamatan |
| Confusion matrix | `.cpu()` tiap batch | Dihitung di GPU |
| `run_epoch` | Menerima optimizer saat eval | Fungsi terpisah |
| Resume | Tidak ada | Optimizer + scheduler + scaler + EMA |

### `src/pidnet/model.py`

| Aspek | Lama | Baru |
|---|---|---|
| Domain generalization | Tidak ada | IBN blocks (nol biaya inference) |
| Fusion | Concat + conv | Bag fusion dipandu boundary |
| Konteks | Tidak ada | Pyramid pooling (PAPPM-style) |
| Stem | 2x conv stride-2 langsung dari 3ch | Naikkan channel dulu sebelum downsample kedua |
| Deep supervision | Tidak ada | Head aux di cabang I |
| Ekspor | `return_aux` bercabang | `InferenceWrapper` selalu satu tensor |
| Efisiensi | `p_down(s)` dihitung 2x | Dibagi ke cabang P dan D |

### `src/pidnet/dataset.py`

| Aspek | Lama | Baru |
|---|---|---|
| Augmentasi | Horizontal flip saja | Geometri tersinkron + fotometrik + cuaca + malam + optik |
| Resize | Squash ke persegi | LongestMaxSize + pad (jaga aspect ratio) |
| Padding | Dihitung sebagai kelas 0 | `ignore_index=255`, tidak dihitung di loss & metrik |
| Interpolasi mask | Tidak eksplisit | NEAREST dipaksa di semua transform geometris |
| Copy-paste hazard | Tidak ada | Ada, dengan blending tepi |

### `07_export_*.py`

| Aspek | Lama | Baru |
|---|---|---|
| Verifikasi | "tidak error = berhasil" | Bandingkan numerik PyTorch vs ONNX vs TFLite |
| opset | 12 | 17 |
| Dynamic axes | Batch saja | Batch + opsional H/W |
| Simplifikasi | Tidak ada | onnxsim |
| TFLite | `onnx-tf` (mati) | `onnx2tf` (NHWC-native) |
| Benchmark | Tidak ada | Latency p50/p90 |

---

## Kenapa mAP saja tidak cukup

Untuk aplikasi keselamatan, kesalahan tidak simetris:

- **Lubang tidak terdeteksi** → pengguna terperosok
- **Bayangan disangka lubang** → pengguna melambat sebentar

mAP memperlakukan keduanya sama. Model dengan mAP 0,85 yang sering melewatkan
lubang lebih berbahaya daripada model mAP 0,80 yang konservatif.

Karena itu pipeline ini:
- Melaporkan **recall per kelas**, bukan cuma mAP agregat
- Memisahkan **kelas bahaya** dari **kelas informasional**
- Punya **gate rilis** eksplisit
- Menghitung threshold **per kelas** (bahaya: rendah, info: tinggi)

Untuk PIDNet ada tambahan: **akurasi keputusan zona**. App tidak membacakan mask
piksel ke pengguna, yang dibacakan itu kesimpulan zona ("geser ke kanan"). Model
bisa punya mIoU bagus tapi keputusan zona kacau kalau kesalahannya menumpuk di
satu sisi.

---

## Urutan prioritas kalau waktumu terbatas

1. **Perbaiki target boundary dan bobot kelas PIDNet.** Ini bug, bukan
   optimasi. Efeknya langsung.
2. **Nyalakan augmentasi outdoor untuk keduanya.** Murah, gain cepat.
3. **Tuning threshold per kelas.** Menaikkan recall kelas bahaya tanpa
   training ulang sama sekali.
4. **IBN untuk PIDNet.** Nol biaya inference, bantu domain shift.
5. **Copy-paste kelas minoritas** kalau rasio masih > 8x.
6. **Pseudo-labeling** kalau punya foto jalanan tak berlabel.
7. **Feature distillation** paling akhir. Literatur melaporkan sekitar
   +1,9 sampai +2,5 poin mAP50 untuk YOLO11x→YOLO11n. Nyata tapi tidak
   dramatis, dan tidak sepadan sebelum langkah 1-6 beres.

---

## Catatan yang perlu diverifikasi sendiri

- **Angka distillation (+1,9 sampai +2,5 mAP50)** berasal dari literatur pada
  dataset dan domain lain, bukan pengukuran di dataset GUIDIO. Mekanismenya
  masuk akal, magnitudenya bisa beda.
- **Benchmark base_ch** diukur di container ini dengan PyTorch eager di CPU.
  Setelah kuantisasi TFLite INT8 di HP, karakteristiknya bisa cukup berbeda.
  Ukur di HP target.
- **Gate rilis** (recall bahaya 0,85, safety error 5%, akurasi zona 85%) adalah
  titik awal yang masuk akal untuk aplikasi bantu tunanetra, bukan standar
  industri baku. Sesuaikan setelah uji lapangan dengan pengguna sungguhan.
- **Monkeypatch Ultralytics** di `src/yolo_aug.py` bergantung pada struktur
  internal yang bisa berubah antar versi. Script memverifikasi dulu dan
  membatalkan patch dengan pesan jelas kalau tidak cocok, jadi training tetap
  jalan. Tapi kalau kamu update Ultralytics, cek pesannya di awal training.
- **PIDNet-S di sini implementasi terinspirasi paper, bukan port persis.**
  Kalau butuh reproduksi angka paper, pakai repo resmi plus bobot pretrained
  ImageNet-nya.
- **Copy-paste bbox bisa menurunkan precision** kalau berlebihan, karena model
  belajar artefak tempelan. Mulai dari multiplier 2.0 dan ukur.
- **Pseudo-labeling bisa menurunkan performa** kalau teacher kurang bagus,
  karena kesalahan teacher ikut diwariskan. Selalu bandingkan dengan student
  tanpa pseudo-label.
