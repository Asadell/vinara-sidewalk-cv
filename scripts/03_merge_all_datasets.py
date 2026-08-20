"""
Script 03 (Unified): Merge all datasets (SafeWalkBD + Kaggle + Roboflow Potholes & Stairs)
into a unified dataset_master_yolo/ structure.

Target Classes (6):
  0: lubang        (potholes, road holes, cracks)
  1: got_terbuka   (open manholes, drains)
  2: tangga        (stairs up/down, steps)
  3: orang         (person, pedestrian)
  4: motor         (vehicles, motorcycles, cars)
  5: tiang         (poles, obstacles, bollards)
"""
import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

RANDOM_SEED = 42
TARGET_CLASSES = ["lubang", "got_terbuka", "tangga", "orang", "motor", "tiang"]
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

# Flexible keyword mapping to target class names
KEYWORD_MAPPING = {
    # Potholes -> 0
    "pothole": "lubang",
    "potholes": "lubang",
    "hole": "lubang",
    "crack": "lubang",
    "d40": "lubang", "d00": "lubang", "d10": "lubang", "d20": "lubang",
    
    # Got / Manhole -> 1
    "manhole": "got_terbuka",
    "drain": "got_terbuka",
    "got_terbuka": "got_terbuka",
    "open_manhole": "got_terbuka",

    # Tangga / Stairs -> 2
    "stair": "tangga",
    "stairs": "tangga",
    "stairs_up": "tangga",
    "stairs_down": "tangga",
    "stair_up": "tangga",
    "stair_down": "tangga",
    "upstairs": "tangga",
    "downstairs": "tangga",
    "step": "tangga",
    "steps": "tangga",
    "stairway": "tangga",

    # Orang -> 3
    "person": "orang",
    "pedestrian": "orang",

    # Motor / Vehicles -> 4
    "vehicle": "motor",
    "car": "motor",
    "motorcycle": "motor",
    "motorbike": "motor",
    "bus": "motor",
    "truck": "motor",

    # Tiang / Obstacle -> 5
    "pole": "tiang",
    "obstacle": "tiang",
    "bollard": "tiang",
    "post": "tiang",
}


def map_class_name(raw_name: str, fallback_default: str = None) -> str:
    cleaned = raw_name.strip().lower()
    if cleaned in KEYWORD_MAPPING:
        return KEYWORD_MAPPING[cleaned]
    # Check substring matches
    for k, v in KEYWORD_MAPPING.items():
        if k in cleaned:
            return v
    return fallback_default


def parse_voc_xml(xml_path: Path, fallback_class: str = None):
    """Parses Pascal VOC XML file and returns YOLO lines (class_id, x_center, y_center, w, h)."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            return []
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)
        if img_w <= 0 or img_h <= 0:
            return []

        yolo_lines = []
        for obj in root.findall("object"):
            raw_name = obj.find("name").text
            target_cls = map_class_name(raw_name, fallback_default=fallback_class)
            if target_cls is None:
                continue
            cls_id = CLASS_NAME_TO_ID[target_cls]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Normalize to 0..1
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            cx = (xmin + xmax) / (2.0 * img_w)
            cy = (ymin + ymax) / (2.0 * img_h)

            # Clamp
            cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
            w, h   = max(0.0, min(1.0, w)), max(0.0, min(1.0, h))

            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return yolo_lines
    except Exception:
        return []


def process_dataset_folder(dataset_dir: Path, out_root: Path, prefix: str, default_class: str = None):
    """Generic processor for YOLO or VOC datasets."""
    if not dataset_dir.exists():
        return 0

    # Read data.yaml if available
    yaml_files = list(dataset_dir.glob("data*.yaml")) + list(dataset_dir.glob("*.yaml"))
    src_class_map = {}
    if yaml_files:
        try:
            cfg = yaml.safe_load(yaml_files[0].read_text())
            names = cfg.get("names", [])
            if isinstance(names, dict):
                id_to_name = {int(k): str(v) for k, v in names.items()}
            else:
                id_to_name = {i: str(n) for i, n in enumerate(names)}
            for src_id, src_name in id_to_name.items():
                mapped = map_class_name(src_name, fallback_default=default_class)
                if mapped:
                    src_class_map[src_id] = CLASS_NAME_TO_ID[mapped]
        except Exception:
            pass

    # Collect images
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    found_items = []

    # Check for VOC XMLs first
    xml_files = list(dataset_dir.rglob("*.xml"))
    if xml_files:
        for xml_file in xml_files:
            stem = xml_file.stem
            if stem.lower().endswith(".jpg") or stem.lower().endswith(".png"):
                stem = Path(stem).stem
            
            parent = xml_file.parent
            img_candidates = [
                parent / f"{stem}{ext}" for ext in image_extensions
            ] + [
                parent.parent / f"{stem}{ext}" for ext in image_extensions
            ] + [
                parent.parent / "images" / f"{stem}{ext}" for ext in image_extensions
            ] + list(dataset_dir.rglob(f"{stem}.*"))

            img_path = None
            for cand in img_candidates:
                if cand.is_file() and cand.suffix.lower() in image_extensions:
                    img_path = cand
                    break

            if img_path:
                yolo_lines = parse_voc_xml(xml_file, fallback_class=default_class)
                if yolo_lines:
                    found_items.append((img_path, yolo_lines))
    else:
        # Standard YOLO text annotations
        all_imgs = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in image_extensions]
        for img_path in all_imgs:
            label_candidates = [
                img_path.with_suffix(".txt"),
                img_path.parent.parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt",
                img_path.parent / f"{img_path.stem}.txt",
            ] + list(dataset_dir.rglob(f"{img_path.stem}.txt"))

            lbl_file = None
            for cand in label_candidates:
                if cand.is_file() and cand != img_path:
                    lbl_file = cand
                    break

            if lbl_file and lbl_file.exists():
                lines = lbl_file.read_text().strip().splitlines()
                yolo_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        src_id = int(parts[0])
                    except ValueError:
                        continue
                    if src_id in src_class_map:
                        target_id = src_class_map[src_id]
                    elif default_class:
                        target_id = CLASS_NAME_TO_ID[default_class]
                    else:
                        continue
                    yolo_lines.append(f"{target_id} " + " ".join(parts[1:]))

                if yolo_lines:
                    found_items.append((img_path, yolo_lines))

    if not found_items:
        return 0

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(found_items)
    n = len(found_items)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    copied = 0
    for idx, (img_path, lines) in enumerate(found_items):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "valid"
        else:
            split = "test"

        img_out_dir = out_root / "images" / split
        lbl_out_dir = out_root / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        new_stem = f"{prefix}_{idx:05d}_{img_path.stem}"
        dest_img = img_out_dir / f"{new_stem}{img_path.suffix}"
        dest_lbl = lbl_out_dir / f"{new_stem}.txt"

        # Gunakan Hardlink / Symlink agar HAMPIR 0 MB Disk Space (bebas dari error No Space Left)
        if dest_img.exists():
            dest_img.unlink()
        try:
            os.link(img_path, dest_img)
        except Exception:
            try:
                os.symlink(img_path.resolve(), dest_img)
            except Exception:
                shutil.copy2(img_path, dest_img)

        dest_lbl.write_text("\n".join(lines) + "\n")
        copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(description="Merge all pothole and stairs datasets")
    parser.add_argument("--base-dir", default=os.path.expanduser("~/datasets"), help="Path to ~/datasets")
    parser.add_argument("--out-dir", default=None, help="Output dataset_master_yolo path")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        # Default: letakkan di dalam base_dir itu sendiri, bukan parent-nya
        # GPU: /root/datasets/dataset_master_yolo
        out_dir = base_dir / "dataset_master_yolo"

    print("=" * 65)
    print("  GUIDIO — Comprehensive Dataset Merger (Potholes + Stairs + Obstacles)")
    print(f"  Base Dir: {base_dir}")
    print(f"  Output  : {out_dir}")
    print("=" * 65)

    if out_dir.exists():
        print(f"[!] Target directory {out_dir} already exists. Cleaning old files...")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    total_added = 0
    subdirs = [d for d in base_dir.rglob("*") if d.is_dir()]
    processed_paths = set()

    for d in sorted(subdirs):
        if d in processed_paths or any(p in processed_paths for p in d.parents):
            continue

        folder_name = d.name.lower()
        default_cls = None
        if "pothole" in folder_name or "hole" in folder_name or "damage" in folder_name:
            default_cls = "lubang"
        elif "stair" in folder_name or "step" in folder_name:
            default_cls = "tangga"

        has_imgs = any(d.glob("*.jpg")) or any(d.glob("*.png")) or any(d.rglob("*.jpg")) or any(d.rglob("*.png"))
        if has_imgs and ("train" in folder_name or "kaggle" in folder_name or "rf-" in folder_name or "stairs" in folder_name):
            prefix = d.name.replace("-", "_")
            print(f"\n Processing: {d.relative_to(base_dir)} (default_cls={default_cls})...")
            cnt = process_dataset_folder(d, out_dir, prefix=prefix, default_class=default_cls)
            print(f"   [✓] Added {cnt} annotated samples")
            total_added += cnt
            processed_paths.add(d)

    # Generate custom_navigasi.yaml
    yaml_content = f"""# Konfigurasi dataset untuk training YOLO11n (Deteksi Rintangan Mode Navigasi)
path: {out_dir}
train: images/train
val:   images/valid
test:  images/test

nc: 6
names:
  0: lubang        # pothole / lubang trotoar / lubang aspal
  1: got_terbuka   # manhole tanpa tutup / selokan terbuka
  2: tangga        # tangga (naik atau turun, digabung)
  3: orang         # person / pedestrian
  4: motor         # kendaraan bermotor (motor, mobil, bus, truk)
  5: tiang         # pole / obstacle / tiang listrik / gerobak PKL
"""
    # Coba tulis ke workspace repo (GPU: /workspace/guidio-cv-training-/configs/)
    # Fallback: di sebelah out_dir
    workspace_cfg = Path("/workspace/guidio-cv-training-/configs/custom_navigasi.yaml")
    if workspace_cfg.parent.exists():
        config_file = workspace_cfg
    else:
        config_file = out_dir.parent / "guido_cv_training" / "configs" / "custom_navigasi.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(yaml_content)

    print("\n" + "=" * 65)
    print(f" [OK] Unified dataset complete! Total {total_added} samples processed.")
    print(f" [OK] Updated config: {config_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
