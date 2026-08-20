"""
Script 01: Konversi pothole_1 (Pascal VOC XML) → YOLO TXT

Sumber : pothole_1/annotated-images/*.jpg + *.xml
Output : dataset_sidewalk/pothole_1_yolo/images/{train,valid,test}/
         dataset_sidewalk/pothole_1_yolo/labels/{train,valid,test}/

Split  : pakai splits.json jika ada, fallback 80/10/10 random.
Kelas  : pothole → lubang (class id 0)
"""
import argparse
import json
import random
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.class_mapping import POTHOLE_1_TO_TARGET, YOLO_NAME_TO_ID

RANDOM_SEED = 42


def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in POTHOLE_1_TO_TARGET:
            continue
        target_name = POTHOLE_1_TO_TARGET[cls_name]
        cls_id = YOLO_NAME_TO_ID[target_name]
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        x_c = ((xmin + xmax) / 2) / img_w
        y_c = ((ymin + ymax) / 2) / img_h
        w   = (xmax - xmin) / img_w
        h   = (ymax - ymin) / img_h
        boxes.append((cls_id, x_c, y_c, w, h))
    return boxes


def load_splits(splits_json: Path):
    if not splits_json.exists():
        return None
    try:
        data = json.loads(splits_json.read_text())
        if all(k in data for k in ("train", "valid", "test")):
            return data
        if all(k in data for k in ("train", "val", "test")):
            data["valid"] = data.pop("val")
            return data
    except Exception:
        pass
    return None


def random_split(stems, train=0.8, valid=0.1):
    rng = random.Random(RANDOM_SEED)
    stems = list(stems)
    rng.shuffle(stems)
    n = len(stems)
    n_train = int(n * train)
    n_valid = int(n * valid)
    return {
        "train": stems[:n_train],
        "valid": stems[n_train:n_train + n_valid],
        "test":  stems[n_train + n_valid:],
    }


def main():
    parser = argparse.ArgumentParser(description="Konversi pothole_1 VOC XML -> YOLO TXT")
    parser.add_argument("--dataset-root", default="/home/asadel/kuliah/lomba/smstr6/guido/dataset_sidewalk")
    args = parser.parse_args()

    root    = Path(args.dataset_root)
    src_dir = root / "pothole_1" / "annotated-images"
    out_root = root / "pothole_1_yolo"

    xml_files = sorted(src_dir.glob("*.xml"))
    if not xml_files:
        print(f"[!] Tidak ada .xml di {src_dir}")
        return

    stems  = [f.stem for f in xml_files]
    splits = load_splits(root / "pothole_1" / "splits.json") or random_split(stems)

    stem_to_split = {}
    for split_name, files in splits.items():
        for f in files:
            stem_to_split[Path(f).stem] = split_name

    counts  = {"train": 0, "valid": 0, "test": 0}
    skipped = 0

    for xml_path in xml_files:
        stem  = xml_path.stem
        split = stem_to_split.get(stem, "train")
        img_path = xml_path.with_suffix(".jpg")
        if not img_path.exists():
            candidates = [c for c in src_dir.glob(f"{stem}.*") if c.suffix.lower() != ".xml"]
            if not candidates:
                skipped += 1
                continue
            img_path = candidates[0]

        boxes = parse_voc_xml(xml_path)

        img_out = out_root / "images" / split
        lbl_out = out_root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, img_out / img_path.name)
        lbl_lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
        (lbl_out / f"{stem}.txt").write_text("\n".join(lbl_lines))
        counts[split] += 1

    print(f"[OK] pothole_1 -> YOLO selesai | Output: {out_root}")
    print(f"     train={counts['train']}  valid={counts['valid']}  test={counts['test']}  skip={skipped}")


if __name__ == "__main__":
    main()
