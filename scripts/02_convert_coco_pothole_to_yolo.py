"""
Script 02: Konversi pothole_2 (COCO JSON) → YOLO TXT

Sumber : pothole_2/data/annotations_coco.json + images/
Output : dataset_sidewalk/pothole_2_yolo/images/train/
         dataset_sidewalk/pothole_2_yolo/labels/train/
         (semua masuk train dulu, nanti dibagi ulang di script 03)

Kelas  : pothole → lubang (0), manhole → got_terbuka (1)
         crack   → DIBUANG (retak tidak kritis untuk navigasi tunanetra)

Catatan: labels-YOLO/ sudah ada di pothole_2/data/ tapi class-id-nya
         pakai skema lama yang tidak sesuai target kita, jadi konversi
         ulang dari annotations_coco.json (sumber kebenaran).
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.class_mapping import POTHOLE_2_TO_TARGET, YOLO_NAME_TO_ID


def main():
    parser = argparse.ArgumentParser(description="Konversi pothole_2 COCO JSON -> YOLO TXT")
    parser.add_argument("--dataset-root", default="/home/asadel/kuliah/lomba/smstr6/guido/dataset_sidewalk")
    args = parser.parse_args()

    root     = Path(args.dataset_root)
    coco_json = root / "pothole_2" / "data" / "annotations_coco.json"
    img_dir  = root / "pothole_2" / "data" / "images"
    out_root = root / "pothole_2_yolo"

    if not coco_json.exists():
        print(f"[!] File tidak ditemukan: {coco_json}")
        return

    coco = json.loads(coco_json.read_text())
    cat_id_to_name  = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_info  = {img["id"]: img for img in coco["images"]}
    anns_by_image   = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    img_out = out_root / "images" / "train"
    lbl_out = out_root / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    written, skipped_no_box, skipped_class = 0, 0, 0

    for img_id, info in img_id_to_info.items():
        file_name        = info["file_name"]
        img_w, img_h     = info["width"], info["height"]
        src_img_path     = img_dir / file_name
        if not src_img_path.exists():
            continue

        lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_name = cat_id_to_name.get(ann["category_id"], "")
            if cat_name not in POTHOLE_2_TO_TARGET:
                skipped_class += 1
                continue
            target_name = POTHOLE_2_TO_TARGET[cat_name]
            cls_id      = YOLO_NAME_TO_ID[target_name]
            x_min, y_min, w, h = ann["bbox"]
            x_c     = (x_min + w / 2) / img_w
            y_c     = (y_min + h / 2) / img_h
            w_norm  = w / img_w
            h_norm  = h / img_h
            lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")

        if not lines:
            skipped_no_box += 1
            continue

        shutil.copy2(src_img_path, img_out / file_name)
        stem = Path(file_name).stem
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
        written += 1

    print(f"[OK] pothole_2 -> YOLO selesai | Output: {out_root}")
    print(f"     ditulis={written}  tanpa-box-relevan={skipped_no_box}  skip-crack/lain={skipped_class}")


if __name__ == "__main__":
    main()
