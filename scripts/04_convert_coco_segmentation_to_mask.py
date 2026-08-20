"""
Script 04: Konversi Sidewalk Segmentation (COCO JSON polygon) → PNG Mask 8-bit

Sumber : Sidewalk Segmentation.v1i.coco-segmentation/{train,valid,test}/
Output : dataset_sidewalk/dataset_master_seg/images/{split}/*.jpg
         dataset_sidewalk/dataset_master_seg/masks/{split}/*.png
           nilai piksel: 0=non_walkable, 1=walkable, 2=hazard

Prioritas overlap: hazard (2) menimpa walkable (1) menimpa non_walkable (0)
→ lubang/tangga di atas trotoar tetap tergambar sebagai hazard.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.class_mapping import SIDEWALK_SEG_TO_TARGET, SEG_NON_WALKABLE

DRAW_PRIORITY = [0, 1, 2]  # digambar urut; index lebih tinggi 'menang' di overlap


def ann_to_binary_mask(coco, ann, img_h, img_w):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, img_h, img_w)
        rle  = mask_utils.merge(rles)
    elif isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, img_h, img_w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def process_split(src_split_dir, out_img_dir, out_mask_dir):
    ann_path = src_split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"  [!] {ann_path} tidak ditemukan, skip split ini.")
        return 0

    coco = COCO(str(ann_path))
    cat_id_to_name = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for img_id, info in coco.imgs.items():
        file_name = info["file_name"]
        img_h, img_w = info["height"], info["width"]
        src_img = src_split_dir / file_name
        if not src_img.exists():
            continue

        mask = np.full((img_h, img_w), SEG_NON_WALKABLE, dtype=np.uint8)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        anns_by_target = {0: [], 1: [], 2: []}
        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"], "")
            if cat_name in SIDEWALK_SEG_TO_TARGET:
                anns_by_target[SIDEWALK_SEG_TO_TARGET[cat_name]].append(ann)

        for cls in DRAW_PRIORITY:
            for ann in anns_by_target[cls]:
                mask[ann_to_binary_mask(coco, ann, img_h, img_w)] = cls

        stem = Path(file_name).stem
        Image.open(src_img).convert("RGB").save(out_img_dir / f"{stem}.jpg", quality=95)
        Image.fromarray(mask, mode="L").save(out_mask_dir / f"{stem}.png")
        written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description="Konversi Sidewalk Segmentation COCO -> PNG mask")
    parser.add_argument("--dataset-root", default="/home/asadel/kuliah/lomba/smstr6/guido/dataset_sidewalk")
    args     = parser.parse_args()
    root     = Path(args.dataset_root)
    src_root = root / "Sidewalk Segmentation.v1i.coco-segmentation"
    out_root = root / "dataset_master_seg"

    if not src_root.exists():
        print(f"[!] Folder tidak ditemukan: {src_root}")
        return

    total = 0
    for split in ("train", "valid", "test"):
        src_split = src_root / split
        if not src_split.exists():
            continue
        n = process_split(src_split, out_root / "images" / split, out_root / "masks" / split)
        print(f"  split={split}: {n} pasangan image+mask")
        total += n

    print(f"[OK] Dataset segmentasi siap di: {out_root}")
    print(f"     Total: {total} pasangan image+mask (nilai piksel mask: 0/1/2)")


if __name__ == "__main__":
    main()
