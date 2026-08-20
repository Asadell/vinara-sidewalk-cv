"""
Script 04 (Extra): Konversi Dataset Segmentasi Tambahan -> dataset_master_seg/

Mendukung:
1. COCO JSON polygon (rf-sidewalk-v4gpn)
2. PNG Mask Semantic (rf-sidewalk-dz4ug & rf-sidewalk-zhrul)

Output:
  dataset_sidewalk/dataset_master_seg/images/{split}/*.jpg
  dataset_sidewalk/dataset_master_seg/masks/{split}/*.png
    nilai piksel: 0=non_walkable, 1=walkable, 2=hazard
"""
import argparse
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

SEG_NON_WALKABLE = 0
SEG_WALKABLE     = 1
SEG_HAZARD       = 2

SIDEWALK_SEG_TO_TARGET = {
    "Roadway":    SEG_NON_WALKABLE,
    "roadway":    SEG_NON_WALKABLE,
    "Sidewalks":  SEG_WALKABLE,
    "sidewalks":  SEG_WALKABLE,
    "Sidewalk":   SEG_WALKABLE,
    "sidewalk":   SEG_WALKABLE,
    "path":       SEG_WALKABLE,
    "Path":       SEG_WALKABLE,
    "Upstairs":   SEG_HAZARD,
    "upstairs":   SEG_HAZARD,
    "Downstairs": SEG_HAZARD,
    "downstairs": SEG_HAZARD,
}


def process_coco_folder(src_split_dir: Path, out_img_dir: Path, out_mask_dir: Path, prefix: str):
    ann_path = src_split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return 0

    try:
        coco = COCO(str(ann_path))
    except Exception:
        return 0

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
            mapped = SIDEWALK_SEG_TO_TARGET.get(cat_name, SEG_WALKABLE)
            anns_by_target[mapped].append(ann)

        for cls_id in [0, 1, 2]:
            for ann in anns_by_target[cls_id]:
                try:
                    seg = ann["segmentation"]
                    if isinstance(seg, list):
                        rles = mask_utils.frPyObjects(seg, img_h, img_w)
                        rle  = mask_utils.merge(rles)
                    elif isinstance(seg.get("counts"), list):
                        rle = mask_utils.frPyObjects(seg, img_h, img_w)
                    else:
                        rle = seg
                    bin_mask = mask_utils.decode(rle).astype(bool)
                    mask[bin_mask] = cls_id
                except Exception:
                    continue

        stem = f"{prefix}_{Path(file_name).stem}"
        try:
            Image.open(src_img).convert("RGB").save(out_img_dir / f"{stem}.jpg", quality=95)
            Image.fromarray(mask, mode="L").save(out_mask_dir / f"{stem}.png")
            written += 1
        except Exception:
            continue

    return written


def process_png_mask_folder(src_dir: Path, out_img_dir: Path, out_mask_dir: Path, prefix: str):
    """Processes PNG Semantic mask folders (image.jpg and image_mask.png)."""
    if not src_dir.exists():
        return 0

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    written = 0

    for img_file in img_files:
        if img_file.name.endswith("_mask.png"):
            continue

        # Look for corresponding mask file
        mask_candidates = [
            src_dir / f"{img_file.stem}_mask.png",
            src_dir / f"{img_file.stem}.mask.png",
        ]
        mask_file = None
        for cand in mask_candidates:
            if cand.exists():
                mask_file = cand
                break

        if not mask_file:
            continue

        try:
            img = Image.open(img_file).convert("RGB")
            raw_mask = Image.open(mask_file)
            mask_np = np.array(raw_mask)

            # Convert non-zero mask pixels to 1 (walkable)
            if mask_np.ndim == 3:
                mask_binary = (mask_np.sum(axis=-1) > 0).astype(np.uint8)
            else:
                mask_binary = (mask_np > 0).astype(np.uint8)

            stem = f"{prefix}_{img_file.stem}"
            dest_img = out_img_dir / f"{stem}.jpg"
            if dest_img.exists():
                dest_img.unlink()
            try:
                os.link(img_file, dest_img)
            except Exception:
                img.save(dest_img, quality=95)

            Image.fromarray(mask_binary, mode="L").save(out_mask_dir / f"{stem}.png")
            written += 1
        except Exception:
            continue

    return written


def main():
    parser = argparse.ArgumentParser(description="Convert extra sidewalk datasets to dataset_master_seg")
    parser.add_argument("--base-dir", default="/root/datasets/sidewalk-extra", help="Path to sidewalk-extra")
    parser.add_argument("--out-dir", default=None, help="Output dataset_master_seg path")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = base_dir.parent / "dataset_master_seg"

    print("=" * 65)
    print("  GUIDIO — Conversion of Extra Sidewalk Datasets to dataset_master_seg")
    print(f"  Base Dir: {base_dir}")
    print(f"  Out Dir : {out_dir}")
    print("=" * 65)

    total_added = 0
    if not base_dir.exists():
        print(f"[!] Path {base_dir} not found.")
        return

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        prefix = folder.name.replace("-", "_")
        print(f"\n Processing: {folder.name}...")

        for split in ("train", "valid", "test"):
            src_split = folder / split
            if not src_split.exists():
                src_split = folder

            out_img = out_dir / "images" / (split if split in ("train", "valid", "test") else "train")
            out_msk = out_dir / "masks" / (split if split in ("train", "valid", "test") else "train")

            # Try COCO first, then PNG mask
            cnt = process_coco_folder(src_split, out_img, out_msk, prefix)
            if cnt == 0:
                cnt = process_png_mask_folder(src_split, out_img, out_msk, prefix)

            if cnt > 0:
                print(f"   [✓] split={split}: added {cnt} image+mask pairs")
                total_added += cnt

    print("\n" + "=" * 65)
    print(f" [OK] Extra segmentation datasets processed! Added {total_added} pairs to {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
