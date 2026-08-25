#!/usr/bin/env python3
"""
Master Download Script for Vinara ML Training Datasets
Downloads: Pothole, Stairs, Sidewalk datasets from Kaggle & Roboflow
Base directory: ~/datasets/
"""

import os
import subprocess
import sys
from roboflow import Roboflow

BASE_DIR = os.path.expanduser("~/datasets")
os.makedirs(BASE_DIR, exist_ok=True)

# ============ KAGGLE DATASETS ============
kaggle_pothole_datasets = [
    ("anggadwisunarto/potholes-detection-yolov8", "pothole/kaggle-potholes-yolov8"),
    ("denisg04/pothle-detect", "pothole/kaggle-pothole-v8-detect"),
    ("andrewmvd/pothole-detection", "pothole/kaggle-pothole-detection-larxel"),
    ("abhinavkulshreshth/pothole-detection-dataset", "pothole/kaggle-pothole-cnn"),
    ("chitholian/annotated-potholes-dataset", "pothole/kaggle-annotated-potholes"),
]

kaggle_stairs_datasets = [
    ("dataclusterlabs/stairs-image-dataset", "stairs/kaggle-stairs-dataclusterlabs"),
    ("samuelayman/stairs", "stairs/kaggle-stairs-samuelayman"),
]

# ============ ROBOFLOW DATASETS ============
# (workspace, project, version, format, folder_name)
roboflow_pothole_datasets = [
    ("intel-unnati-training-program", "pothole-detection-bqu6s", 9, "yolov8", "pothole/rf-pothole-intel-unnati"),
    ("projects-hjaax", "pothole-detection-using-yolov5", 1, "yolov5", "pothole/rf-pothole-yolov5"),
    ("roaddamage-ak8w6", "road-damage-uyvns", 3, "yolov8", "pothole/rf-road-damage"),
    ("imacs-pothole-detection-wo8mu", "pothole-detection-irkz9", 4, "coco-segmentation", "pothole/rf-pothole-segmentation"),
]

roboflow_stairs_datasets = [
    ("stair-eyhvv", "stairs-detection-6cq2a", 1, "yolov8", "stairs/rf-stairs-updown"),
    ("tesisusbbog", "stairs-data", 1, "yolov8", "stairs/rf-stairs-data"),
    ("group10textdetect", "stair-detect", 1, "yolov8", "stairs/rf-stair-detect"),
]

roboflow_sidewalk_datasets = [
    ("project-nlr2u", "sidewalk-segmentation-v4gpn", 1, "coco-segmentation", "sidewalk/rf-sidewalk-v4gpn"),
    ("school-stpl7", "sidewalk-dz4ug", 1, "png-mask-semantic", "sidewalk/rf-sidewalk-dz4ug"),
    ("project-ii3cz", "sidewalk-1smxs", 2, "yolov8", "sidewalk/rf-sidewalk-1smxs"),
    ("happy-jmswg", "sidewalk-eqnxe", 1, "yolov8", "sidewalk/rf-sidewalk-eqnxe"),
    ("dika-biyq4", "sidewalk-zhrul", 2, "png-mask-semantic", "sidewalk/rf-sidewalk-zhrul"),
    ("alharth-alhaj-hussein-1hig7", "sidewalk-road", 1, "yolov8", "sidewalk/rf-sidewalk-road"),
    ("project-xagxj", "road_segment_v2", 1, "yolov8", "sidewalk/rf-road-segment-v2"),
    ("project-xagxj", "road_person_recognition", 1, "yolov8", "sidewalk/rf-road-person-rec"),
    ("project-xagxj", "road_person_recognition_v2", 1, "yolov8", "sidewalk/rf-road-person-rec-v2"),
    ("sidewalk", "sidewalk-segmentation", 1, "coco-segmentation", "sidewalk/rf-sidewalk-seg"),
    ("sidewalk", "sidewalks-seg", 1, "coco-segmentation", "sidewalk/rf-sidewalks-seg"),
    ("senior-design-scl0l", "sidewalk_semantics_segmentation", 1, "png-mask-semantic", "sidewalk/rf-sidewalk-semantics"),
    ("yolo-s6mwf", "sidewalk-6imhx", 1, "yolov8", "sidewalk/rf-sidewalk-6imhx"),
    ("fieldlinedetection", "sidewalk_test", 1, "yolov8", "sidewalk/rf-sidewalk-test"),
    ("elvis-cmqng", "sidewalk-and-stair-train-image", 1, "yolov8", "sidewalk/rf-sidewalk-stair"),
    ("capstone-project-nhlns", "sidewalk-detection-ykwpf", 1, "yolov8", "sidewalk/rf-sidewalk-det-ykwpf"),
    ("do-hvtgm", "test_sidewalk_1", 1, "yolov8", "sidewalk/rf-test-sidewalk-1"),
    ("projects-5k1o6", "sidewalk-dlu6l", 1, "yolov8", "sidewalk/rf-sidewalk-dlu6l"),
]

roboflow_pole_datasets = [
    ("poleproject", "pole-detection-v2-bvqug", 1, "yolov8", "pole/rf-pole-detection-v2"),
    ("akshay-anand-bfabb", "pole-data", 1, "yolov8", "pole/rf-pole-data"),
]

# ============ MAIN SCRIPT ============

def download_kaggle_datasets(datasets_list, category_name):
    """Download dari Kaggle dengan error handling"""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING KAGGLE {category_name.upper()} DATASETS")
    print(f"{'='*60}")
    
    success_count = 0
    failed_count = 0
    
    for slug, rel_path in datasets_list:
        target = os.path.join(BASE_DIR, rel_path)
        os.makedirs(target, exist_ok=True)
        
        print(f"\n>>> [{success_count + failed_count + 1}/{len(datasets_list)}] {slug}")
        try:
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", slug,
                "-p", target,
                "--unzip"
            ], check=True, capture_output=True)
            print(f"    ✓ Success")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Failed: {e}")
            failed_count += 1
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            failed_count += 1
    
    return success_count, failed_count


def download_roboflow_datasets(datasets_list, category_name):
    """Download dari Roboflow dengan error handling & version fallback loop"""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING ROBOFLOW {category_name.upper()} DATASETS")
    print(f"{'='*60}")
    
    try:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            print("✗ ERROR: ROBOFLOW_API_KEY not set. Export it first:")
            print("  export ROBOFLOW_API_KEY='your_key_here'")
            return 0, len(datasets_list)
        
        rf = Roboflow(api_key=api_key)
    except Exception as e:
        print(f"✗ Failed to initialize Roboflow: {e}")
        return 0, len(datasets_list)
    
    success_count = 0
    failed_count = 0
    
    for workspace, project, version, fmt, rel_path in datasets_list:
        target = os.path.join(BASE_DIR, rel_path)
        print(f"\n>>> [{success_count + failed_count + 1}/{len(datasets_list)}] {workspace}/{project} ({fmt})")
        
        downloaded = False
        versions_to_try = [version] + [v for v in range(1, 11) if v != version]
        for v in versions_to_try:
            try:
                proj = rf.workspace(workspace).project(project)
                proj.version(v).download(fmt, location=target)
                print(f"    ✓ Success (v{v}) -> {target}")
                success_count += 1
                downloaded = True
                break
            except Exception as err:
                continue
                
        if not downloaded:
            print(f"    ✗ Failed to download any version (v1..v10)")
            print(f"    → Check URL at: https://universe.roboflow.com/{workspace}/{project}")
            failed_count += 1
    
    return success_count, failed_count


def print_summary(all_results):
    """Print hasil download summary"""
    print(f"\n\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    total_success = 0
    total_failed = 0
    
    for category, (success, failed) in all_results.items():
        total_success += success
        total_failed += failed
        status = "✓" if failed == 0 else "⚠"
        print(f"{status} {category:30} {success:2d} success, {failed:2d} failed")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_success} success, {total_failed} failed")
    print(f"{'='*60}")
    print(f"\nAll datasets saved to: {BASE_DIR}/")
    print("\nDirectory structure:")
    print(f"  {BASE_DIR}/")
    print(f"  ├── pothole/")
    print(f"  │   ├── kaggle-*/ (5 datasets)")
    print(f"  │   └── rf-*/ (4 datasets)")
    print(f"  ├── stairs/")
    print(f"  │   ├── kaggle-*/ (2 datasets)")
    print(f"  │   └── rf-*/ (3 datasets)")
    print(f"  └── sidewalk/")
    print(f"      └── rf-*/ (5 datasets)")
    
    if total_failed > 0:
        print(f"\n⚠ {total_failed} dataset(s) failed. Check errors above and retry manually if needed.")
        return 1
    else:
        print(f"\n✓ All datasets downloaded successfully!")
        return 0


def main():
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║   VINARA ML TRAINING DATASETS DOWNLOADER              ║")
    print("║   Pothole + Stairs + Sidewalk (Kaggle + Roboflow)     ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    # Pre-flight checks
    print("\n🔍 Pre-flight checks...")
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        print("  ✓ Kaggle CLI found")
    except:
        print("  ✗ Kaggle CLI not found. Install: pip install kaggle --break-system-packages")
        return 1
    
    # Kaggle mendukung dua format credential:
    # 1. ~/.kaggle/access_token  (format baru, isi: KGAT_xxxx)
    # 2. ~/.kaggle/kaggle.json   (format lama, isi: {"username":..., "key":...})
    kaggle_token  = os.path.expanduser("~/.kaggle/access_token")
    kaggle_json   = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_token) and not os.path.exists(kaggle_json):
        print("  ✗ Kaggle credentials tidak ditemukan.")
        print("    Untuk format baru (KGAT_xxx):")
        print("      mkdir -p ~/.kaggle && echo 'KGAT_xxxx' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token")
        print("    Untuk format lama (kaggle.json):")
        print("      Download dari: https://www.kaggle.com/settings -> API -> Create New Token")
        return 1
    else:
        found = kaggle_token if os.path.exists(kaggle_token) else kaggle_json
        print(f"  ✓ Kaggle credentials found ({os.path.basename(found)})")
    
    if not os.environ.get("ROBOFLOW_API_KEY"):
        print("  ⚠ ROBOFLOW_API_KEY not set (Roboflow datasets will be skipped)")
        print("    → Run: export ROBOFLOW_API_KEY='your_key_here'")
    else:
        print("  ✓ Roboflow API key found")
    
    # Download
    all_results = {}
    
    # Pothole
    success, failed = download_kaggle_datasets(kaggle_pothole_datasets, "pothole")
    all_results["Kaggle Pothole"] = (success, failed)
    
    success, failed = download_roboflow_datasets(roboflow_pothole_datasets, "pothole")
    all_results["Roboflow Pothole"] = (success, failed)
    
    # Stairs
    success, failed = download_kaggle_datasets(kaggle_stairs_datasets, "stairs")
    all_results["Kaggle Stairs"] = (success, failed)
    
    success, failed = download_roboflow_datasets(roboflow_stairs_datasets, "stairs")
    all_results["Roboflow Stairs"] = (success, failed)
    
    # Sidewalk
    success, failed = download_roboflow_datasets(roboflow_sidewalk_datasets, "sidewalk")
    all_results["Roboflow Sidewalk"] = (success, failed)
    
    # Pole (Tiang)
    success, failed = download_roboflow_datasets(roboflow_pole_datasets, "pole")
    all_results["Roboflow Pole"] = (success, failed)
    
    # Summary & exit
    return print_summary(all_results)


if __name__ == "__main__":
    sys.exit(main())