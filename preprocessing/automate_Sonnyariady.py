"""
Automated preprocessing script for Indonesian Food Classification.
This script downloads the Kaggle dataset, selects class folders, and copies them
into a ready-to-train folder named namadataset_preprocessing.
"""

import json
import shutil
from pathlib import Path

import kagglehub

DATASET = "rizkyyk/dataset-food-classification"
MAX_CLASSES = 5
OUTPUT_DIR = Path("namadataset_preprocessing")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_folders(root: Path):
    candidates = []
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        direct_images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if len(direct_images) >= 50:
            candidates.append((folder, len(direct_images)))
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def main():
    dataset_path = Path(kagglehub.dataset_download(DATASET))
    candidates = find_class_folders(dataset_path)
    if len(candidates) < 3:
        raise RuntimeError("Tidak menemukan minimal 3 folder kelas yang berisi gambar.")

    selected = candidates[:MAX_CLASSES]
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = []
    for folder, count in selected:
        labels.append(folder.name)
        shutil.copytree(folder, OUTPUT_DIR / folder.name)
        print(f"Copied {folder.name}: {count} images")

    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print("Preprocessing selesai.")
    print("Output:", OUTPUT_DIR.resolve())
    print("Labels:", labels)


if __name__ == "__main__":
    main()
