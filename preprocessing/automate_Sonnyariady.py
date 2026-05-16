"""
Automated preprocessing script for Indonesian Food Classification.

Jalankan dari mana saja. Output dataset akan selalu dibuat di folder yang sama
dengan script ini:

preprocessing/
├── automate_Sonnyariady.py
├── Eksperimen_Sonnyariady.ipynb
└── namadataset_preprocessing/
"""

import json
import shutil
from pathlib import Path

import kagglehub

DATASET = "rizkyyk/dataset-food-classification"
MAX_CLASSES = 5
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "namadataset_preprocessing"
LABELS_PATH = SCRIPT_DIR.parent / "labels.json"


def find_class_folders(root: Path):
    candidates = []
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue

        direct_images = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

        if len(direct_images) >= 50:
            candidates.append((folder, len(direct_images)))

    return sorted(candidates, key=lambda item: item[1], reverse=True)


def main():
    print("Script dir:", SCRIPT_DIR)
    print("Output dir:", OUTPUT_DIR)

    dataset_path = Path(kagglehub.dataset_download(DATASET))
    print("Kaggle dataset path:", dataset_path)

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
        target = OUTPUT_DIR / folder.name
        shutil.copytree(folder, target)
        print(f"Copied {folder.name}: {count} images -> {target}")

    with LABELS_PATH.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print("Preprocessing selesai.")
    print("Output:", OUTPUT_DIR.resolve())
    print("Labels:", labels)
    print("Labels JSON:", LABELS_PATH.resolve())


if __name__ == "__main__":
    main()
