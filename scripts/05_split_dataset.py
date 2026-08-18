"""
05_split_dataset.py

Splits the weather classification dataset into
Train / Validation / Test folders.
"""

import random
import shutil
from pathlib import Path

# ==========================================================
# SETTINGS
# ==========================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets" / "weather_classification"

CLASSES = [
    "clear",
    "rain",
    "fog",
    "night"
]

random.seed(RANDOM_SEED)

# ==========================================================
# SPLIT DATASET
# ==========================================================

for class_name in CLASSES:

    print(f"\nProcessing class: {class_name}")

    source_folder = DATASET_ROOT / class_name

    images = list(source_folder.glob("*"))

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        destination = DATASET_ROOT / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, destination / image.name)

    print(f"Total Images : {total}")
    print(f"Train : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Test : {len(test_images)}")

print("\n========================================")
print("DATASET SPLITTING COMPLETED")
print("========================================")