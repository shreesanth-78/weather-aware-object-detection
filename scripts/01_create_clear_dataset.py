"""
01_create_clear_dataset.py

Creates a balanced CLEAR dataset from BDD100K
"""

import json
import random
from pathlib import Path

from utils import (
    create_folder,
    copy_image,
    generate_filename,
    is_valid_image,
    progress,
)

# ======================================================
# SETTINGS
# ======================================================

MAX_IMAGES = 4000
RANDOM_SEED = 42

# ======================================================
# PATHS
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_JSON = PROJECT_ROOT / "downloads" / "BDD100K" / "labels" / "det_v2_train_release.json"
VAL_JSON = PROJECT_ROOT / "downloads" / "BDD100K" / "labels" / "det_v2_val_release.json"

TRAIN_IMAGES = PROJECT_ROOT / "downloads" / "BDD100K" / "bdd100k" / "bdd100k" / "images" / "100k" / "train"
VAL_IMAGES = PROJECT_ROOT / "downloads" / "BDD100K" / "bdd100k" / "bdd100k" / "images" / "100k" / "val"

OUTPUT = PROJECT_ROOT / "datasets" / "weather_classification" / "clear"

create_folder(OUTPUT)

# ======================================================
# LOAD ANNOTATIONS
# ======================================================

print("Loading annotations...")

with open(TRAIN_JSON, "r") as f:
    train_data = json.load(f)

with open(VAL_JSON, "r") as f:
    val_data = json.load(f)

all_data = train_data + val_data

print(f"Total annotations : {len(all_data)}")

# ======================================================
# FILTER CLEAR IMAGES
# ======================================================

clear_images = [
    item for item in all_data
    if item["attributes"]["weather"] == "clear"
]

print(f"Total clear images available : {len(clear_images)}")

# ======================================================
# RANDOM SELECTION
# ======================================================

random.seed(RANDOM_SEED)
random.shuffle(clear_images)

clear_images = clear_images[:MAX_IMAGES]

print(f"Selected images : {len(clear_images)}")

# ======================================================
# COPY IMAGES
# ======================================================

copied = 0

for idx, item in enumerate(clear_images, start=1):

    image_name = item["name"]

    train_path = TRAIN_IMAGES / image_name
    val_path = VAL_IMAGES / image_name

    if train_path.exists():
        image_path = train_path
    elif val_path.exists():
        image_path = val_path
    else:
        continue

    if not is_valid_image(image_path):
        continue

    new_name = generate_filename(
        "clear",
        copied + 1,
        image_path.suffix
    )

    copy_image(
        image_path,
        OUTPUT / new_name
    )

    copied += 1

    progress(idx, len(clear_images))

print("\n")
print("=" * 50)
print("CLEAR DATASET CREATED")
print("=" * 50)
print(f"Images Copied : {copied}")
print("=" * 50)