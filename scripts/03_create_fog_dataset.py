"""
03_create_fog_dataset.py

Creates a balanced FOG dataset from RTTS
"""

import random
from pathlib import Path

from utils import (
    create_folder,
    copy_image,
    generate_filename,
    get_all_images,
    is_valid_image,
    progress,
)

MAX_IMAGES = 4000
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RTTS_IMAGES = PROJECT_ROOT / "downloads" / "RTTS" / "RTTS" / "JPEGImages"

OUTPUT = PROJECT_ROOT / "datasets" / "weather_classification" / "fog"

create_folder(OUTPUT)

images = get_all_images(RTTS_IMAGES)

print(f"Fog images available : {len(images)}")

random.seed(RANDOM_SEED)
random.shuffle(images)

images = images[:MAX_IMAGES]

print(f"Selected : {len(images)}")

copied = 0

for idx, image_path in enumerate(images, start=1):

    if not is_valid_image(image_path):
        continue

    new_name = generate_filename(
        "fog",
        copied + 1,
        image_path.suffix
    )

    copy_image(
        image_path,
        OUTPUT / new_name
    )

    copied += 1

    progress(idx, len(images))

print("\n")
print("="*50)
print("FOG DATASET CREATED")
print("="*50)
print(f"Images Copied : {copied}")
print("="*50)