"""
06_verify_dataset.py

Verifies the weather classification dataset.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "datasets" / "weather_classification"

SPLITS = ["train", "val", "test"]
CLASSES = ["clear", "rain", "fog", "night"]

print("=" * 50)
print("WEATHER CLASSIFICATION DATASET")
print("=" * 50)

grand_total = 0

for split in SPLITS:

    print(f"\n{split.upper()}")

    split_total = 0

    for cls in CLASSES:

        folder = DATASET / split / cls

        count = len(list(folder.glob("*")))

        split_total += count

        print(f"{cls:<10}: {count}")

    grand_total += split_total

    print(f"Total {split:<5}: {split_total}")

print("\n" + "=" * 50)
print(f"TOTAL IMAGES : {grand_total}")
print("=" * 50)