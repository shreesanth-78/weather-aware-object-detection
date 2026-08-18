from pathlib import Path

# Project Root
ROOT = Path.cwd()

# ==============================
# Folder Structure
# ==============================

folders = [
    "datasets",
    "datasets/weather_classification",

    "datasets/weather_classification/clear",
    "datasets/weather_classification/rain",
    "datasets/weather_classification/fog",
    "datasets/weather_classification/night",

    "scripts",

    "logs",

    "models",

    "output",

    "src"
]

# ==============================
# Python Script Files
# ==============================

files = [
    "scripts/01_create_clear_dataset.py",
    "scripts/02_create_rain_dataset.py",
    "scripts/03_create_fog_dataset.py",
    "scripts/04_create_night_dataset.py",
    "scripts/05_split_dataset.py",
    "scripts/06_verify_dataset.py",
    "scripts/utils.py"
]

# ==============================
# Create folders
# ==============================

for folder in folders:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

# ==============================
# Create files
# ==============================

for file in files:
    path = ROOT / file
    path.touch(exist_ok=True)

print("\nProject structure created successfully!\n")

print(ROOT)