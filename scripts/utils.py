"""
utils.py

Common utility functions for Weather-Aware Detection Project
"""

from pathlib import Path
import shutil
from PIL import Image


# ==========================================================
# Create Folder
# ==========================================================

def create_folder(folder_path):
    """
    Create a folder if it does not exist.
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)


# ==========================================================
# Validate Image
# ==========================================================

def is_valid_image(image_path):
    """
    Returns True if image is readable.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ==========================================================
# Copy Image
# ==========================================================

def copy_image(source, destination):
    """
    Copy image while preserving metadata.
    """
    shutil.copy2(source, destination)


# ==========================================================
# Rename Image
# ==========================================================

def generate_filename(prefix, number, extension):
    """
    Example:
    clear_000001.jpg
    """

    return f"{prefix}_{number:06d}{extension}"


# ==========================================================
# Supported Image Formats
# ==========================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
]


# ==========================================================
# Find Images Recursively
# ==========================================================

def get_all_images(folder):

    folder = Path(folder)

    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.rglob(f"*{ext}"))
        images.extend(folder.rglob(f"*{ext.upper()}"))

    return sorted(images)


# ==========================================================
# Simple Logger
# ==========================================================

def log(message):

    print(message)


# ==========================================================
# Progress Printer
# ==========================================================

def progress(current, total):

    percent = (current / total) * 100

    print(
        f"\rProcessed {current}/{total} ({percent:.1f}%)",
        end=""
    )