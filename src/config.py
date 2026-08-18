"""
config.py
Centralized configuration file for the Weather Classification model.
Contains paths, hyperparameters, and device settings.
"""

import os
import torch

# ---------------------------------------------------------
# 1. DIRECTORY PATHS
# ---------------------------------------------------------
# os.path.dirname(__file__) gets the path to the 'src' folder
# We use os.path.join and '..' to navigate to the project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(ROOT_DIR, 'datasets', 'weather_classification')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Where we will save the trained ResNet-18 model weights
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'weather_classifier_resnet18.pth')

# ---------------------------------------------------------
# 2. DATASET PARAMETERS
# ---------------------------------------------------------
# ResNet-18 was originally trained on ImageNet, which expects 224x224 images
IMAGE_SIZE = 224
# The 4 classes we established in the custom dataset
CLASSES = ['clear', 'fog', 'night', 'rain']
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------
# 3. TRAINING HYPERPARAMETERS
# ---------------------------------------------------------
# Adjust batch size down (e.g., to 16) if your RTX 5050 runs out of VRAM, 
# though 8GB VRAM can comfortably handle 32 for ResNet-18.
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# ---------------------------------------------------------
# 4. HARDWARE CONFIGURATION
# ---------------------------------------------------------
# Automatically select the GPU we configured earlier, fallback to CPU if needed
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # A quick test block to verify the config file runs correctly
    print("="*40)
    print("Project Configuration Loaded")
    print("="*40)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Save Path: {MODEL_SAVE_PATH}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Classes: {CLASSES}")
    print(f"Device: {DEVICE}")