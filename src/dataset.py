"""
dataset.py
Handles image transformations, dataset loading, and PyTorch DataLoaders.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import our centralized variables
import config

def get_transforms():
    """
    Defines the image transformations for training and validation/testing.
    """
    # ImageNet statistics required for pretrained PyTorch models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transformations (includes augmentation to prevent overfitting)
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip horizontally
        transforms.RandomRotation(degrees=10),  # Slight random rotation
        transforms.ToTensor(),                  # Convert image to PyTorch Tensor (0-1 range)
        transforms.Normalize(mean=mean, std=std) # Normalize colors using ImageNet stats
    ])

    # Validation & Test transformations (strictly resizing and normalizing, NO augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_test_transform

def get_dataloaders():
    """
    Creates PyTorch DataLoaders from the directory structure.
    """
    train_transform, val_test_transform = get_transforms()

    # torchvision.datasets.ImageFolder automatically assigns labels based on subfolder names
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=config.VAL_DIR, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=val_test_transform)

    # DataLoaders handle shuffling and packaging the datasets into batches
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, # Shuffle training data so the model doesn't learn the order
        num_workers=4, # Use 4 CPU cores to load images faster
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, # No need to shuffle validation data
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    # Test block to verify everything works
    print("="*40)
    print("Testing DataLoaders")
    print("="*40)
    
    try:
        train_loader, val_loader, test_loader, class_names = get_dataloaders()
        
        print(f"Detected Classes: {class_names}")
        print(f"Number of Training Batches: {len(train_loader)}")
        print(f"Number of Validation Batches: {len(val_loader)}")
        print(f"Number of Test Batches: {len(test_loader)}")
        
        # Grab one single batch of data to inspect its shape
        images, labels = next(iter(train_loader))
        
        print("\nBatch Verification:")
        print(f"Images Tensor Shape: {images.shape} -> (Batch Size, Channels, Height, Width)")
        print(f"Labels Tensor Shape: {labels.shape} -> (Batch Size)")
        
        print("\nSUCCESS: dataset.py is functioning correctly.")
    except Exception as e:
        print(f"\nERROR: Failed to load datasets. Details: {e}")