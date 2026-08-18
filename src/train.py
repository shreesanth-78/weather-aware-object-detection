"""
train.py
Executes the training and validation loop for the Weather Classifier.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import WeatherClassifier

def train_model():
    # 1. Load Data
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    
    # 2. Initialize Model
    print(f"Initializing model on {config.DEVICE}...")
    model = WeatherClassifier(num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(config.DEVICE)
    
    # 3. Define Loss Function and Optimizer
    # CrossEntropyLoss is the mathematical standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer is highly efficient for adjusting neural network weights
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_accuracy = 0.0
    
    print("="*50)
    print(f"Starting Training Loop for {config.EPOCHS} Epochs")
    print("="*50)
    
    # 4. Training Loop
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        # --- TRAINING PHASE ---
        model.train() # Set model to training mode (enables dropout and batch normalization)
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # tqdm wraps the loader to give us a progress bar in the terminal
        loop = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in loop:
            # Move images and labels to the GPU
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Step A: Clear old gradients from the previous step
            optimizer.zero_grad()
            
            # Step B: Forward pass (make predictions)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Step C: Backward pass (calculate the error gradients)
            loss.backward()
            
            # Step D: Optimize (update the model weights based on gradients)
            optimizer.step()
            
            # Track running statistics for the epoch
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update the progress bar with the current loss
            loop.set_postfix(loss=loss.item())
            
        # Calculate average training loss and accuracy
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # --- VALIDATION PHASE ---
        model.eval() # Set model to evaluation mode (freezes dropout/batchnorm)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Disable gradient calculation to save GPU memory during validation
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        # Calculate average validation loss and accuracy
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Print epoch summary
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")
        
        # Save the best model
        if epoch_val_acc > best_val_accuracy:
            print(f"--> Validation accuracy improved from {best_val_accuracy:.4f} to {epoch_val_acc:.4f}. Saving model...")
            best_val_accuracy = epoch_val_acc
            
            # Save the internal state (weights and biases) of the model
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            
    print("="*50)
    print(f"Training Complete! Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved at: {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()