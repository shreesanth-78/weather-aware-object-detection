"""
evaluate.py
Evaluates the trained Weather Classifier on the unseen Test Dataset.
Generates a Classification Report and a Confusion Matrix.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import WeatherClassifier

def evaluate_model():
    print("="*50)
    print("Starting Model Evaluation on Test Dataset")
    print("="*50)
    
    # 1. Load the Test DataLoader
    _, _, test_loader, class_names = get_dataloaders()
    
    # 2. Initialize the model and load the saved weights
    print(f"Loading saved weights from: {config.MODEL_SAVE_PATH}")
    model = WeatherClassifier(num_classes=config.NUM_CLASSES, pretrained=False)
    
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Please ensure you have trained the model and the .pth file exists.")
        return
        
    model = model.to(config.DEVICE)
    model.eval() # STRICTLY evaluation mode
    
    # Lists to store the true labels and the model's predictions
    y_true = []
    y_pred = []
    
    # 3. Run Inference on the Test Set
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(config.DEVICE)
            
            # Get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Move data back to CPU and convert to standard Python lists
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    # 4. Calculate and Print Metrics
    print("\n\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # 5. Generate and Save Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Weather Classification Confusion Matrix')
    plt.ylabel('Actual True Weather')
    plt.xlabel('Model Prediction')
    
    # Ensure the output directory exists
    output_dir = os.path.join(config.ROOT_DIR, 'output', 'evaluation_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {plot_path}")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()