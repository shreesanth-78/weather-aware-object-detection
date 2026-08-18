"""
model.py
Defines the Weather Classifier architecture based on a pretrained ResNet-18.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

import config

class WeatherClassifier(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        """
        Initializes the ResNet-18 model and modifies the final layer.
        """
        super(WeatherClassifier, self).__init__()
        
        # Load the pretrained ResNet-18 model
        if pretrained:
            # ResNet18_Weights.DEFAULT loads the best available ImageNet weights
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=None)
            
        # Extract the number of input features going into the final fully connected (fc) layer
        num_ftrs = self.model.fc.in_features
        
        # Replace the final layer to output our specific number of classes (4)
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        """
        return self.model(x)

if __name__ == "__main__":
    # Test block to verify the architecture
    print("="*40)
    print("Testing Model Architecture")
    print("="*40)
    
    try:
        # Initialize the model
        model = WeatherClassifier(num_classes=config.NUM_CLASSES, pretrained=True)
        
        # Move the model to the configured device (GPU if available)
        model = model.to(config.DEVICE)
        print(f"Model successfully loaded and moved to {config.DEVICE}.")
        
        # Create a dummy tensor representing a single batch of images
        # Shape: (Batch Size, Channels, Height, Width) -> (32, 3, 224, 224)
        dummy_input = torch.randn(config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
        
        # Pass the dummy tensor through the model
        output = model(dummy_input)
        
        print("\nArchitecture Verification:")
        print(f"Dummy Input Shape:  {dummy_input.shape}")
        print(f"Model Output Shape: {output.shape} -> (Batch Size, Number of Classes)")
        
        if output.shape == (config.BATCH_SIZE, config.NUM_CLASSES):
            print("\nSUCCESS: model.py is functioning correctly and the final layer is modified.")
        else:
            print("\nWARNING: Output shape does not match expectations.")
            
    except Exception as e:
        print(f"\nERROR: Failed to load model. Details: {e}")