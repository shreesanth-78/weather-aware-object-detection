"""
pipeline.py
The Master Routing Pipeline.
Integrates Weather Classification, Dynamic Image Enhancement, and Object Detection.
"""

import os
import glob
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Import configurations and local modules
import config
from model import WeatherClassifier
from enhance_fog import dehaze_image
from enhance_rain import derain_image
from enhance_night import brighten_image
from detector import detect_objects

# ---------------------------------------------------------
# 1. Initialize Weather Classifier
# ---------------------------------------------------------
def load_classifier():
    print(f"Loading Custom Weather Classifier (ResNet-18) on {config.DEVICE}...")
    model = WeatherClassifier(num_classes=config.NUM_CLASSES, pretrained=False)
    
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"ERROR: Failed to load Classifier. Details: {e}")
        return None

classifier_model = load_classifier()

# ImageNet normalization standard (must exactly match dataset.py)
classifier_transforms = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_weather(image_bgr):
    """Predicts weather condition from a raw OpenCV BGR image."""
    if classifier_model is None:
        return "clear" # Fallback if classifier fails
        
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    input_tensor = classifier_transforms(pil_img).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        
    return config.CLASSES[class_idx]

# ---------------------------------------------------------
# 2. Master Processing Function
# ---------------------------------------------------------
def process_frame(image_path):
    """Executes the end-to-end weather-aware pipeline on a single image."""
    print("\n" + "-"*50)
    print(f"Processing: {os.path.basename(image_path)}")
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Error: Could not read image.")
        return
        
    # --- STEP A: WEATHER CLASSIFICATION ---
    predicted_weather = predict_weather(original_img)
    print(f"Classification : [{predicted_weather.upper()}]")
    
    # --- STEP B: DYNAMIC ROUTING & ENHANCEMENT ---
    enhanced_img = original_img.copy()
    
    if predicted_weather == 'fog':
        print("Routing        : DehazeFormer-S")
        enhanced_img = dehaze_image(original_img)
        
    elif predicted_weather == 'rain':
        print("Routing        : Restormer")
        enhanced_img = derain_image(original_img)
        
    elif predicted_weather == 'night':
        print("Routing        : Zero-DCE++")
        enhanced_img = brighten_image(original_img)
        
    elif predicted_weather == 'clear':
        print("Routing        : Bypassing enhancement (Clear Weather)")
        
    # --- STEP C: OBJECT DETECTION ---
    print("Detecting      : YOLO11n")
    annotated_img, _ = detect_objects(enhanced_img)
    
    # --- STEP D: VISUALIZATION & SAVING ---
    # Draw the detected weather class on the original image for comparison
    cv2.putText(original_img, f"Raw Input ({predicted_weather.upper()})", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
    cv2.putText(annotated_img, "Enhanced + YOLO11", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    comparison = np.hstack((original_img, annotated_img))
    
    out_filename = f"pipeline_out_{predicted_weather}_{os.path.basename(image_path)}"
    out_path = os.path.join(config.ROOT_DIR, 'output', out_filename)
    cv2.imwrite(out_path, comparison)
    
    print(f"Result Saved   : {out_path}")

# ---------------------------------------------------------
# 3. Pipeline Automated Test
# ---------------------------------------------------------
if __name__ == "__main__":
    print("============================================================")
    print("INITIALIZING ADAPTIVE WEATHER-AWARE PIPELINE")
    print("============================================================")
    
    # Automatically grab one image from each weather class in the Test Set
    test_images = []
    for condition in config.CLASSES:
        pattern = os.path.join(config.DATA_DIR, 'test', condition, '*.[jp][pn]*')
        imgs = glob.glob(pattern)
        if imgs:
            test_images.append(imgs[0]) # Grab the first available image
            
    if not test_images:
        print("ERROR: Could not find test images.")
    else:
        print(f"Initiating batch processing for {len(test_images)} scenarios...")
        for img_path in test_images:
            process_frame(img_path)
            
    print("\n============================================================")
    print("PIPELINE TEST COMPLETE.")
    print("============================================================")