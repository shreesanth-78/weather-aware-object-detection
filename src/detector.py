"""
detector.py
Wrapper for the YOLO11 Object Detection model.
Handles initialization, inference, and bounding box visualization.
"""

import os
import cv2
from ultralytics import YOLO

import config

def load_yolo():
    """
    Initializes the YOLO11 Nano model. 
    Ultralytics will automatically download the yolo11n.pt weights if they don't exist.
    """
    print(f"Loading YOLO11n on {config.DEVICE}...")
    try:
        # We use YOLO11 'nano' (n) for maximum speed. 
        # (You could change this to 'yolo11s.pt' later if you want slightly higher accuracy)
        model = YOLO('yolo11n.pt') 
        model.to(config.DEVICE)
        print("============================================================")
        print("SUCCESS: YOLO11 loaded successfully.")
        print("============================================================")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load YOLO11. Details: {e}")
        return None

# Initialize the model once in memory
yolo_model = load_yolo()

def detect_objects(image_bgr):
    """
    Takes an OpenCV BGR image, runs YOLO11 detection, and returns the 
    annotated image along with the raw detection results.
    """
    if yolo_model is None:
        return image_bgr, None
        
    # Run inference. verbose=False keeps the terminal clean during pipeline loops
    results = yolo_model(image_bgr, verbose=False)
    
    # YOLO results is a list (one for each image in the batch). We only passed one image.
    result = results[0]
    
    # result.plot() automatically draws the bounding boxes, labels, and confidence scores
    annotated_image = result.plot()
    
    return annotated_image, result

# ---------------------------------------------------------
# Standalone Visual Test
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n============================================================")
    print("Testing YOLO11 Standalone")
    print("============================================================")
    
    if yolo_model is None:
        print("Test aborted. Model could not be loaded.")
    else:
        import glob
        
        # Let's test on a clear image to see baseline performance
        clear_dir = os.path.join(config.DATA_DIR, 'test', 'clear')
        # Handle multiple possible extensions
        search_pattern = os.path.join(clear_dir, '*.[jp][pn]*') 
        
        clear_images = glob.glob(search_pattern)
        
        if not clear_images:
            print(f"Could not find any test images in {clear_dir}.")
        else:
            test_img_path = clear_images[0]
            print(f"Testing on image: {test_img_path}")
            
            original_img = cv2.imread(test_img_path)
            
            print("Running YOLO11 inference...")
            annotated_img, raw_results = detect_objects(original_img)
            
            # Print how many objects were found
            num_objects = len(raw_results.boxes)
            print(f"\nObjects detected: {num_objects}")
            
            save_path = os.path.join(config.ROOT_DIR, 'output', 'yolo_test_result.jpg')
            cv2.imwrite(save_path, annotated_img)
            
            print(f"\nSUCCESS: Annotated image saved to: {save_path}")
            print("============================================================")
            print("SUCCESS: YOLO11 test completed.")
            print("============================================================")