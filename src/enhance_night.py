"""
enhance_night.py
Implements Zero-DCE++ for ultra-fast, state-of-the-art low-light enhancement.
"""

import os
import cv2
import torch
import numpy as np

import config
from models_arch.zerodce import ZeroDCE_PP

# ---------------------------------------------------------
# 1. Model Initialization & Strict Weight Loading
# ---------------------------------------------------------
NIGHT_WEIGHTS_PATH = os.path.join(config.MODEL_SAVE_DIR, 'pretrained_enhancers', 'zerodce_pp.pth')

def load_zerodce():
    """Initializes Zero-DCE++ and performs strict weight verification."""
    if not os.path.exists(NIGHT_WEIGHTS_PATH):
        print(f"ERROR: Weights not found at {NIGHT_WEIGHTS_PATH}")
        return None

    print(f"Loading Zero-DCE++ on {config.DEVICE}...")
    
    # Initialize the architecture
    model = ZeroDCE_PP()
    
    try:
        # Load the checkpoint directly as an OrderedDict
        checkpoint = torch.load(NIGHT_WEIGHTS_PATH, map_location=config.DEVICE)
        
        # Verify it's not nested
        state_dict = checkpoint['state_dict'] if (isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint
            
        # Clean "module." prefix if it exists
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            clean_state_dict[new_key] = v
            
        # Strict Key Verification
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(clean_state_dict.keys())
        
        matching_keys = model_keys.intersection(ckpt_keys)
        missing_keys = model_keys - ckpt_keys
        unexpected_keys = ckpt_keys - model_keys
        
        print(f"Weight Verification: {len(matching_keys)} keys match.")
        
        if missing_keys or unexpected_keys:
            print("============================================================")
            print("ERROR: STRICT WEIGHT VERIFICATION FAILED")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            if missing_keys: print(f"Sample missing: {list(missing_keys)[:3]}")
            if unexpected_keys: print(f"Sample unexpected: {list(unexpected_keys)[:3]}")
            print("============================================================")
            return None
            
        # Load weights strictly
        model.load_state_dict(clean_state_dict, strict=True)
        model = model.to(config.DEVICE)
        model.eval()
        print("============================================================")
        print("SUCCESS: Zero-DCE++ loaded successfully.")
        print("============================================================")
        return model
        
    except Exception as e:
        print(f"Failed to load Zero-DCE++ weights: {e}")
        return None

# Initialize the model once in memory
night_model = load_zerodce()

# ---------------------------------------------------------
# 2. Main Wrapper Function for Pipeline
# ---------------------------------------------------------
def brighten_image(image_bgr):
    """
    Takes a raw OpenCV BGR image, runs it through Zero-DCE++, 
    and returns the enhanced BGR image.
    """
    if night_model is None:
        return image_bgr # Fail-safe
        
    with torch.no_grad():
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_normalized = np.float32(img_rgb) / 255.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(config.DEVICE)
        
        # Zero-DCE++ does not downsample, so it intrinsically supports arbitrary 
        # input resolutions without padding logic.
        enhanced_tensor = night_model(img_tensor)

        # Convert back to OpenCV BGR (0-255)
        enhanced_numpy = enhanced_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        enhanced_numpy = np.clip(enhanced_numpy, 0, 1) * 255.0
        enhanced_bgr = cv2.cvtColor(enhanced_numpy.astype(np.uint8), cv2.COLOR_RGB2BGR)

    return enhanced_bgr

# ---------------------------------------------------------
# 3. Standalone Visual Test
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n============================================================")
    print("Testing Zero-DCE++ Standalone")
    print("============================================================")
    
    if night_model is None:
        print("Test aborted. Model could not be loaded.")
    else:
        import glob
        
        night_dir = os.path.join(config.DATA_DIR, 'test', 'night')
        # Adjust extension depending on ExDark's format (usually jpg, jpeg, or png)
        search_pattern = os.path.join(night_dir, '*.[jp][pn]*') 
        
        night_images = glob.glob(search_pattern)
        
        if not night_images:
            print(f"Could not find any test images in {night_dir}.")
        else:
            test_img_path = night_images[0]
            print(f"Testing on image: {test_img_path}")
            
            original_img = cv2.imread(test_img_path)
            
            print("Running Zero-DCE++ inference...")
            enhanced_img = brighten_image(original_img)
            
            comparison = np.hstack((original_img, enhanced_img))
            
            save_path = os.path.join(config.ROOT_DIR, 'output', 'zerodce_test_result.jpg')
            cv2.imwrite(save_path, comparison)
            
            print(f"\nInput Shape: {original_img.shape}")
            print(f"Output Shape: {enhanced_img.shape}")
            print(f"\nSUCCESS: Comparison image saved to: {save_path}")
            print("============================================================")
            print("SUCCESS: Zero-DCE++ test completed.")
            print("============================================================")