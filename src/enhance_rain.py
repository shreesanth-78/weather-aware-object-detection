"""
enhance_rain.py
Implements Restormer for state-of-the-art rain streak removal.
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

import config
from models_arch.restormer import Restormer

# ---------------------------------------------------------
# 1. Model Initialization & Strict Weight Loading
# ---------------------------------------------------------
RAIN_WEIGHTS_PATH = os.path.join(config.MODEL_SAVE_DIR, 'pretrained_enhancers', 'restormer_deraining.pth')

def load_restormer():
    """Initializes Restormer and performs strict weight verification."""
    if not os.path.exists(RAIN_WEIGHTS_PATH):
        print(f"ERROR: Weights not found at {RAIN_WEIGHTS_PATH}")
        return None

    print(f"Loading Restormer (Deraining) on {config.DEVICE}...")
    
    # Initialize the architecture
    model = Restormer(
        inp_channels=3, 
        out_channels=3, 
        dim=48, 
        num_blocks=[4,6,6,8], 
        num_refinement_blocks=4, 
        heads=[1,2,4,8], 
        ffn_expansion_factor=2.66, 
        bias=False, 
        LayerNorm_type='WithBias', 
        dual_pixel_task=False
    )
    
    try:
        # 1. Load Checkpoint safely
        checkpoint = torch.load(RAIN_WEIGHTS_PATH, map_location=config.DEVICE)
        
        # 2. Extract actual state dictionary
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 3. Clean "module." prefix if it exists
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            clean_state_dict[new_key] = v
            
        # 4. Strict Key Verification
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
            
        # 5. Load weights strictly
        model.load_state_dict(clean_state_dict, strict=True)
        model = model.to(config.DEVICE)
        model.eval()
        print("============================================================")
        print("SUCCESS: Restormer loaded successfully.")
        print("============================================================")
        return model
        
    except Exception as e:
        print(f"Failed to load Restormer weights: {e}")
        return None

# Initialize the model once in memory
rain_model = load_restormer()

# ---------------------------------------------------------
# 2. Main Wrapper Function for Pipeline
# ---------------------------------------------------------
def derain_image(image_bgr):
    """
    Takes a raw OpenCV BGR image, runs it through Restormer, 
    and returns the enhanced BGR image.
    """
    if rain_model is None:
        return image_bgr # Fail-safe
        
    with torch.no_grad():
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_normalized = np.float32(img_rgb) / 255.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(config.DEVICE)
        
        # Restormer downsamples hierarchically by 8x. Pad dynamically to avoid tensor shape mismatch.
        _, _, h, w = img_tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')

        # Forward pass
        enhanced_tensor = rain_model(img_tensor)

        # Crop padding back to exact original resolution
        enhanced_tensor = enhanced_tensor[:, :, :h, :w]

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
    print("Testing Restormer Standalone")
    print("============================================================")
    
    if rain_model is None:
        print("Test aborted. Model could not be loaded.")
    else:
        import glob
        
        rain_dir = os.path.join(config.DATA_DIR, 'test', 'rain')
        search_pattern = os.path.join(rain_dir, '*.jpg') 
        
        rain_images = glob.glob(search_pattern)
        
        if not rain_images:
            print(f"Could not find any test images in {rain_dir}.")
        else:
            test_img_path = rain_images[0]
            print(f"Testing on image: {test_img_path}")
            
            original_img = cv2.imread(test_img_path)
            
            print("Running Restormer inference (this may take a moment)...")
            enhanced_img = derain_image(original_img)
            
            comparison = np.hstack((original_img, enhanced_img))
            
            save_path = os.path.join(config.ROOT_DIR, 'output', 'restormer_test_result.jpg')
            cv2.imwrite(save_path, comparison)
            
            print(f"\nInput Shape: {original_img.shape}")
            print(f"Output Shape: {enhanced_img.shape}")
            print(f"\nSUCCESS: Comparison image saved to: {save_path}")
            print("============================================================")
            print("SUCCESS: Restormer test completed.")
            print("============================================================")