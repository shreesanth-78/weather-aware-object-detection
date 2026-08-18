
"""
enhance_fog.py
Implements DehazeFormer-S for fog/haze removal.

Responsibilities:
1. Load the DehazeFormer-S architecture.
2. Load the pretrained checkpoint safely.
3. Handle DataParallel checkpoints containing "module." prefixes.
4. Run dehazing on OpenCV BGR images.
5. Preserve the original image resolution.
6. Provide a standalone test when this file is executed directly.
"""

import os
import glob

import cv2
import torch
import numpy as np
import torch.nn.functional as F

import config

# DehazeFormer-S architecture
from models_arch.dehazeformer import dehazeformer_s


# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

FOG_WEIGHTS_PATH = os.path.join(
    config.MODEL_SAVE_DIR,
    "pretrained_enhancers",
    "dehazeformer-s.pth"
)


# =========================================================
# 2. CHECKPOINT LOADING
# =========================================================

def _extract_state_dict(checkpoint):
    """
    Extract the actual model state_dict from different checkpoint formats.

    Supported formats:
        1. Raw state_dict
        2. {"state_dict": state_dict}
        3. {"model": state_dict}
        4. {"params": state_dict}

    Returns:
        dict: Cleaned state dictionary.
    """

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unexpected checkpoint type: {type(checkpoint)}"
        )

    # Common checkpoint wrappers
    possible_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "params"
    ]

    state_dict = checkpoint

    for key in possible_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            break

    return state_dict


def _remove_module_prefix(state_dict):
    """
    Removes 'module.' prefix from checkpoints saved using
    torch.nn.DataParallel.

    Example:

        module.patch_embed.proj.weight
                    ↓
        patch_embed.proj.weight
    """

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        # Remove only ONE leading "module.".
        # This avoids accidentally modifying legitimate layer names.
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        cleaned_state_dict[new_key] = value

    return cleaned_state_dict


def load_dehazeformer():
    """
    Initializes DehazeFormer-S and loads pretrained weights.

    Returns:
        torch.nn.Module or None
    """

    if not os.path.isfile(FOG_WEIGHTS_PATH):
        print()
        print("=" * 60)
        print("ERROR: DehazeFormer weights not found.")
        print("=" * 60)
        print(f"Expected file:")
        print(f"  {FOG_WEIGHTS_PATH}")
        print()
        return None

    print(f"Loading DehazeFormer-S on {config.DEVICE}...")
    print(f"Weights: {FOG_WEIGHTS_PATH}")

    try:
        # -------------------------------------------------
        # Initialize architecture
        # -------------------------------------------------

        model = dehazeformer_s()

        # -------------------------------------------------
        # Load checkpoint
        # -------------------------------------------------

        checkpoint = torch.load(
            FOG_WEIGHTS_PATH,
            map_location="cpu"
        )

        print(f"Checkpoint type: {type(checkpoint).__name__}")

        # -------------------------------------------------
        # Extract state dictionary
        # -------------------------------------------------

        state_dict = _extract_state_dict(checkpoint)

        # -------------------------------------------------
        # Remove DataParallel "module." prefix
        # -------------------------------------------------

        state_dict = _remove_module_prefix(state_dict)

        # -------------------------------------------------
        # Display a few keys for debugging
        # -------------------------------------------------

        print("Checkpoint successfully read.")

        sample_keys = list(state_dict.keys())[:5]

        print("Sample checkpoint keys:")
        for key in sample_keys:
            print(f"  {key}")

        # -------------------------------------------------
        # Load weights
        # -------------------------------------------------

        load_result = model.load_state_dict(
            state_dict,
            strict=True
        )

        # Normally strict=True returns an IncompatibleKeys
        # object with empty missing/unexpected keys.
        print("Checkpoint keys successfully matched model architecture.")

        if hasattr(load_result, "missing_keys"):
            if load_result.missing_keys:
                print("WARNING - Missing keys:")
                for key in load_result.missing_keys:
                    print(f"  {key}")

        if hasattr(load_result, "unexpected_keys"):
            if load_result.unexpected_keys:
                print("WARNING - Unexpected keys:")
                for key in load_result.unexpected_keys:
                    print(f"  {key}")

        # -------------------------------------------------
        # Move model to selected device
        # -------------------------------------------------

        model = model.to(config.DEVICE)

        # Inference mode
        model.eval()

        print()
        print("=" * 60)
        print("SUCCESS: DehazeFormer-S loaded successfully.")
        print("=" * 60)
        print()

        return model

    except Exception as e:

        print()
        print("=" * 60)
        print("ERROR: Failed to load DehazeFormer weights.")
        print("=" * 60)
        print(f"{type(e).__name__}: {e}")
        print()
        print("Possible causes:")
        print("1. Checkpoint architecture does not match dehazeformer_s().")
        print("2. Checkpoint is incomplete/corrupted.")
        print("3. Wrong DehazeFormer checkpoint was downloaded.")
        print("4. Model architecture file does not match the checkpoint.")
        print("=" * 60)
        print()

        return None


# =========================================================
# 3. INITIALIZE MODEL ONCE
# =========================================================

fog_model = load_dehazeformer()


# =========================================================
# 4. IMAGE DEHAZING FUNCTION
# =========================================================

def dehaze_image(image_bgr):
    """
    Dehazes a single OpenCV BGR image using DehazeFormer-S.

    Parameters
    ----------
    image_bgr : numpy.ndarray
        OpenCV image in BGR format.

    Returns
    -------
    numpy.ndarray
        Enhanced/dehazed BGR image.

    Fail-safe behavior
    ------------------
    If the model failed to load, the original image is returned.
    """

    # -----------------------------------------------------
    # Fail-safe
    # -----------------------------------------------------

    if fog_model is None:
        return image_bgr

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if image_bgr is None:
        raise ValueError("dehaze_image received None as input.")

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError(
            "dehaze_image expects a numpy.ndarray."
        )

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected BGR image with shape (H, W, 3), "
            f"got {image_bgr.shape}"
        )

    # -----------------------------------------------------
    # Save original dimensions
    # -----------------------------------------------------

    original_height, original_width = image_bgr.shape[:2]

    # -----------------------------------------------------
    # OpenCV BGR -> RGB
    # -----------------------------------------------------

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------------------------
    # Convert uint8 [0,255] -> float32 [0,1]
    # -----------------------------------------------------

    image_rgb = image_rgb.astype(np.float32) / 255.0

    # -----------------------------------------------------
    # NumPy -> PyTorch
    #
    # HWC -> CHW -> BCHW
    # -----------------------------------------------------

    image_tensor = torch.from_numpy(
        image_rgb
    ).permute(2, 0, 1).unsqueeze(0)

    image_tensor = image_tensor.to(
        config.DEVICE,
        non_blocking=True
    )

    # -----------------------------------------------------
    # DehazeFormer requires compatible spatial dimensions.
    #
    # Pad to the nearest multiple of 16.
    # -----------------------------------------------------

    _, _, height, width = image_tensor.shape

    pad_height = (16 - height % 16) % 16
    pad_width = (16 - width % 16) % 16

    if pad_height > 0 or pad_width > 0:

        image_tensor = F.pad(
            image_tensor,
            (
                0,
                pad_width,
                0,
                pad_height
            ),
            mode="reflect"
        )

    # -----------------------------------------------------
    # Model inference
    # -----------------------------------------------------

    with torch.inference_mode():

        enhanced_tensor = fog_model(
            image_tensor
        )

    # -----------------------------------------------------
    # Some implementations return a tensor directly.
    # Keep this defensive in case the architecture returns
    # a tuple/list.
    # -----------------------------------------------------

    if isinstance(enhanced_tensor, (tuple, list)):
        enhanced_tensor = enhanced_tensor[0]

    # -----------------------------------------------------
    # Remove padding
    # -----------------------------------------------------

    enhanced_tensor = enhanced_tensor[
        :,
        :,
        :height,
        :width
    ]

    # -----------------------------------------------------
    # BCHW -> HWC
    # -----------------------------------------------------

    enhanced_numpy = (
        enhanced_tensor
        .squeeze(0)
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    # -----------------------------------------------------
    # Clamp output to valid RGB range
    # -----------------------------------------------------

    enhanced_numpy = np.clip(
        enhanced_numpy,
        0.0,
        1.0
    )

    # -----------------------------------------------------
    # [0,1] -> [0,255]
    # -----------------------------------------------------

    enhanced_numpy = (
        enhanced_numpy * 255.0
    ).round().astype(np.uint8)

    # -----------------------------------------------------
    # RGB -> BGR
    # -----------------------------------------------------

    enhanced_bgr = cv2.cvtColor(
        enhanced_numpy,
        cv2.COLOR_RGB2BGR
    )

    return enhanced_bgr


# =========================================================
# 5. STANDALONE TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing DehazeFormer-S Standalone")
    print("=" * 60)

    # -----------------------------------------------------
    # Check whether model loaded
    # -----------------------------------------------------

    if fog_model is None:

        print()
        print("Test aborted.")
        print("DehazeFormer-S could not be loaded.")
        print()

        raise SystemExit(1)

    # -----------------------------------------------------
    # Locate test fog images
    # -----------------------------------------------------

    fog_dir = os.path.join(
        config.DATA_DIR,
        "test",
        "fog"
    )

    print(f"Searching for fog images in:")
    print(f"  {fog_dir}")
    print()

    # Support common image formats
    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.webp"
    ]

    fog_images = []

    for extension in image_extensions:

        fog_images.extend(
            glob.glob(
                os.path.join(
                    fog_dir,
                    extension
                )
            )
        )

    # -----------------------------------------------------
    # No images found
    # -----------------------------------------------------

    if not fog_images:

        print("No test fog images were found.")
        print()
        print(f"Expected directory:")
        print(f"  {fog_dir}")
        print()
        print("Supported formats:")
        print("  JPG / JPEG / PNG / BMP / WEBP")
        print()

        raise SystemExit(0)

    # -----------------------------------------------------
    # Select first test image
    # -----------------------------------------------------

    test_img_path = fog_images[0]

    print(f"Testing image:")
    print(f"  {test_img_path}")
    print()

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    original_img = cv2.imread(
        test_img_path,
        cv2.IMREAD_COLOR
    )

    if original_img is None:

        print("ERROR: OpenCV could not read the image.")
        print(f"Image path: {test_img_path}")

        raise SystemExit(1)

    print(
        f"Input Shape: {original_img.shape}"
    )

    # -----------------------------------------------------
    # Run DehazeFormer
    # -----------------------------------------------------

    try:

        enhanced_img = dehaze_image(
            original_img
        )

    except Exception as e:

        print()
        print("=" * 60)
        print("ERROR during DehazeFormer inference")
        print("=" * 60)
        print(f"{type(e).__name__}: {e}")
        print()

        raise SystemExit(1)

    # -----------------------------------------------------
    # Verify output
    # -----------------------------------------------------

    print(
        f"Output Shape: {enhanced_img.shape}"
    )

    if enhanced_img.shape != original_img.shape:

        print()
        print("WARNING:")
        print("Output dimensions do not match input dimensions.")

    # -----------------------------------------------------
    # Create side-by-side comparison
    # -----------------------------------------------------

    comparison = np.hstack(
        (
            original_img,
            enhanced_img
        )
    )

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    output_dir = os.path.join(
        config.ROOT_DIR,
        "output"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------

    save_path = os.path.join(
        output_dir,
        "dehazeformer_test_result.jpg"
    )

    success = cv2.imwrite(
        save_path,
        comparison
    )

    if not success:

        print()
        print("ERROR: Failed to save comparison image.")
        print(f"Path: {save_path}")
        raise SystemExit(1)

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("SUCCESS: DehazeFormer-S test completed.")
    print("=" * 60)
    print()
    print(f"Comparison saved to:")
    print(f"  {save_path}")
    print()
    print("LEFT  = Original fog image")
    print("RIGHT = Dehazed image")
    print()

