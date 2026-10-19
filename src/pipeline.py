
"""
pipeline.py
Adaptive Weather-Aware Object Detection Pipeline.

Workflow:
1. Classify weather using ResNet-18.
2. Apply condition-specific enhancement.
3. Detect objects using YOLO11n.
4. Save side-by-side comparison images.

Each run randomly selects one test image from each weather class.
"""

import os
import glob
import random
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

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
    print(
        f"Loading Custom Weather Classifier "
        f"(ResNet-18) on {config.DEVICE}..."
    )

    model = WeatherClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=False
    )

    try:
        model.load_state_dict(
            torch.load(
                config.MODEL_SAVE_PATH,
                map_location=config.DEVICE
            )
        )
        model.to(config.DEVICE)
        model.eval()
        print("SUCCESS: Weather classifier loaded.")
        return model

    except Exception as e:
        print(f"ERROR: Failed to load classifier. Details: {e}")
        return None


classifier_model = load_classifier()

classifier_transforms = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_weather(image_bgr):
    """Predict the weather condition from an OpenCV BGR image."""

    if classifier_model is None:
        raise RuntimeError(
            "Weather classifier failed to load. "
            "Pipeline stopped to avoid incorrect weather routing."
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    input_tensor = classifier_transforms(
        pil_image
    ).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        class_idx = outputs.argmax(dim=1).item()

    return config.CLASSES[class_idx]


# ---------------------------------------------------------
# 2. Process One Image
# ---------------------------------------------------------

def process_frame(image_path):
    """Classify, enhance, detect, and save a comparison image."""

    print("\n" + "-" * 60)
    print(f"Processing: {os.path.basename(image_path)}")

    original_img = cv2.imread(image_path)

    if original_img is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    # STEP A: Weather classification
    predicted_weather = predict_weather(original_img)
    print(f"Classification : [{predicted_weather.upper()}]")

    # STEP B: Weather-specific enhancement
    enhanced_img = original_img.copy()

    if predicted_weather == "fog":
        print("Routing        : DehazeFormer-S")
        enhanced_img = dehaze_image(original_img)

    elif predicted_weather == "rain":
        print("Routing        : Restormer")
        enhanced_img = derain_image(original_img)

    elif predicted_weather == "night":
        print("Routing        : Zero-DCE++")
        enhanced_img = brighten_image(original_img)

    elif predicted_weather == "clear":
        print("Routing        : Bypassing enhancement")

    else:
        raise ValueError(
            f"Unexpected predicted weather class: {predicted_weather}"
        )

    if enhanced_img is None:
        raise RuntimeError(
            f"Enhancement returned no image for {image_path}"
        )

    if enhanced_img.shape[:2] != original_img.shape[:2]:
        raise ValueError(
            f"Image dimensions changed during enhancement: "
            f"{original_img.shape[:2]} -> {enhanced_img.shape[:2]}"
        )

    # STEP C: Object detection
    print("Detecting      : YOLO11n")
    annotated_img, detection_results = detect_objects(enhanced_img)

    if annotated_img is None:
        raise RuntimeError(
            f"Object detector returned no image for {image_path}"
        )

    if detection_results is not None and hasattr(
        detection_results, "boxes"
    ):
        print(
            f"Detections     : {len(detection_results.boxes)} objects"
        )
    else:
        print(
            "WARNING: Detection results are unavailable; "
            "verify that YOLO11n loaded correctly."
        )

    # STEP D: Create side-by-side comparison
    original_display = original_img.copy()
    enhanced_display = annotated_img.copy()

    cv2.putText(
        original_display,
        f"Raw Input ({predicted_weather.upper()})",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    cv2.putText(
        enhanced_display,
        "Enhanced + YOLO11",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    comparison = np.hstack([
        original_display,
        enhanced_display
    ])

    output_dir = os.path.join(config.ROOT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Include the original filename to distinguish random samples.
    base_name = os.path.basename(image_path)
    out_filename = (
        f"pipeline_out_{predicted_weather}_{base_name}"
    )
    out_path = os.path.join(output_dir, out_filename)

    if not cv2.imwrite(out_path, comparison):
        raise RuntimeError(f"Failed to save output image: {out_path}")

    print(f"Result Saved   : {out_path}")


# ---------------------------------------------------------
# 3. Random Test Image Selection
# ---------------------------------------------------------

def get_random_test_images():
    """
    Select one random image from each weather-class test folder.

    Supported extensions: JPG, JPEG, PNG, BMP, TIFF, WEBP.
    Uses a fresh random selection each time the script is run.
    """

    image_extensions = (
        "*.jpg", "*.jpeg", "*.png", "*.bmp",
        "*.tif", "*.tiff", "*.webp"
    )

    test_images = []

    print("\nSelecting random images from the test dataset...")

    for condition in config.CLASSES:
        class_dir = os.path.join(
            config.DATA_DIR, "test", condition
        )

        class_images = []

        for extension in image_extensions:
            class_images.extend(
                glob.glob(os.path.join(class_dir, extension))
            )

        # Remove duplicates and ensure stable input to random.choice.
        class_images = sorted(set(class_images))

        if not class_images:
            print(
                f"WARNING: No supported images found for "
                f"class '{condition}' in {class_dir}"
            )
            continue

        selected_image = random.choice(class_images)
        test_images.append(selected_image)

        print(
            f"  {condition.upper():<6} -> "
            f"{os.path.basename(selected_image)} "
            f"({len(class_images)} available)"
        )

    return test_images


# ---------------------------------------------------------
# 4. Main Pipeline
# ---------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE WEATHER-AWARE OBJECT DETECTION PIPELINE")
    print("=" * 60)

    test_images = get_random_test_images()

    if not test_images:
        print("ERROR: No test images were found.")
    else:
        print(
            f"\nProcessing {len(test_images)} randomly selected "
            f"test images..."
        )

        for image_path in test_images:
            try:
                process_frame(image_path)
            except Exception as e:
                print(
                    f"ERROR processing "
                    f"{os.path.basename(image_path)}: {e}"
                )

    print("\n" + "=" * 60)
    print("PIPELINE RUN FINISHED")
    print("=" * 60)

