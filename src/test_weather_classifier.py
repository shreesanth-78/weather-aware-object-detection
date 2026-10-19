"""
Weather-Aware Object Detection System
Review 1 Demonstration: ResNet-18 Weather Classification Inference
"""

import os
import sys
import glob
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Ensure root directory is in Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import existing project configuration and model
try:
    from src import config
    from src.model import WeatherClassifier
except ImportError:
    # Fallback if running directly without package resolution
    import config
    from model import WeatherClassifier


def get_inference_transforms():
    """Exact transforms matching validation/testing during model training."""
    img_size = getattr(config, "IMAGE_SIZE", (224, 224))
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    mean = getattr(config, "NORM_MEAN", [0.485, 0.456, 0.406])
    std = getattr(config, "NORM_STD", [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_classifier(device):
    """Loads the trained ResNet-18 weights into the architecture."""
    # Resolve checkpoint path from config or default models path
    checkpoint_path = getattr(
        config,
        "MODEL_PATH",
        os.path.join(ROOT_DIR, "models", "weather_classifier_resnet18.pth")
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"\n[ERROR] Trained checkpoint not found at: {checkpoint_path}\n"
            "Please verify that 'weather_classifier_resnet18.pth' exists in the 'models/' directory."
        )

    # Initialize blank architecture (num_classes=4)
    num_classes = getattr(config, "NUM_CLASSES", 4)
    model = WeatherClassifier(num_classes=num_classes, pretrained=False)

    # Load trained state dictionary
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint_path


def get_test_samples(test_dir, num_samples_per_class=1):
    """Finds test images from each class subfolder."""
    if not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"\n[ERROR] Test dataset directory not found at: {test_dir}\n"
            "Please check the test dataset path in src/config.py."
        )

    class_names = getattr(config, "CLASS_NAMES", ["clear", "fog", "night", "rain"])
    sample_images = []

    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    for cls in class_names:
        cls_folder = os.path.join(test_dir, cls)
        found_files = []
        for ext in valid_extensions:
            found_files.extend(glob.glob(os.path.join(cls_folder, ext)))
            found_files.extend(glob.glob(os.path.join(cls_folder, ext.upper())))

        if not found_files:
            print(f"[WARNING] No images found for class '{cls}' in {cls_folder}")
            continue

        found_files.sort()
        # Pick samples
        selected = found_files[:num_samples_per_class]
        for img_path in selected:
            sample_images.append({
                "path": img_path,
                "ground_truth": cls.capitalize(),
                "filename": os.path.basename(img_path)
            })

    if not sample_images:
        raise RuntimeError(f"No test images found inside {test_dir}.")

    return sample_images, class_names


def run_demo():
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print(" WEATHER CLASSIFICATION DEMONSTRATION (REVIEW 1) ")
    print("=" * 60)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Load Model
    model, checkpoint_path = load_classifier(device)
    print(f"Loaded Checkpoint: {os.path.relpath(checkpoint_path, ROOT_DIR)}")

    # 3. Locate Test Data
    test_dir = getattr(
        config,
        "TEST_DIR",
        os.path.join(ROOT_DIR, "datasets", "weather_classification", "test")
    )
    samples, class_names = get_test_samples(test_dir, num_samples_per_class=1)
    transform = get_inference_transforms()

    results = []
    print("\nRunning Inference on Test Scenarios...\n" + "-" * 60)

    # 4. Perform Inference
    with torch.no_grad():
        for item in samples:
            img_path = item["path"]
            pil_img = Image.open(img_path).convert("RGB")
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1)[0]

            conf, pred_idx = torch.max(probabilities, dim=0)
            pred_class = class_names[pred_idx.item()].capitalize()
            confidence_pct = conf.item() * 100.0

            item["pred_class"] = pred_class
            item["confidence"] = confidence_pct
            item["pil_image"] = pil_img
            results.append(item)

            print(f"Image             : {item['filename']}")
            print(f"Ground Truth      : {item['ground_truth']}")
            print(f"Predicted Weather : {item['pred_class']}")
            print(f"Confidence        : {confidence_pct:.2f}%")
            print("-" * 60)

    print("============================================================")
    print("Inference completed successfully.")
    print("============================================================\n")

    # 5. Visual Grid Display
    num_results = len(results)
    fig, axes = plt.subplots(1, num_results, figsize=(4 * num_results, 4.5))
    if num_results == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        ax.imshow(res["pil_image"])
        is_correct = res["ground_truth"].lower() == res["pred_class"].lower()
        title_color = "green" if is_correct else "red"

        title_text = (
            f"Ground Truth: {res['ground_truth']}\n"
            f"Prediction: {res['pred_class']} ({res['confidence']:.1f}%)"
        )
        ax.set_title(title_text, fontsize=11, fontweight="bold", color=title_color)
        ax.axis("off")

    plt.tight_layout()

    # Save output plot
    output_dir = getattr(config, "OUTPUT_DIR", os.path.join(ROOT_DIR, "output"))
    os.makedirs(output_dir, exist_ok=True)
    demo_save_path = os.path.join(output_dir, "review1_classifier_demo.png")
    plt.savefig(demo_save_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Visual demonstration plot saved to: {demo_save_path}")

    # Display interactive window
    plt.show()


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)