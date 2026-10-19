"""
visualize_cam.py
Generates Grad-CAM visual heatmaps for ResNet-18 weather classification.
Proves the model learns meteorological features rather than spurious background cues.
"""

import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from PIL import Image

import config
from model import WeatherClassifier

def load_eval_classifier():
    """Loads the trained ResNet-18 weather classifier in evaluation mode."""
    model = WeatherClassifier(num_classes=config.NUM_CLASSES, pretrained=False)
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.to(config.DEVICE)
    model.eval()
    return model

def run_gradcam_analysis(sample_per_class=1):
    model = load_eval_classifier()

    # The target layer for ResNet-18 is the final convolutional block of layer4
    target_layers = [model.model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # Standard preprocessing
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    output_dir = os.path.join(config.ROOT_DIR, 'output', 'gradcam_explanations')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("RUNNING GRAD-CAM EXPLAINABILITY ANALYSIS")
    print(f"Target Layer: {target_layers[0]}")
    print("=" * 60)

    summary_figures = []

    for class_idx, class_name in enumerate(config.CLASSES):
        folder_path = os.path.join(config.DATA_DIR, 'test', class_name)
        image_files = sorted(glob.glob(os.path.join(folder_path, '*.[jp][pn]*')))

        if not image_files:
            print(f"[!] Warning: No images found in {folder_path}")
            continue

        selected_files = image_files[:sample_per_class]

        for img_path in selected_files:
            bgr_img = cv2.imread(img_path)
            if bgr_img is None:
                continue

            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb_img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            rgb_float = np.float32(rgb_resized) / 255.0

            pil_img = Image.fromarray(rgb_img)
            input_tensor = transform(pil_img).unsqueeze(0).to(config.DEVICE)

            # Predict class
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item() * 100.0

            # Target the predicted class for CAM generation
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

            # Overlay heatmap on normalized float image
            visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

            pred_name = config.CLASSES[pred_idx]
            filename = os.path.basename(img_path)

            print(f"File: {filename:<18} | GT: {class_name:<6} | Pred: {pred_name:<6} ({confidence:.2f}%)")

            # Save individual result
            save_name = f"cam_{class_name}_{pred_name}_{filename}.jpg"
            save_path = os.path.join(output_dir, save_name)
            
            # Combine original resized with heatmap
            combined = np.hstack((cv2.cvtColor((rgb_float * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                                  cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)))
            cv2.imwrite(save_path, combined)

            summary_figures.append({
                'original': rgb_resized,
                'cam': visualization,
                'gt': class_name,
                'pred': pred_name,
                'conf': confidence
            })

    # Generate a consolidated 4-panel publication-ready figure
    if summary_figures:
        fig, axes = plt.subplots(len(summary_figures), 2, figsize=(8, 3 * len(summary_figures)))
        if len(summary_figures) == 1:
            axes = np.expand_dims(axes, 0)

        for i, item in enumerate(summary_figures):
            axes[i, 0].imshow(item['original'])
            axes[i, 0].set_title(f"Input: {item['gt'].capitalize()}", fontsize=11)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(item['cam'])
            is_match = item['gt'].lower() == item['pred'].lower()
            color = 'green' if is_match else 'red'
            axes[i, 1].set_title(f"Grad-CAM: {item['pred'].capitalize()} ({item['conf']:.1f}%)", 
                                 fontsize=11, fontweight='bold', color=color)
            axes[i, 1].axis('off')

        plt.tight_layout()
        summary_path = os.path.join(config.ROOT_DIR, 'output', 'gradcam_summary_grid.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("=" * 60)
        print(f"Analysis Complete. Consolidated grid saved to: {summary_path}")
        print("=" * 60)

if __name__ == '__main__':
    run_gradcam_analysis(sample_per_class=1)