
"""
evaluate.py
Evaluates the trained Weather Classifier on the unseen Test Dataset.
Generates a classification report, confusion matrix, and a list of
actual rain images incorrectly predicted as clear.
"""

import os
import csv
import random

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import WeatherClassifier


def evaluate_model():
    print("=" * 60)
    print("Starting Model Evaluation on Test Dataset")
    print("=" * 60)

    # Load test DataLoader
    _, _, test_loader, class_names = get_dataloaders()

    print("Class index mapping:")
    for index, name in enumerate(class_names):
        print(f"  {index}: {name}")

    # Load trained classifier
    print(f"\nLoading saved weights from: {config.MODEL_SAVE_PATH}")

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
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model = model.to(config.DEVICE)
    model.eval()

    # Find the actual dataset behind the test DataLoader.
    # Standard PyTorch datasets expose samples as (path, label).
    test_dataset = test_loader.dataset

    if hasattr(test_dataset, "samples"):
        samples = test_dataset.samples
    elif hasattr(test_dataset, "imgs"):
        samples = test_dataset.imgs
    else:
        raise RuntimeError(
            "Cannot retrieve test image filenames from the dataset. "
            "Please inspect src/dataset.py."
        )

    # IMPORTANT: The filename mapping below assumes test_loader is not
    # shuffled and uses the complete dataset in its original order.
    if getattr(test_loader, "sampler", None).__class__.__name__ != "SequentialSampler":
        raise RuntimeError(
            "The test DataLoader does not use SequentialSampler. "
            "Filename-to-prediction mapping would be unsafe. "
            "Set shuffle=False for the test DataLoader or use sample indices."
        )

    y_true = []
    y_pred = []
    rain_as_clear = []

    sample_offset = 0

    with torch.no_grad():
        for images, labels in tqdm(
            test_loader,
            desc="Evaluating",
            leave=False
        ):
            images = images.to(config.DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1)

            labels_cpu = labels.cpu().tolist()
            predictions_cpu = predicted.cpu().tolist()
            probabilities_cpu = probabilities.cpu().tolist()

            batch_size = len(labels_cpu)

            for i in range(batch_size):
                dataset_index = sample_offset + i

                image_path, dataset_label = samples[dataset_index]
                true_label = labels_cpu[i]
                predicted_label = predictions_cpu[i]

                y_true.append(true_label)
                y_pred.append(predicted_label)

                true_name = class_names[true_label].lower()
                predicted_name = class_names[predicted_label].lower()

                # Save actual rain images incorrectly predicted as clear.
                if true_name == "rain" and predicted_name == "clear":
                    rain_as_clear.append({
                        "filename": os.path.basename(image_path),
                        "filepath": os.path.abspath(image_path),
                        "true_class": true_name,
                        "predicted_class": predicted_name,
                        "predicted_confidence": probabilities_cpu[i][predicted_label],
                        "rain_probability": probabilities_cpu[i][
                            class_names.index("rain")
                        ],
                    })

            sample_offset += batch_size

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

    # Confusion matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    output_dir = os.path.join(
        config.ROOT_DIR,
        "output",
        "evaluation_plots"
    )
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Weather Classification Confusion Matrix")
    plt.ylabel("Actual True Weather")
    plt.xlabel("Model Prediction")

    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to: {plot_path}")

    # Save every rain -> clear error as CSV
    csv_path = os.path.join(
        output_dir,
        "rain_predicted_as_clear.csv"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "filename",
            "filepath",
            "true_class",
            "predicted_class",
            "predicted_confidence",
            "rain_probability",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rain_as_clear)

    print(f"\nActual rain predicted as clear: {len(rain_as_clear)}")
    print(f"Error list saved to: {csv_path}")

    # Save a reproducible random sample of up to 30 errors.
    random_generator = random.Random(42)
    sample_size = min(30, len(rain_as_clear))
    sampled_errors = random_generator.sample(
        rain_as_clear,
        sample_size
    )

    sample_path = os.path.join(
        output_dir,
        "rain_predicted_as_clear_sample.txt"
    )

    with open(sample_path, "w", encoding="utf-8") as sample_file:
        sample_file.write(
            f"Random sample of {sample_size} rain-to-clear errors\n\n"
        )

        for item in sampled_errors:
            sample_file.write(
                f"{item['filepath']} | "
                f"predicted confidence: "
                f"{item['predicted_confidence']:.4f} | "
                f"rain probability: {item['rain_probability']:.4f}\n"
            )

    print(f"Random inspection sample saved to: {sample_path}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
