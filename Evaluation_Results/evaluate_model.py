"""
PlantAI Model Evaluation Script
================================
Run this script to compute real accuracy, precision, recall and F1 score
for your MobileNetV2 plant disease detection model.

Requirements:
- A folder of test images organised by class (see instructions below)
- Your venv activated

Usage:
    python evaluate_model.py --test_dir path/to/test_images
"""

import os
import json
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForImageClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Label normalisation map ───────────────────────────────────────────────────
# Maps YOUR test-folder names  →  the model's exact id2label strings.
# The model uses "Pepper,_bell___..." and lowercase "healthy".
# Add/edit entries here if you add more classes.
LABEL_MAP = {
    # Pepper: folder uses "___Bell" but model uses ",_bell"
    "Pepper___Bell_Bacterial_spot" : "Pepper,_bell___Bacterial_spot",
    "Pepper___Bell_healthy"        : "Pepper,_bell___healthy",
    # Potato: folder uses capital "Healthy" but model uses lowercase
    "Potato___Healthy"             : "Potato___healthy",
    # Tomato: same capitalisation issue
    "Tomato___Healthy"             : "Tomato___healthy",
}

# ── Load Model ───────────────────────────────────────────────────────────────
def load_model():
    print("🌿 Loading model...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("✅ Model loaded!\n")
    return model

# ── Predict single image ─────────────────────────────────────────────────────
def predict(model, image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            logits = outputs.logits
        pred_idx = logits.argmax(dim=-1).item()
        pred_label = model.config.id2label[pred_idx]
        return pred_label
    except Exception as e:
        print(f"  ⚠️  Skipping {image_path}: {e}")
        return None

# ── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate(test_dir, model):
    """
    test_dir should have this structure:
    test_images/
        Tomato___Early_blight/
            img1.jpg
            img2.jpg
        Potato___Late_blight/
            img1.jpg
        ...
    """

    if not os.path.exists(test_dir):
        print(f"❌ Directory not found: {test_dir}")
        print("\nPlease create a test_images/ folder with subfolders per disease class.")
        print("Example structure:")
        print("  test_images/")
        print("    Tomato___Early_blight/  ← folder name must match model labels")
        print("      img1.jpg")
        print("      img2.jpg")
        return

    classes = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    if not classes:
        print("❌ No class subfolders found in test directory!")
        return

    print(f"📁 Found {len(classes)} classes in test directory")
    print(f"📋 Classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}\n")

    y_true = []
    y_pred = []
    skipped = 0
    total = 0

    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]

        print(f"  🔍 Evaluating: {class_name} ({len(images)} images)")

        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            pred = predict(model, img_path)
            total += 1

            if pred is None:
                skipped += 1
                continue

            # Normalise folder name → model label (fixes mismatched class names)
            normalised_true = LABEL_MAP.get(class_name, class_name)
            y_true.append(normalised_true)
            y_pred.append(pred)

    print(f"\n✅ Evaluated {len(y_true)} images ({skipped} skipped)\n")

    if not y_true:
        print("❌ No predictions made. Check your image files.")
        return

    # ── Compute metrics ──────────────────────────────────────────────────────
    # Normalize labels (model might return slightly different format)
    all_labels = sorted(set(y_true + y_pred))

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted',
                                 labels=all_labels, zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted',
                              labels=all_labels, zero_division=0)
    f1        = f1_score(y_true, y_pred, average='weighted',
                          labels=all_labels, zero_division=0)

    print("=" * 55)
    print("         📊 PLANTAI MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  ✅ Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"  🎯 Precision        : {precision * 100:.2f}%")
    print(f"  🔍 Recall           : {recall * 100:.2f}%")
    print(f"  ⚖️  F1 Score         : {f1 * 100:.2f}%")
    print(f"  📸 Images Tested    : {len(y_true)}")
    print(f"  📁 Classes Tested   : {len(classes)}")
    print("=" * 55)

    # Per-class breakdown
    print("\n📋 Per-Class Results:\n")
    # Reverse map so the JSON/display uses your original folder names
    reverse_map = {v: k for k, v in LABEL_MAP.items()}
    report = classification_report(
        y_true, y_pred,
        labels=sorted(set(y_true)),
        zero_division=0,
        output_dict=True
    )

    per_class = {}
    for label, metrics in report.items():
        if label in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        # Display with your original folder name where applicable
        display_name = reverse_map.get(label, label)
        per_class[display_name] = {
            'precision': round(metrics['precision'] * 100, 1),
            'recall':    round(metrics['recall'] * 100, 1),
            'f1':        round(metrics['f1-score'] * 100, 1),
            'support':   int(metrics['support'])
        }
        print(f"  {display_name}")
        print(f"    Precision: {per_class[display_name]['precision']}%  "
              f"Recall: {per_class[display_name]['recall']}%  "
              f"F1: {per_class[display_name]['f1']}%  "
              f"({per_class[display_name]['support']} images)")

    # ── Save results to JSON ─────────────────────────────────────────────────
    results = {
        "model": MODEL_NAME,
        "images_tested": len(y_true),
        "classes_tested": len(classes),
        "overall": {
            "accuracy":  round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall":    round(recall * 100, 2),
            "f1_score":  round(f1 * 100, 2),
        },
        "per_class": per_class
    }

    output_path = "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
    print("    → Use this file to populate your credibility page!\n")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PlantAI model")
    parser.add_argument(
        "--test_dir",
        default="test_images",
        help="Path to test images directory (default: test_images/)"
    )
    args = parser.parse_args()

    # Install sklearn if not present
    try:
        import sklearn
    except ImportError:
        print("📦 Installing scikit-learn...")
        os.system("pip install scikit-learn --break-system-packages -q")
        import sklearn

    model = load_model()
    evaluate(args.test_dir, model)
