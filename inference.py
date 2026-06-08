import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from model import LungMaxViTHybrid

# ─── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT  = r"checkpoints/best_model_finetuned.pth"
OUTPUT_CSV  = r"predictions.csv"

THRESHOLDS = {
    'ILD':          0.65,
    'Lung Cancer':  0.28,
    'Normal':       0.81,
    'Tuberculosis': 0.88,
    'COVID':        0.50,
    'Pneumonia':    0.68,
}

LABELS     = ['ILD', 'Lung Cancer', 'Normal', 'Tuberculosis', 'COVID', 'Pneumonia']
NUM_LABELS = len(LABELS)

DISPLAY_NAMES = {
    'ILD':          'ILD/Fibrosis',
    'Lung Cancer':  'Lung Cancer/Nodules',
    'Normal':       'Normal',
    'Tuberculosis': 'Tuberculosis',
    'COVID':        'COVID-19',
    'Pneumonia':    'Pneumonia',
}

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cpu')

# ─── Load model ───────────────────────────────────────────────────────────────
model = LungMaxViTHybrid(num_labels=NUM_LABELS).to(device)
checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded from {CHECKPOINT}")
print(f"Checkpoint epoch:    {checkpoint.get('epoch', 'N/A')}")
print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}\n")

# ─── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_for_inference(image_path):
    img = Image.open(image_path).convert('L')
    arr = np.array(img, dtype=np.float32)
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip(arr, p2, p98)
    if p98 - p2 > 0:
        arr = (arr - p2) / (p98 - p2) * 255.0
    return Image.fromarray(arr.astype(np.uint8)).convert('RGB')


def is_ct_scan(image_path):
    """
    CT scan validator.

    Validates on the PREPROCESSED image (after intensity normalisation)
    so that bright/dark windowed CT scans are not incorrectly rejected.

    Only rejects:
      1. Coloured images  — photos, screenshots, coloured overlays
      2. Completely blank — pure black with no structure at all
      3. No variance      — solid colour image (even after normalisation)
    """
    try:
        # ── Check 1: raw image must be near-grayscale ─────────────────────────
        # Do this on the RAW image before preprocessing — a colour photo is
        # still colour before and after normalisation.
        raw = Image.open(image_path).convert('RGB')
        arr = np.array(raw, dtype=np.float32)
        r, g, b  = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        avg_diff = (np.mean(np.abs(r - g)) +
                    np.mean(np.abs(r - b)) +
                    np.mean(np.abs(g - b))) / 3
        if avg_diff >= 15:
            return False          # colour photo / screenshot

        # ── Run preprocessing so checks 2 & 3 are on normalised image ─────────
        processed = preprocess_for_inference(image_path)
        gray      = np.array(processed.convert('L'), dtype=np.float32)

        # ── Check 2: not completely blank after normalisation ─────────────────
        non_black = (gray > 10).sum() / gray.size
        if non_black < 0.02:
            return False          # essentially empty image

        # ── Check 3: must have structural variance after normalisation ─────────
        # After preprocessing the range is stretched to 0-255, so a real CT
        # scan will always have std > 20. A solid-colour / featureless image
        # will still be flat even after stretching.
        if gray.std() < 20:
            return False

        return True

    except Exception:
        return False

# ─── Predict ──────────────────────────────────────────────────────────────────
def predict(image_path, validate=True):
    if validate and not is_valid_ct(image_path):
        print(f"⚠️  Warning: '{os.path.basename(image_path)}' may not be a CT scan.")

    image  = preprocess_for_inference(image_path)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.sigmoid(outputs).squeeze().cpu().tolist()

    if isinstance(probs, float):
        probs = [probs]

    results  = {}
    detected = []

    print(f"\n── Predictions: {os.path.basename(image_path)} ──")
    print(f"{'Label':<22} {'Probability':>12} {'Result':>12}")
    print("─" * 50)

    for label, prob in zip(LABELS, probs):
        threshold   = THRESHOLDS[label]
        is_detected = prob >= threshold
        display     = DISPLAY_NAMES[label]
        status      = "✓ DETECTED" if is_detected else "✗"
        results[label] = round(prob, 4)
        print(f"{display:<22} {prob:>11.1%} {status:>12}")
        if is_detected:
            detected.append((display, prob))

    print("─" * 50)

    if detected:
        detected.sort(key=lambda x: x[1], reverse=True)
        print(f"Detected ({len(detected)}): {', '.join([d[0] for d in detected])}")
        print(f"Top: {detected[0][0]} ({detected[0][1]:.1%})")
    else:
        top_idx = probs.index(max(probs))
        print(f"No findings. Highest: "
              f"{DISPLAY_NAMES[LABELS[top_idx]]} ({max(probs):.1%})")

    print()
    return results, probs

# ─── Save to CSV ──────────────────────────────────────────────────────────────
def save_to_csv(image_path, results):
    display_results = {DISPLAY_NAMES[k]: v for k, v in results.items()}
    row = {
        'filename':  os.path.basename(image_path),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **display_results
    }
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Result saved → {OUTPUT_CSV}")

# ─── Batch folder inference ───────────────────────────────────────────────────
def predict_folder(folder_path, validate=True):
    IMG_EXTS = {'.png', '.jpg', '.jpeg'}
    images   = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]
    if not images:
        print(f"No images found in {folder_path}")
        return
    print(f"Running inference on {len(images)} images in {folder_path}\n")
    for fname in images:
        img_path = os.path.join(folder_path, fname)
        results, _ = predict(img_path, validate=validate)
        save_to_csv(img_path, results)
    print(f"\nAll predictions saved to {OUTPUT_CSV}")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image : python inference.py <image_path>")
        print("  Folder       : python inference.py <folder_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: Path not found → {input_path}")
        sys.exit(1)

    if os.path.isdir(input_path):
        predict_folder(input_path)
    else:
        results, probs = predict(input_path)
        save_to_csv(input_path, results)