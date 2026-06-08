import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from PIL import Image
from torchvision import transforms
from model import LungMaxViTHybrid
import sys
import os

# ─── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT = r"best_model_finetuned.pth"
LABELS     = ['ILD', 'Lung Cancer', 'Normal', 'Tuberculosis', 'COVID', 'Pneumonia']
NUM_LABELS = len(LABELS)

DISPLAY_NAMES = {
    'ILD':          'ILD/Fibrosis',
    'Lung Cancer':  'Lung Cancer/Nodules',
    'Normal':       'Normal',
    'Tuberculosis': 'Tuberculosis',
    'COVID':        'COVID-19/Infection',
    'Pneumonia':    'Pneumonia',
}

THRESHOLDS = {
    'ILD':          0.65,
    'Lung Cancer':  0.28,
    'Normal':       0.81,
    'Tuberculosis': 0.88,
    'COVID':        0.50,
    'Pneumonia':    0.68,
}

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cpu')

# ─── Load model ───────────────────────────────────────────────────────────────
model = LungMaxViTHybrid(num_labels=NUM_LABELS).to(device)
checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully")

# ─── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def generate_shap(image_path):
    image   = Image.open(image_path).convert('RGB').resize((224, 224))
    rgb_img = np.array(image) / 255.0
    tensor  = transform(image).unsqueeze(0).to(device)

    print(f"\nGenerating SHAP explanations for: {os.path.basename(image_path)}")
    print("This may take a few minutes...")

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.sigmoid(outputs).squeeze().tolist()

    print(f"\n── Predictions ──")
    for label, prob in zip(LABELS, probs):
        display  = DISPLAY_NAMES[label]
        status   = "✓ DETECTED" if prob >= THRESHOLDS[label] else "✗"
        print(f"  {display:<25} {prob:.1%} {status}")

    background  = torch.zeros_like(tensor)
    explainer   = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(tensor)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 10))
    fig.patch.set_facecolor('#0D1117')

    for i, label in enumerate(LABELS):
        display  = DISPLAY_NAMES[label]
        prob     = probs[i]
        detected = prob >= THRESHOLDS[label]

        # Original image row
        ax_orig = fig.add_subplot(2, NUM_LABELS, i + 1)
        ax_orig.imshow(rgb_img, cmap='gray')
        ax_orig.set_title(
            f"{display}\n{prob:.1%} {'✓' if detected else '✗'}",
            fontsize=9, fontweight='bold',
            color='#4CAF50' if detected else '#888888',
            pad=6
        )
        ax_orig.axis('off')
        if detected:
            for spine in ax_orig.spines.values():
                spine.set_edgecolor('#4CAF50')
                spine.set_linewidth(2)

        # SHAP heatmap row
        ax_shap = fig.add_subplot(2, NUM_LABELS, NUM_LABELS + i + 1)
        sv           = shap_values[i][0]
        sv           = np.transpose(sv, (1, 2, 0))
        sv_magnitude = np.abs(sv).mean(axis=2)

        im = ax_shap.imshow(sv_magnitude, cmap='inferno')
        ax_shap.set_title(f'SHAP: {display}', fontsize=8, color='#CCCCCC', pad=4)
        ax_shap.axis('off')
        plt.colorbar(im, ax=ax_shap, fraction=0.046, pad=0.04)

    plt.suptitle(
        f'SHAP Analysis — {os.path.basename(image_path)}',
        fontsize=14, fontweight='bold', color='white', y=1.01
    )
    plt.tight_layout()

    output_path = f"shap_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0D1117', edgecolor='none')
    plt.show()
    print(f"\nSHAP saved → {output_path}")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python shap_viz.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    generate_shap(image_path)