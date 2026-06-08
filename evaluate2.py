import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import os
from model import LungMaxViTHybrid

# ─── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT = r"checkpoints/best_model_finetuned.pth"
TEST_CSV   = r"D:\projects\lungmaxvit\dataset\final_test.csv"

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

THRESHOLDS = {
    'ILD':          0.65,
    'Lung Cancer':  0.28,
    'Normal':       0.81,
    'Tuberculosis': 0.88,
    'COVID':        0.50,
    'Pneumonia':    0.68,
}

# ─── Dataset base & dirs ──────────────────────────────────────────────────────
DATASET_BASE = r'D:\projects\lungmaxvit\dataset'

ALL_IMG_DIRS = [
    r"D:\projects\lungmaxvit\dataset\ver1\train",
    r"D:\projects\lungmaxvit\dataset\ver1\valid",
    r"D:\projects\lungmaxvit\dataset\ver1\test",
    r"D:\projects\lungmaxvit\dataset\lungcancerdataset\train",
    r"D:\projects\lungmaxvit\dataset\lungcancerdataset\valid",  # ← add
    r"D:\projects\lungmaxvit\dataset\lungcancerdataset\test",   # ← add
    r"D:\projects\lungmaxvit\dataset\covidDATAset\COVID",
    r"D:\projects\lungmaxvit\dataset\covidDATAset\non-COVID",
    r"D:\projects\lungmaxvit\dataset\Pneumonia Balanced CT\Pneumonia",
    r"D:\projects\lungmaxvit\dataset\Pneumonia Balanced CT\Normal",
    r"D:\projects\lungmaxvit\dataset\FibrosisDataset",
    r"D:\projects\lungmaxvit\dataset\TBdataset",
    r"D:\projects\lungmaxvit\dataset\2COVID",
    r"D:\projects\lungmaxvit\dataset\PNEUMONIA-CT\train\Pneumonia",
    r"D:\projects\lungmaxvit\dataset\PNEUMONIA-CT\valid\Pneumonia",
    r"D:\projects\lungmaxvit\dataset\PNEUMONIA-CT\test\Pneumonia",
]

# ─── Pre-build file index ─────────────────────────────────────────────────────
print("Building file index...")
_FILE_INDEX = {}
for _d in ALL_IMG_DIRS:
    for _root, _, _files in os.walk(_d):
        for _f in _files:
            if _f.lower().endswith(('.png', '.jpg', '.jpeg')):
                _full = os.path.join(_root, _f)
                _rel  = os.path.relpath(_full, DATASET_BASE)
                _FILE_INDEX[_rel] = _full
                if _f not in _FILE_INDEX:
                    _FILE_INDEX[_f] = _full
print(f"File index built: {len(_FILE_INDEX)} entries")

# ─── Dataset ──────────────────────────────────────────────────────────────────
class MultiDirDataset(Dataset):
    def __init__(self, csv_path, img_dirs, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dirs  = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        filename = row['filename']

        filename_norm = filename.replace('/', os.sep).replace('\\', os.sep)
        img_path = (
            _FILE_INDEX.get(filename_norm) or
            _FILE_INDEX.get(os.path.basename(filename_norm))
        )

        if img_path is None:
            raise FileNotFoundError(f"Image not found: {filename}")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[LABELS].values.astype(float), dtype=torch.float32)
        return {'image': image, 'label': label}

# ─── Device & Model ───────────────────────────────────────────────────────────
device = torch.device('cpu')

model = LungMaxViTHybrid(num_labels=NUM_LABELS).to(device)
checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded — {CHECKPOINT}")
print(f"Epoch: {checkpoint.get('epoch', 'N/A')} | Val loss: {checkpoint['val_loss']:.4f}\n")

# ─── Transform & DataLoader ───────────────────────────────────────────────────
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = MultiDirDataset(csv_path=TEST_CSV, img_dirs=ALL_IMG_DIRS, transform=test_transforms)
loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
print(f"Evaluating on {len(dataset)} images\n")

# ─── Run Inference ────────────────────────────────────────────────────────────
all_probs  = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating"):
        images  = batch['image'].to(device)
        labels  = batch['label'].cpu().numpy()
        outputs = model(images)
        probs   = torch.sigmoid(outputs).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

all_probs  = np.vstack(all_probs)
all_labels = np.vstack(all_labels)
all_preds  = np.zeros_like(all_probs, dtype=int)

for i, label in enumerate(LABELS):
    all_preds[:, i] = (all_probs[:, i] >= THRESHOLDS[label]).astype(int)

# ─── Debug ────────────────────────────────────────────────────────────────────
print("\nLabel variance check:")
for i, label in enumerate(LABELS):
    unique     = np.unique(all_labels[:, i])
    prob_range = all_probs[:, i].max() - all_probs[:, i].min()
    print(f"  {DISPLAY_NAMES[label]:<25} unique: {unique}  prob range: {prob_range:.4f}")

print("\nRaw probabilities for first 5 images:")
header = f"{'':15}" + "".join(f"{DISPLAY_NAMES[l]:>10}" for l in LABELS)
print(header)
for i in range(min(5, len(all_probs))):
    prob_row  = "".join(f"{all_probs[i][j]:>10.4f}" for j in range(NUM_LABELS))
    label_row = "".join(f"{int(all_labels[i][j]):>10}" for j in range(NUM_LABELS))
    print(f"Image {i+1:<9}{prob_row}")
    print(f"  Actual:    {label_row}")

print("\nPrediction counts:")
for i, label in enumerate(LABELS):
    predicted = all_preds[:, i].sum()
    actual    = int(all_labels[:, i].sum())
    print(f"  {DISPLAY_NAMES[label]:<25} Predicted: {predicted:>4}  Actual: {actual:>4}")

# ─── Metrics ──────────────────────────────────────────────────────────────────
print("\n" + "═" * 62)
print("  EVALUATION RESULTS")
print("═" * 62)
print(f"\n{'Label':<25} {'Accuracy':>10} {'F1 Score':>10} {'AUC':>10}")
print("─" * 58)

results = []
for i, label in enumerate(LABELS):
    acc  = accuracy_score(all_labels[:, i], all_preds[:, i])
    f1   = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
    try:
        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
    except ValueError:
        auc = float('nan')

    display = DISPLAY_NAMES[label]
    print(f"{display:<25} {acc:>10.1%} {f1:>10.4f} {auc:>10.4f}")
    results.append({
        'Label':    display,
        'Accuracy': acc,
        'F1 Score': f1,
        'AUC':      auc
    })

print("─" * 58)

overall_acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
overall_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
try:
    overall_auc = roc_auc_score(all_labels, all_probs, average='macro')
except ValueError:
    overall_auc = float('nan')

print(f"{'OVERALL':<25} {overall_acc:>10.1%} {overall_f1:>10.4f} {overall_auc:>10.4f}")
print("═" * 62)

# ─── Interpretation ───────────────────────────────────────────────────────────
print("\n📊 Interpretation:")
if overall_auc >= 0.90:
    print("  ✓ Excellent — AUC ≥ 0.90, model is performing very well.")
elif overall_auc >= 0.80:
    print("  ✓ Good — AUC ≥ 0.80, solid performance with room to improve.")
elif overall_auc >= 0.70:
    print("  △ Fair — AUC ≥ 0.70, consider more training or augmentation.")
else:
    print("  ✗ Poor — AUC < 0.70, model needs more training or data.")

if overall_f1 >= 0.80:
    print("  ✓ F1 is strong — good balance of precision and recall.")
elif overall_f1 >= 0.60:
    print("  △ F1 is moderate — model may struggle with some classes.")
else:
    print("  ✗ F1 is low — consider more epochs or rebalancing data.")

# ─── Per-class weak flag ──────────────────────────────────────────────────────
print("\n⚠️  Weak class check (AUC < 0.80):")
weak_found = False
for r in results:
    if not np.isnan(r['AUC']) and r['AUC'] < 0.80:
        print(f"  → {r['Label']:<25} AUC: {r['AUC']:.4f}")
        weak_found = True
if not weak_found:
    print("  All classes above 0.80 ✓")

# ─── Save ─────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.loc[len(df)] = {
    'Label': 'OVERALL', 'Accuracy': overall_acc,
    'F1 Score': overall_f1, 'AUC': overall_auc
}
df.to_csv("evaluation_results.csv", index=False)
print(f"\n  Results saved → evaluation_results.csv")
print("═" * 62)