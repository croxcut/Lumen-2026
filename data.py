import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─── Label columns ────────────────────────────────────────────────────────────
LABEL_COLS = ['ILD', 'Lung Cancer', 'Normal', 'Tuberculosis', 'COVID', 'Pneumonia']
NUM_LABELS = len(LABEL_COLS)  # 6

# ─── Dataset base directory ───────────────────────────────────────────────────
DATASET_BASE = r'D:\projects\lungmaxvit\dataset'

# ─── Image directories ────────────────────────────────────────────────────────
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

# ─── Pre-build file index at import time ──────────────────────────────────────
# Walks all dirs once and maps:
#   relative path → full path  (for new datasets)
#   bare filename  → full path  (for old datasets, first occurrence wins)
print("Building file index...")
_FILE_INDEX = {}
for _d in ALL_IMG_DIRS:
    for _root, _, _files in os.walk(_d):
        for _f in _files:
            if _f.lower().endswith(('.png', '.jpg', '.jpeg')):
                _full = os.path.join(_root, _f)
                _rel  = os.path.relpath(_full, DATASET_BASE)
                # Relative path (always stored)
                _FILE_INDEX[_rel] = _full
                # Bare filename (first occurrence only)
                if _f not in _FILE_INDEX:
                    _FILE_INDEX[_f] = _full
print(f"File index built: {len(_FILE_INDEX)} entries")

# ─── Single folder Dataset ────────────────────────────────────────────────────
class LungCTDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[LABEL_COLS].values.astype(float), dtype=torch.float32)
        return {'image': image, 'label': label}

# ─── Multi-folder Dataset ─────────────────────────────────────────────────────
class MultiDirDataset(Dataset):
    def __init__(self, csv_path, img_dirs, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dirs  = img_dirs  # kept for compatibility
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        filename = row['filename']

        # Normalize path separators
        filename_norm = filename.replace('/', os.sep).replace('\\', os.sep)

        # 1. Try relative path (new datasets: FibrosisDataset\patient\slice.png)
        # 2. Try bare filename  (old datasets: slice_0028.png)
        img_path = (
            _FILE_INDEX.get(filename_norm) or
            _FILE_INDEX.get(os.path.basename(filename_norm))
        )

        if img_path is None:
            raise FileNotFoundError(f"Image not found: {filename}")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[LABEL_COLS].values.astype(float), dtype=torch.float32)
        return {'image': image, 'label': label}

# ─── Transforms ───────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),       # keep — left/right flip is valid
    transforms.RandomRotation(10),           # reduce from 15 to 10
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # reduce translation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ─── Datasets ─────────────────────────────────────────────────────────────────
train_dataset = MultiDirDataset(
    csv_path=r"D:\projects\lungmaxvit\dataset\final_train.csv",
    img_dirs=ALL_IMG_DIRS,
    transform=train_transforms
)

val_dataset = MultiDirDataset(
    csv_path=r"D:\projects\lungmaxvit\dataset\final_valid.csv",
    img_dirs=ALL_IMG_DIRS,
    transform=val_transforms
)

test_dataset = MultiDirDataset(
    csv_path=r"D:\projects\lungmaxvit\dataset\final_test.csv",
    img_dirs=ALL_IMG_DIRS,
    transform=val_transforms
)

# ─── DataLoaders ──────────────────────────────────────────────────────────────
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=0, pin_memory=False)