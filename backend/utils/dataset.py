"""
dataset.py — Dataset, transforms, DataLoader factory.
No external args beyond split name. Zero import surprises.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_RAW  = ROOT / "data" / "raw" / "chest_xray"
WEIGHTS_F = ROOT / "backend" / "class_weights.json"

CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IMG_SIZE     = 224
SEED         = 42


# ── X-Ray specific augmentation ───────────────────────────────────────────────

class XRayAugment:
    """Simulates scanner variability — autocontrast, equalize, slight blur."""
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = ImageOps.autocontrast(img, cutoff=random.uniform(0, 2))
        if random.random() < 0.3:
            img = ImageOps.equalize(img)
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        return img


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.08),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                    shear=8, scale=(0.88, 1.12)),
            transforms.ColorJitter(brightness=0.25, contrast=0.3),
            XRayAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_tta_transforms() -> List[transforms.Compose]:
    """5 deterministic test-time augmentation views."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    norm = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return [
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + norm),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.RandomHorizontalFlip(p=1.0)] + norm),
        transforms.Compose([transforms.Resize((240, 240)),
                            transforms.CenterCrop(IMG_SIZE)] + norm),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.ColorJitter(brightness=0.15, contrast=0.15)] + norm),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            XRayAugment()] + norm),
    ]


# ── Dataset ───────────────────────────────────────────────────────────────────

class ChestXRayDataset(Dataset):
    """
    Loads chest X-ray images for a given split.
    Kaggle val set has only 16 images, so we merge train+val
    and re-split 85/15 stratified by class.
    """
    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split     = split
        self.transform = get_transforms("train" if split == "train" else "test")
        self.samples: List[Tuple[Path, int]] = []

        if split == "test":
            self._load_folder("test")
        else:
            # Merge Kaggle train + val, re-split 85/15
            all_samples: List[Tuple[Path, int]] = []
            for s in ("train", "val"):
                self._load_folder(s, out=all_samples)

            rng = random.Random(SEED)
            by_class: dict = {0: [], 1: []}
            for item in all_samples:
                by_class[item[1]].append(item)

            train_s, val_s = [], []
            for cls_samples in by_class.values():
                rng.shuffle(cls_samples)
                cut = int(len(cls_samples) * 0.85)
                train_s.extend(cls_samples[:cut])
                val_s.extend(cls_samples[cut:])

            self.samples = train_s if split == "train" else val_s

        if not self.samples:
            raise FileNotFoundError(
                f"No images found for split='{split}'.\n"
                f"Expected dataset at: {DATA_RAW}\n"
                "Structure: chest_xray/train|val|test/NORMAL|PNEUMONIA/"
            )

    def _load_folder(self, folder_name: str, out: list = None):
        target = out if out is not None else self.samples
        base   = DATA_RAW / folder_name
        for cls, idx in CLASS_TO_IDX.items():
            d = base / cls
            if d.exists():
                for ext in ("*.jpeg", "*.jpg", "*.png"):
                    for p in d.glob(ext):
                        target.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def get_labels(self) -> List[int]:
        return [s[1] for s in self.samples]

    def class_counts(self) -> dict:
        lbl = self.get_labels()
        return {"NORMAL": lbl.count(0), "PNEUMONIA": lbl.count(1)}


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loader(split: str,
                batch_size: int = 16,
                num_workers: int = 0,
                use_weighted_sampler: bool = True) -> DataLoader:
    dataset = ChestXRayDataset(split)
    counts  = dataset.class_counts()
    print(f"  [{split:5s}] NORMAL={counts['NORMAL']}  PNEUMONIA={counts['PNEUMONIA']}")

    if split == "train" and use_weighted_sampler:
        n0, n1 = counts["NORMAL"], counts["PNEUMONIA"]
        total  = n0 + n1
        w0     = total / (2.0 * n0)
        w1     = total / (2.0 * n1)

        WEIGHTS_F.parent.mkdir(parents=True, exist_ok=True)
        with open(WEIGHTS_F, "w") as f:
            json.dump({"NORMAL": round(w0, 6), "PNEUMONIA": round(w1, 6)}, f, indent=2)
        print(f"         class weights → NORMAL:{w0:.4f}  PNEUMONIA:{w1:.4f}")

        sample_weights = [w0 if l == 0 else w1 for l in dataset.get_labels()]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float64),
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=False)
    else:
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=(split == "train"),
                          num_workers=num_workers, pin_memory=False)


if __name__ == "__main__":
    loader = make_loader("train", batch_size=4)
    imgs, labels = next(iter(loader))
    print(f"✅ Batch: {imgs.shape}  labels: {labels.tolist()}")