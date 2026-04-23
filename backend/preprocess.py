"""
preprocess.py — Analyze dataset and compute class weights.
Run from project root: python backend/preprocess.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "raw" / "chest_xray"
OUT_PLOTS = ROOT / "outputs" / "plots"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

SPLITS      = ["train", "val", "test"]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

print("\n" + "="*60)
print("  PNEUMONIA DETECTOR — PREPROCESSING")
print("="*60)

if not DATA_RAW.exists():
    print(f"\n❌ Dataset not found at: {DATA_RAW}")
    print("   Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("   Extract 'chest_xray' folder to: data/raw/")
    sys.exit(1)

print(f"\n✅ Dataset found at: {DATA_RAW}")

# ── Count images ──────────────────────────────────────────────────────────
counts = {}
for split in SPLITS:
    counts[split] = {}
    for cls in CLASS_NAMES:
        folder = DATA_RAW / split / cls
        n = len(list(folder.glob("*.jpeg")) +
                list(folder.glob("*.jpg"))  +
                list(folder.glob("*.png")))
        counts[split][cls] = n
        print(f"   {split:6s}/{cls:10s}: {n:5d}")

# ── Class distribution plot ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, split in zip(axes, SPLITS):
    vals  = [counts[split][c] for c in CLASS_NAMES]
    bars  = ax.bar(CLASS_NAMES, vals, color=["#34d399","#f87171"],
                   edgecolor="black", width=0.5)
    ax.set_title(split.upper(), fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(v), ha="center", fontweight="bold")
    n0, n1 = vals[0], vals[1]
    ratio  = n1/n0 if n0 else 0
    ax.set_xlabel(f"Total: {n0+n1} | Ratio: {ratio:.2f}x")
plt.suptitle("Class Distribution per Split", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_PLOTS / "class_distribution.png", dpi=150)
plt.close()
print(f"\n  📊 Saved: class_distribution.png")

# ── Class weights ─────────────────────────────────────────────────────────
n0    = counts["train"]["NORMAL"]
n1    = counts["train"]["PNEUMONIA"]
total = n0 + n1
w0    = total / (2 * n0)
w1    = total / (2 * n1)
print(f"\n  ⚖️  Class weights (for loss function):")
print(f"     NORMAL    → {w0:.4f}")
print(f"     PNEUMONIA → {w1:.4f}")

out = ROOT / "backend" / "class_weights.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"NORMAL": round(w0,6), "PNEUMONIA": round(w1,6)}, indent=2))
print(f"  💾 Saved → {out}")

print("\n✅ Preprocessing complete!")
print("\n▶️  Next:  python backend/train.py")
print("   Or for balanced retrain:  python backend/retrain_balanced.py")