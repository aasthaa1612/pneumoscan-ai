"""
diagnose.py — Understand why the model is biased.
Run from project root: python backend/diagnose.py

Outputs:
  1. Dataset class counts per split
  2. Model checkpoint info
  3. Predictions on ALL test images at threshold 0.5
  4. Per-class accuracy (NORMAL & PNEUMONIA separately)
  5. Optimal threshold via Youden's J (ROC curve)
  6. Confidence histogram + confusion matrix plots
  7. Saves backend/calibration.json  (used by app.py automatically)
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Clean imports — only what dataset.py actually exports ──────────────────
from backend.models.cnn_model import get_model
from backend.utils.dataset    import get_transforms, ChestXRayDataset

MODEL_PATH = ROOT / "outputs" / "models" / "best_model.pt"
DATA_RAW   = ROOT / "data"   / "raw"    / "chest_xray"
OUT_PLOTS  = ROOT / "outputs" / "plots"
CALIB_FILE = ROOT / "backend" / "calibration.json"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

print("\n" + "="*60)
print("  PNEUMONIA DETECTOR — FULL DIAGNOSIS")
print("="*60)


# ── 1. Dataset counts ──────────────────────────────────────────────────────
print("\n📊 Step 1: Dataset image counts")
for split in ("train", "val", "test"):
    for cls in CLASS_NAMES:
        folder = DATA_RAW / split / cls
        n = len(list(folder.glob("*.jpeg")) +
                list(folder.glob("*.jpg"))  +
                list(folder.glob("*.png")))
        print(f"   {split:6s}/{cls:10s}: {n:5d} images")


# ── 2. Load model ──────────────────────────────────────────────────────────
print("\n📦 Step 2: Loading model checkpoint")
if not MODEL_PATH.exists():
    print(f"❌ No model at {MODEL_PATH}")
    print("   Run: python backend/train.py")
    sys.exit(1)

model = get_model().to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"   Epoch   : {ckpt.get('epoch', '?')}")
print(f"   Val AUC : {ckpt.get('val_auc', 0):.4f}")

existing_thresh = 0.5
if CALIB_FILE.exists():
    calib = json.loads(CALIB_FILE.read_text())
    existing_thresh = calib.get("threshold", 0.5)
    print(f"   Calibration file found — threshold={existing_thresh:.4f}")
else:
    print("   ⚠️  No calibration.json — threshold=0.5 (default)")


# ── 3. Run inference on test set ───────────────────────────────────────────
print("\n🔍 Step 3: Running inference on test images")
tfm = get_transforms("test")
all_labels, all_probs = [], []

for cls_idx, cls in enumerate(CLASS_NAMES):
    folder = DATA_RAW / "test" / cls
    imgs   = (list(folder.glob("*.jpeg")) +
              list(folder.glob("*.jpg"))  +
              list(folder.glob("*.png")))
    print(f"   {cls}: {len(imgs)} images")
    for p in tqdm(imgs, desc=f"   {cls}", leave=False):
        try:
            img = Image.open(p).convert("RGB")
            t   = tfm(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(t)
                prob   = F.softmax(logits, dim=1)[0, 1].item()
            all_labels.append(cls_idx)
            all_probs.append(prob)
        except Exception as e:
            print(f"   ⚠️  Skipping {p.name}: {e}")

labels = np.array(all_labels)
probs  = np.array(all_probs)
preds  = (probs >= 0.5).astype(int)


# ── 4. Per-class accuracy ──────────────────────────────────────────────────
print("\n📋 Step 4: Results at threshold = 0.50")
overall_acc  = (preds == labels).mean()
normal_acc   = (preds[labels == 0] == 0).mean() if (labels == 0).any() else 0
pneum_acc    = (preds[labels == 1] == 1).mean() if (labels == 1).any() else 0

try:
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report
    auc = roc_auc_score(labels, probs)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    print(f"   Overall accuracy  : {overall_acc:.4f}")
    print(f"   NORMAL accuracy   : {normal_acc:.4f}  ← should be > 0.70")
    print(f"   PNEUMONIA accuracy: {pneum_acc:.4f}")
    print(f"   AUC-ROC           : {auc:.4f}")
    print(f"   F1 Score          : {f1:.4f}")
    print()
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
except ImportError:
    print("   Install scikit-learn for full metrics: pip install scikit-learn")
    auc = 0.0


# ── 5. Find optimal threshold ──────────────────────────────────────────────
print("\n🎯 Step 5: Finding optimal decision threshold")
try:
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden     = tpr - fpr
    best_idx   = int(np.argmax(youden))
    best_thresh = float(thresholds[best_idx])

    preds_opt = (probs >= best_thresh).astype(int)
    acc_opt   = (preds_opt == labels).mean()
    norm_opt  = (preds_opt[labels==0] == 0).mean() if (labels==0).any() else 0
    pneu_opt  = (preds_opt[labels==1] == 1).mean() if (labels==1).any() else 0
    f1_opt    = f1_score(labels, preds_opt, average="binary", zero_division=0)

    print(f"   Optimal threshold : {best_thresh:.4f}  (was 0.5000)")
    print(f"   Accuracy          : {acc_opt:.4f}  (was {overall_acc:.4f})")
    print(f"   NORMAL accuracy   : {norm_opt:.4f}  (was {normal_acc:.4f})")
    print(f"   PNEUMONIA accuracy: {pneu_opt:.4f}  (was {pneum_acc:.4f})")
    print(f"   F1                : {f1_opt:.4f}  (was {f1:.4f})")
except Exception as e:
    best_thresh = 0.5
    print(f"   Could not compute optimal threshold: {e}")


# ── 6. Diagnosis verdict ───────────────────────────────────────────────────
print("\n🩺 DIAGNOSIS:")
if normal_acc < 0.40:
    print("   ❌ SEVERE BIAS — model almost always predicts PNEUMONIA")
    print("   Causes: no class weights, only 1 epoch trained, or bad data split")
    print(f"   Quick fix: change threshold 0.5 → {best_thresh:.3f}  (Step 7 does this)")
    print("   Full fix:  python backend/retrain_balanced.py")
elif normal_acc < 0.65:
    print("   ⚠️  MODERATE BIAS — many NORMAL X-rays misclassified")
    print(f"   Fix: raise threshold 0.5 → {best_thresh:.3f}")
else:
    print("   ✅ Model looks reasonably balanced")
    print(f"   Threshold {best_thresh:.3f} may still improve NORMAL accuracy slightly")


# ── 7. Save calibration ────────────────────────────────────────────────────
print("\n💾 Step 6: Saving calibration.json")
calib_data = {
    "temperature": 1.0,
    "threshold":   round(best_thresh, 6),
    "auc":         round(auc, 6),
    "normal_acc_at_best_thresh": round(float(norm_opt) if 'norm_opt' in dir() else normal_acc, 4),
}
CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
CALIB_FILE.write_text(json.dumps(calib_data, indent=2))
print(f"   Saved → {CALIB_FILE}")
print(f"   threshold = {best_thresh:.4f}")
print(f"   app.py will use this automatically on next restart")


# ── 8. Plots ───────────────────────────────────────────────────────────────
print("\n📊 Step 7: Generating plots")

# Confidence histogram
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(probs[labels == 0], bins=50, alpha=0.65, color="#34d399",
        label="NORMAL", density=True)
ax.hist(probs[labels == 1], bins=50, alpha=0.65, color="#f87171",
        label="PNEUMONIA", density=True)
ax.axvline(0.5,         color="white",  lw=1.5, linestyle="--",
           label="Default 0.5")
ax.axvline(best_thresh, color="#fbbf24", lw=1.5, linestyle="-",
           label=f"Optimal {best_thresh:.3f}")
ax.set_xlabel("P(PNEUMONIA)"); ax.set_ylabel("Density")
ax.set_title("Confidence Distribution — test set", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PLOTS / "confidence_histogram.png", dpi=150)
plt.close()
print(f"   Saved: confidence_histogram.png")

# ROC curve
try:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#60a5fa", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1],"--", color="gray", lw=1)
    ax.scatter(fpr[best_idx], tpr[best_idx], color="#fbbf24", s=90, zorder=5,
               label=f"Best thresh = {best_thresh:.3f}")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold"); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "roc_curve.png", dpi=150)
    plt.close()
    print(f"   Saved: roc_curve.png")
except Exception:
    pass

# Confusion matrices side-by-side
try:
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, p, title in [
        (ax1, preds,      "threshold = 0.50 (default)"),
        (ax2, preds_opt,  f"threshold = {best_thresh:.3f} (optimal)"),
    ]:
        cm = confusion_matrix(labels, p)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "confusion_matrices.png", dpi=150)
    plt.close()
    print(f"   Saved: confusion_matrices.png")
except Exception as e:
    print(f"   Could not save confusion matrix: {e}")

print(f"\n✅ Diagnosis complete. Plots → {OUT_PLOTS}")
print("\n▶️  Restart the Flask server to apply the calibration:")
print("     python backend/app.py")
print("\n   Or for a full retrain (recommended):")
print("     python backend/retrain_balanced.py")