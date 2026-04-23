"""
fix_threshold.py — Calibrate threshold + temperature without retraining.
Run from project root: python backend/fix_threshold.py

What it does:
  1. Loads your existing trained model
  2. Runs it on val+test images
  3. Learns temperature T to fix overconfident softmax outputs
  4. Finds optimal decision threshold (Youden's J on ROC curve)
  5. Saves backend/calibration.json  (app.py reads this automatically)
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.cnn_model import get_model
from backend.utils.dataset    import get_transforms

MODEL_PATH = ROOT / "outputs" / "models" / "best_model.pt"
DATA_RAW   = ROOT / "data"   / "raw"    / "chest_xray"
CALIB_FILE = ROOT / "backend" / "calibration.json"

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

print("\n" + "="*60)
print("  THRESHOLD CALIBRATION + TEMPERATURE SCALING")
print("="*60)

# ── Load model ─────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print(f"❌ No model at {MODEL_PATH}")
    sys.exit(1)

model = get_model().to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"✅ Model loaded  val_auc={ckpt.get('val_auc',0):.4f}")


# ── Collect logits on calibration set ────────────────────────────────────
def collect(split):
    tfm = get_transforms("test")
    all_logits, all_labels = [], []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        folder = DATA_RAW / split / cls
        imgs   = (list(folder.glob("*.jpeg")) +
                  list(folder.glob("*.jpg"))  +
                  list(folder.glob("*.png")))
        for p in tqdm(imgs, desc=f"  {split}/{cls}", leave=False):
            try:
                img = Image.open(p).convert("RGB")
                t   = tfm(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(t)
                all_logits.append(logits.cpu())
                all_labels.append(cls_idx)
            except Exception:
                pass
    if not all_logits:
        return None, None
    return torch.cat(all_logits, dim=0), torch.tensor(all_labels)

# Use val; fall back to test if val is tiny
logits_v, labels_v = collect("val")
n_normal_val = (labels_v == 0).sum().item() if labels_v is not None else 0
if n_normal_val < 10:
    print(f"  ⚠️  Val NORMAL only {n_normal_val} images — using test set for calibration")
    logits_v, labels_v = collect("test")

print(f"  Calibration set: {len(labels_v)} images")


# ── Before ────────────────────────────────────────────────────────────────
probs_before = F.softmax(logits_v, dim=1)[:, 1].numpy()
preds_before = (probs_before >= 0.5).astype(int)
lbl          = labels_v.numpy()
norm_acc_before = (preds_before[lbl==0] == 0).mean() if (lbl==0).any() else 0
pneu_acc_before = (preds_before[lbl==1] == 1).mean() if (lbl==1).any() else 0
print(f"\n  BEFORE — threshold=0.50")
print(f"    NORMAL accuracy  : {norm_acc_before:.4f}")
print(f"    PNEUMONIA accuracy: {pneu_acc_before:.4f}")


# ── Temperature scaling ───────────────────────────────────────────────────
print("\n  🌡️  Fitting temperature scaling...")
temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
criterion   = torch.nn.CrossEntropyLoss()
optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)

def step():
    optimizer.zero_grad()
    loss = criterion(logits_v / temperature, labels_v)
    loss.backward()
    return loss

optimizer.step(step)
T = max(0.1, temperature.item())   # clamp — negative T makes no sense
print(f"    Learned T = {T:.4f}  (>1 means model was overconfident)")


# ── Find optimal threshold ─────────────────────────────────────────────────
print("\n  🎯 Finding optimal threshold...")
try:
    from sklearn.metrics import roc_curve, roc_auc_score, classification_report
    probs_cal = F.softmax(logits_v / T, dim=1)[:, 1].detach().numpy()
    fpr, tpr, thresholds = roc_curve(lbl, probs_cal)
    best_idx    = int(np.argmax(tpr - fpr))
    best_thresh = float(thresholds[best_idx])
    auc         = roc_auc_score(lbl, probs_cal)

    preds_after = (probs_cal >= best_thresh).astype(int)
    norm_acc_after = (preds_after[lbl==0] == 0).mean() if (lbl==0).any() else 0
    pneu_acc_after = (preds_after[lbl==1] == 1).mean() if (lbl==1).any() else 0

    print(f"\n  AFTER  — T={T:.4f}  threshold={best_thresh:.4f}")
    print(f"    NORMAL accuracy  : {norm_acc_after:.4f}  (was {norm_acc_before:.4f})")
    print(f"    PNEUMONIA accuracy: {pneu_acc_after:.4f}  (was {pneu_acc_before:.4f})")
    print(f"    AUC-ROC          : {auc:.4f}")
    print()
    print(classification_report(lbl, preds_after, target_names=CLASS_NAMES))

except ImportError:
    print("  ⚠️  scikit-learn not found — using threshold=0.6 as safe default")
    best_thresh = 0.6
    auc         = 0.0


# ── Save calibration ───────────────────────────────────────────────────────
calib = {
    "temperature" : round(T, 6),
    "threshold"   : round(best_thresh, 6),
    "auc"         : round(auc, 6),
}
CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
CALIB_FILE.write_text(json.dumps(calib, indent=2))
print(f"\n✅ Saved → {CALIB_FILE}")
print(f"   temperature = {T:.4f}")
print(f"   threshold   = {best_thresh:.4f}")
print("\n▶️  Restart backend:  python backend/app.py")
print("   The server will load calibration.json automatically.")