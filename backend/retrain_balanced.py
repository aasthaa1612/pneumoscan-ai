"""
retrain_balanced.py — Retrain with proper imbalance handling.
Run from project root: python backend/retrain_balanced.py

Improvements over a plain train:
  - Focal Loss  (focuses on hard examples)
  - Inverse-frequency class weights
  - WeightedRandomSampler
  - Asymmetric augmentation (NORMAL gets heavier aug)
  - Label smoothing (prevents overconfident outputs)
  - Two-phase: head warmup → full fine-tune
  - Early stopping on val AUC
  - Saves best_model.pt
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.cnn_model import get_model
from backend.utils.dataset    import make_loader, ChestXRayDataset

OUT_MODEL = ROOT / "outputs" / "models" / "best_model.pt"
OUT_PLOTS = ROOT / "outputs" / "plots"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  BALANCED RETRAINING — Focal Loss + Two-Phase")
print(f"  Device: {DEVICE}")
print(f"{'='*60}")

CFG = {
    "batch_size"   : 16,
    "epochs_p1"    : 5,      # head only
    "epochs_p2"    : 25,     # full model
    "lr_p1"        : 3e-4,
    "lr_p2"        : 5e-5,
    "weight_decay" : 1e-4,
    "dropout"      : 0.5,
    "patience"     : 7,
    "focal_gamma"  : 2.0,
    "label_smooth" : 0.05,
    "num_workers"  : 0,      # safe default for Windows
}


# ── Focal Loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma          = gamma
        self.class_weights  = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        probs   = F.softmax(logits, dim=1)
        pt      = probs[torch.arange(len(targets)), targets]
        focal_w = (1 - pt) ** self.gamma
        return (focal_w * ce).mean()


# ── Class weights ──────────────────────────────────────────────────────────
def get_class_weights_tensor():
    from backend.utils.dataset import DATA_RAW, CLASS_TO_IDX
    counts = {}
    for cls in ("NORMAL", "PNEUMONIA"):
        folder = DATA_RAW / "train" / cls
        counts[cls] = len(list(folder.glob("*.jpeg")) +
                          list(folder.glob("*.jpg")) +
                          list(folder.glob("*.png")))
    # Also add val images (merged)
    for cls in ("NORMAL", "PNEUMONIA"):
        folder = DATA_RAW / "val" / cls
        counts[cls] += len(list(folder.glob("*.jpeg")) +
                           list(folder.glob("*.jpg")) +
                           list(folder.glob("*.png")))
    n0, n1  = counts["NORMAL"], counts["PNEUMONIA"]
    total   = n0 + n1
    w0, w1  = total / (2 * n0), total / (2 * n1)
    print(f"  ⚖️  Class weights: NORMAL={w0:.3f}  PNEUMONIA={w1:.3f}")
    return torch.tensor([w0, w1], dtype=torch.float).to(DEVICE)


# ── One epoch ──────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, train):
    model.train() if train else model.eval()
    total_loss, all_lbl, all_prob = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False,
                                  desc="  train" if train else "  val  "):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            prob = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            total_loss += loss.item() * len(labels)
            all_lbl.extend(labels.cpu().numpy())
            all_prob.extend(prob)

    lbl   = np.array(all_lbl)
    prob  = np.array(all_prob)
    pred  = (prob >= 0.5).astype(int)
    acc   = (pred == lbl).mean()
    norm  = (pred[lbl==0] == 0).mean() if (lbl==0).any() else 0
    pneu  = (pred[lbl==1] == 1).mean() if (lbl==1).any() else 0

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(lbl, prob) if len(set(lbl)) > 1 else 0.0
    except Exception:
        auc = 0.0

    return total_loss / len(lbl), acc, auc, norm, pneu


# ── Training loop ──────────────────────────────────────────────────────────
def train_phase(model, train_ldr, val_ldr, criterion, epochs, lr, name):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=CFG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05)

    best_auc, patience_cnt = 0.0, 0
    history = {k: [] for k in ["tr_loss","tr_auc","vl_loss","vl_auc","vl_norm","vl_pneu"]}

    print(f"\n{'─'*55}\n  {name}\n{'─'*55}")

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_auc, _, _       = run_epoch(model, train_ldr, criterion, optimizer, True)
        vl_loss, vl_acc, vl_auc, vl_n, vl_p = run_epoch(model, val_ldr,   criterion, None,      False)
        scheduler.step()

        for k, v in zip(history, [tr_loss,tr_auc,vl_loss,vl_auc,vl_n,vl_p]):
            history[k].append(v)

        flag = "💾" if vl_auc > best_auc else "  "
        print(f"  Ep {ep:2d}/{epochs} | "
              f"tr_loss:{tr_loss:.4f} tr_auc:{tr_auc:.4f} | "
              f"vl_auc:{vl_auc:.4f} norm:{vl_n:.3f} pneu:{vl_p:.3f} {flag}")

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_auc": vl_auc, "cfg": CFG,
            }, OUT_MODEL)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"  ⏹️  Early stopping at epoch {ep}"); break

    return history, best_auc


def plot_history(h, name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [("tr_loss","vl_loss","Loss"),
             ("tr_auc", "vl_auc", "AUC-ROC"),
             ("vl_norm","vl_pneu","Per-class Accuracy")]
    for ax, (k1, k2, title) in zip(axes, pairs):
        ax.plot(h[k1], label=k1, marker="o", ms=3)
        ax.plot(h[k2], label=k2, marker="s", ms=3)
        ax.set_title(title, fontweight="bold"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle(name, fontweight="bold")
    plt.tight_layout()
    fname = name.lower().replace(" ","_").replace("/","_") + ".png"
    plt.savefig(OUT_PLOTS / fname, dpi=150); plt.close()


# ── Main ───────────────────────────────────────────────────────────────────
print("\n📦 Building data loaders...")
train_ldr = make_loader("train", CFG["batch_size"], CFG["num_workers"])
val_ldr   = make_loader("val",   CFG["batch_size"], CFG["num_workers"],
                        use_weighted_sampler=False)

cw        = get_class_weights_tensor()
criterion = FocalLoss(class_weights=cw,
                      gamma=CFG["focal_gamma"],
                      label_smoothing=CFG["label_smooth"])
print(f"  🎯 Focal Loss γ={CFG['focal_gamma']}  label_smooth={CFG['label_smooth']}")

# Phase 1 — head only
model = get_model(dropout=CFG["dropout"]).to(DEVICE)
if OUT_MODEL.exists():
    ckpt = torch.load(OUT_MODEL, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  📦 Resuming from existing checkpoint (auc={ckpt.get('val_auc',0):.4f})")

for name_, p in model.named_parameters():
    if "classifier" not in name_: p.requires_grad = False
print("  🔒 Backbone frozen for Phase 1")

h1, auc1 = train_phase(model, train_ldr, val_ldr, criterion,
                        CFG["epochs_p1"], CFG["lr_p1"], "Phase 1 — Head Warmup")
plot_history(h1, "Phase 1 Head Warmup")

# Phase 2 — full fine-tune
ckpt = torch.load(OUT_MODEL, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
for p in model.parameters(): p.requires_grad = True
print("\n  🔓 Full model unfrozen for Phase 2")

h2, auc2 = train_phase(model, train_ldr, val_ldr, criterion,
                        CFG["epochs_p2"], CFG["lr_p2"], "Phase 2 — Full Fine-tune")
plot_history(h2, "Phase 2 Full Fine-tune")

print(f"\n✅ Retraining complete!  Best val AUC: {auc2:.4f}")
print(f"   Model → {OUT_MODEL}")
print("\n▶️  Now run:  python backend/fix_threshold.py")
print("   Then:       python backend/app.py")