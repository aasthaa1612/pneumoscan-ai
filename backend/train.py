"""
train.py — Main training script.
Run from project root: python backend/train.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.cnn_model import get_model
from backend.utils.dataset    import make_loader

WEIGHTS_F = ROOT / "backend" / "class_weights.json"
OUT_MODEL = ROOT / "outputs" / "models" / "best_model.pt"
OUT_PLOTS = ROOT / "outputs" / "plots"
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    "epochs"      : 30,
    "batch_size"  : 16,
    "lr"          : 1e-4,
    "weight_decay": 1e-4,
    "dropout"     : 0.5,
    "patience"    : 7,
    "num_workers" : 0,    # set to 4 on Linux/Mac
}

print(f"\n{'='*60}")
print(f"  PNEUMONIA DETECTOR — TRAINING")
print(f"  Device: {DEVICE}")
print(f"{'='*60}")


def get_criterion():
    if WEIGHTS_F.exists():
        cw = json.loads(WEIGHTS_F.read_text())
        weights = torch.tensor([cw["NORMAL"], cw["PNEUMONIA"]], dtype=torch.float).to(DEVICE)
        print(f"  ⚖️  Class weights: NORMAL={cw['NORMAL']:.3f}  PNEUMONIA={cw['PNEUMONIA']:.3f}")
    else:
        weights = None
    return nn.CrossEntropyLoss(weight=weights)


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
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            total_loss += loss.item() * len(labels)
            all_lbl.extend(labels.cpu().numpy())
            all_prob.extend(prob)

    lbl  = np.array(all_lbl)
    prob = np.array(all_prob)
    pred = (prob >= 0.5).astype(int)
    acc  = (pred == lbl).mean()
    try:
        auc = roc_auc_score(lbl, prob) if len(set(lbl)) > 1 else 0.0
        f1  = f1_score(lbl, pred, average="binary", zero_division=0)
    except Exception:
        auc = f1 = 0.0
    norm_acc = (pred[lbl==0] == 0).mean() if (lbl==0).any() else 0
    pneu_acc = (pred[lbl==1] == 1).mean() if (lbl==1).any() else 0
    return total_loss / len(lbl), acc, auc, f1, norm_acc, pneu_acc


# ── Data ───────────────────────────────────────────────────────────────────
print("\n📦 Loading datasets...")
train_ldr = make_loader("train", CFG["batch_size"], CFG["num_workers"])
val_ldr   = make_loader("val",   CFG["batch_size"], CFG["num_workers"],
                        use_weighted_sampler=False)

model     = get_model(dropout=CFG["dropout"]).to(DEVICE)
criterion = get_criterion()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG["lr"], weight_decay=CFG["weight_decay"],
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, verbose=True)

history   = {k: [] for k in ["tr_loss","tr_auc","tr_f1","vl_loss","vl_auc","vl_f1","vl_norm","vl_pneu"]}
best_auc, patience_cnt = 0.0, 0

print("\n🚀 Training started...\n")
for epoch in range(1, CFG["epochs"] + 1):
    tr_loss, tr_acc, tr_auc, tr_f1, _, _        = run_epoch(model, train_ldr, criterion, optimizer, True)
    vl_loss, vl_acc, vl_auc, vl_f1, vl_n, vl_p = run_epoch(model, val_ldr,   criterion, None,      False)
    scheduler.step(vl_auc)

    for k, v in zip(history, [tr_loss,tr_auc,tr_f1,vl_loss,vl_auc,vl_f1,vl_n,vl_p]):
        history[k].append(v)

    flag = "💾" if vl_auc > best_auc else "  "
    print(f"Ep {epoch:2d}/{CFG['epochs']} | "
          f"tr_loss:{tr_loss:.4f} tr_auc:{tr_auc:.4f} | "
          f"vl_auc:{vl_auc:.4f} norm:{vl_n:.3f} pneu:{vl_p:.3f} {flag}")

    if vl_auc > best_auc:
        best_auc = vl_auc
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_auc": vl_auc, "cfg": CFG,
        }, OUT_MODEL)
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= CFG["patience"]:
            print(f"\n⏹️  Early stopping at epoch {epoch}"); break

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (k1, k2, title) in zip(axes.flat, [
    ("tr_loss","vl_loss","Loss"),
    ("tr_auc", "vl_auc", "AUC-ROC"),
    ("tr_f1",  "vl_f1",  "F1 Score"),
    ("vl_norm","vl_pneu","Val Per-class Accuracy"),
]):
    ax.plot(history[k1], label=k1, marker="o", ms=3)
    ax.plot(history[k2], label=k2, marker="s", ms=3)
    ax.set_title(title, fontweight="bold"); ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle("Training Curves", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_PLOTS / "training_curves.png", dpi=150); plt.close()

print(f"\n✅ Done! Best val AUC: {best_auc:.4f}")
print(f"   Model → {OUT_MODEL}")
print("\n▶️  Next steps:")
print("   python backend/fix_threshold.py  ← calibrate threshold")
print("   python backend/app.py            ← start API server")