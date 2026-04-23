"""
cnn_model.py — ResNet-50 with auto-detecting classifier head.

Your saved checkpoint has:
  classifier.0  Linear(2048,512)
  classifier.1  BatchNorm1d(512)    ← running_mean/var
  classifier.2  ReLU
  classifier.3  Dropout
  classifier.4  Linear(512,128)
  classifier.5  BatchNorm1d(128)    ← running_mean/var  (this is Variant B)
  classifier.6  ReLU
  classifier.7  Dropout
  classifier.8  Linear(128,2)       ← classifier.8.weight / bias
"""

import torch
import torch.nn as nn
from torchvision import models


def _make_backbone(pretrained=True):
    weights  = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet50(weights=weights)
    for name, p in backbone.named_parameters():
        if "layer1" in name or "layer2" in name:
            p.requires_grad = False
    return backbone


# ── Variant A: classifier.7 is the final Linear ──────────────────────────────
class PneumoniaCNN_A(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super().__init__()
        bb = _make_backbone(pretrained)
        self.conv1=bb.conv1; self.bn1=bb.bn1; self.relu=bb.relu
        self.maxpool=bb.maxpool; self.layer1=bb.layer1; self.layer2=bb.layer2
        self.layer3=bb.layer3; self.layer4=bb.layer4; self.avgpool=bb.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),        # 0
            nn.BatchNorm1d(512),         # 1
            nn.ReLU(inplace=True),       # 2
            nn.Dropout(dropout),         # 3
            nn.Linear(512, 128),         # 4
            nn.ReLU(inplace=True),       # 5
            nn.Dropout(dropout/2),       # 6
            nn.Linear(128, num_classes), # 7  ← last
        )
    def forward(self, x):
        x=self.conv1(x); x=self.bn1(x); x=self.relu(x); x=self.maxpool(x)
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        return self.classifier(torch.flatten(self.avgpool(x),1))


# ── Variant B: classifier.8 is the final Linear (extra BN after 2nd linear) ──
class PneumoniaCNN_B(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super().__init__()
        bb = _make_backbone(pretrained)
        self.conv1=bb.conv1; self.bn1=bb.bn1; self.relu=bb.relu
        self.maxpool=bb.maxpool; self.layer1=bb.layer1; self.layer2=bb.layer2
        self.layer3=bb.layer3; self.layer4=bb.layer4; self.avgpool=bb.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),        # 0
            nn.BatchNorm1d(512),         # 1
            nn.ReLU(inplace=True),       # 2
            nn.Dropout(dropout),         # 3
            nn.Linear(512, 128),         # 4
            nn.BatchNorm1d(128),         # 5  ← extra BN (your checkpoint)
            nn.ReLU(inplace=True),       # 6
            nn.Dropout(dropout/2),       # 7
            nn.Linear(128, num_classes), # 8  ← last
        )
    def forward(self, x):
        x=self.conv1(x); x=self.bn1(x); x=self.relu(x); x=self.maxpool(x)
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        return self.classifier(torch.flatten(self.avgpool(x),1))


# ── Safe loader — inspects EVERY key before choosing variant ─────────────────

def get_model_for_checkpoint(ckpt_path: str, device: str = "cpu"):
    """
    Load a checkpoint and return (model, ckpt_dict, variant_str).
    Variant is determined by checking whether 'classifier.8.weight'
    exists in the state_dict — if yes → B, otherwise → A.
    """
    raw  = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Support both bare state_dicts and wrapped dicts
    if "model_state" in raw:
        state = raw["model_state"]
        meta  = raw
    else:
        state = raw
        meta  = {}

    # Detect variant from actual keys present
    has_8 = any("classifier.8" in k for k in state.keys())
    has_5_bn = any("classifier.5.running_mean" in k for k in state.keys())

    if has_8 or has_5_bn:
        variant = "B"
        model   = PneumoniaCNN_B(pretrained=False)
    else:
        variant = "A"
        model   = PneumoniaCNN_A(pretrained=False)

    print(f"  🔍 Checkpoint keys detected → Variant {variant}")
    print(f"     (classifier.8.weight present: {has_8} | classifier.5.running_mean: {has_5_bn})")

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, meta, variant


def get_model(num_classes=2, pretrained=True, dropout=0.5, variant="B"):
    """Return a fresh (untrained) model. Default variant=B matches your checkpoint."""
    cls = PneumoniaCNN_B if variant == "B" else PneumoniaCNN_A
    return cls(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


if __name__ == "__main__":
    for v in ("A","B"):
        m = get_model(pretrained=False, variant=v)
        o = m(torch.randn(2,3,224,224))
        keys = [k for k in m.classifier.state_dict().keys() if "weight" in k]
        print(f"Variant {v}: output={tuple(o.shape)}  classifier weight keys: {keys}")