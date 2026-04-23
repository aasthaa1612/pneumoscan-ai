"""
gradcam.py — Gradient-weighted Class Activation Mapping.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._gradients   = None
        self._activations = None
        self._hooks       = []
        self._register()

    def _register(self):
        def fwd(_, __, out):      self._activations = out.detach()
        def bwd(_, __, grad_out): self._gradients   = grad_out[0].detach()
        self._hooks.append(self.target_layer.register_forward_hook(fwd))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd))

    def remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def __call__(self, input_tensor: torch.Tensor,
                 class_idx: int = None):
        self.model.eval()
        inp = input_tensor.requires_grad_(True)
        logits = self.model(inp)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam).squeeze().cpu().numpy()

        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        probs = F.softmax(logits, dim=1)
        return cam, int(class_idx), probs[0, class_idx].item()

    @staticmethod
    def overlay_heatmap(cam: np.ndarray, original_image: np.ndarray,
                        alpha: float = 0.45,
                        colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        h, w = original_image.shape[:2]
        heatmap = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
        if original_image.ndim == 2:
            base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            base = original_image[:, :, :3].copy()
        return cv2.addWeighted(heatmap, alpha, base, 1 - alpha, 0)

    @staticmethod
    def pil_to_overlay(cam: np.ndarray, pil_image: Image.Image,
                       alpha: float = 0.45) -> Image.Image:
        arr     = np.array(pil_image.convert("RGB"))
        overlay = GradCAM.overlay_heatmap(cam, arr, alpha=alpha)
        return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))