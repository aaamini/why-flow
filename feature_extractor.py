
"""
feature_extractor.py
--------------------
Inception-v3 pool3 feature extractor for metrics.

- Expects image tensors in [0,1], shape [N, C, H, W], C=1 or 3.
- Resizes to 299x299, normalizes with ImageNet mean/std.
- Returns float32 features of shape [N, 2048] on the specified device.

If torchvision is unavailable, raises an ImportError.
"""

from __future__ import annotations
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class InceptionPool3(nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        try:
            import torchvision
            from torchvision.models import inception_v3
        except Exception as e:
            raise ImportError("torchvision is required for Inception features") from e

        # Newer torchvision uses Weights enums
        try:
            weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            self.model = inception_v3(weights=weights, aux_logits=True)
        except Exception:
            self.model = inception_v3(pretrained=True, aux_logits=True)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Register hook to capture pre-fc pooled features (pool3)
        self._feat = None
        def hook_fn(module, inp, out):
            # out: [N, 2048, 1, 1]
            self._feat = out.flatten(1)
        self.model.avgpool.register_forward_hook(hook_fn)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(device)

        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1,3,1,1)
        self.std  = torch.tensor(IMAGENET_STD, device=device).view(1,3,1,1)

    @torch.no_grad()
    def forward(self, imgs: Tensor, *, batch_size: int = 64) -> Tensor:
        """
        imgs: [N, C, H, W], values in [0,1]
        returns: [N, 2048] float32
        """
        assert imgs.dim() == 4, "imgs should be [N,C,H,W]"
        N, C, H, W = imgs.shape
        if C == 1:
            imgs = imgs.repeat(1,3,1,1)
        elif C == 3:
            pass
        else:
            raise ValueError("C must be 1 or 3")

        feats = []
        for i in range(0, N, batch_size):
            x = imgs[i:i+batch_size].to(self.device).float()
            x = F.interpolate(x, size=(299,299), mode="bilinear", align_corners=False) # Resize to 299x299
            x = (x - self.mean) / self.std # Normalize
            _ = self.model(x) # Hook fills self._feat on self.device
            feats.append(self._feat.float()) # returns on device
        return torch.cat(feats, dim=0)



class Whitener:
    def __init__(self, eps: float = 1e-6):
        self.mu: Optional[Tensor] = None
        self.W:  Optional[Tensor] = None
        self.eps = eps
    @torch.no_grad()
    def fit(self, Phi_real: Tensor) -> "Whitener":
        self.mu = Phi_real.mean(dim=0, keepdim=True)
        X = Phi_real - self.mu
        C = (X.T @ X) / max(1, (X.size(0) - 1))
        evals, evecs = torch.linalg.eigh(C)
        evals = torch.clamp(evals, min=self.eps)
        inv_sqrt = 1.0 / torch.sqrt(evals)
        self.W = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.T
        self.W = self.W.to(device=Phi_real.device, dtype=Phi_real.dtype)
        return self
    @torch.no_grad()
    def transform(self, Phi: Tensor) -> Tensor:
        assert self.mu is not None and self.W is not None, "Call fit() first."
        return (Phi - self.mu) @ self.W