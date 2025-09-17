import torch
from torch import Tensor


# --- Metrics ---
@torch.no_grad()
def mmd_rbf(x: Tensor, y: Tensor, sigmas=(0.1,0.2,0.5,1.0)) -> float:
    def k(a,b,s2): return torch.exp(-torch.cdist(a,b)**2 / (2*s2))
    mmd2 = 0.0
    for s in sigmas:
        s2 = float(s*s)
        Kxx = k(x,x,s2).mean()
        Kyy = k(y,y,s2).mean()
        Kxy = k(x,y,s2).mean()
        mmd2 += (Kxx + Kyy - 2*Kxy)
    return float(mmd2 / len(sigmas))

@torch.no_grad()
def energy_distance(x: Tensor, y: Tensor) -> float:
    # E||X-Y'|| - 0.5 E||X-X'|| - 0.5 E||Y-Y'||
    dxy = torch.cdist(x,y).mean()
    dxx = torch.cdist(x,x).mean()
    dyy = torch.cdist(y,y).mean()
    return float(2*dxy - dxx - dyy)

@torch.no_grad()
def nn_rmse_to_targets(x: Tensor, Y: Tensor) -> float:
    return torch.cdist(x, Y).min(dim=1).values.pow(2).mean().sqrt().item()