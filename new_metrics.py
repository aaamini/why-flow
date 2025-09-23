import torch
from torch import Tensor

@torch.no_grad()
def _knn_radii_loo(Phi: Tensor, k: int = 3) -> Tensor:
    """Leave-one-out kNN radii with auto k clamp to avoid 'k out of range'."""
    N = Phi.size(0)
    if N == 0:
        return torch.empty(0, device=Phi.device, dtype=Phi.dtype)
    k_eff = max(1, min(k, N-1))
    D = torch.cdist(Phi, Phi)
    D.fill_diagonal_(float('inf'))
    vals, _ = torch.topk(D, k_eff, dim=1, largest=False)
    return vals[:, -1]

@torch.no_grad()
def _membership(Phi: Tensor, centers: Tensor, radii: Tensor) -> Tensor:
    """Bool mask: each Phi[i] inside union_j B(centers_j, radii[j])."""
    if Phi.numel() == 0 or centers.numel() == 0:
        return torch.zeros(Phi.size(0), dtype=torch.bool, device=Phi.device)
    D = torch.cdist(Phi, centers)
    return (D <= radii.unsqueeze(0)).any(dim=1)

@torch.no_grad()
def pr_knn(Phi_real: Tensor, Phi_gen: Tensor, k: int = 3) -> dict:
    """Standard Kynkäänniemi PR."""
    r_gen  = _knn_radii_loo(Phi_gen,  k=k)
    r_real = _knn_radii_loo(Phi_real, k=k)
    recall = _membership(Phi_real, Phi_gen,  r_gen ).float().mean().item() if Phi_real.size(0) else float('nan')
    prec   = _membership(Phi_gen,  Phi_real, r_real).float().mean().item() if Phi_gen.size(0)  else float('nan')
    return {"precision": prec, "recall": recall}

@torch.no_grad()
def pr_knn_conditioned(
    Phi_real: Tensor, Phi_gen: Tensor, Phi_train: Tensor,
    *, p_gen: float = 0.0, p_real: float = 0.0, k: int = 3, empty_value=float('nan')
) -> dict:
    """
    PR on *novel* subsets:
      - Keep generated points whose distance to TRAIN is in the upper p_gen quantile.
      - Keep real points whose distance to TRAIN is in the upper p_real quantile.
    p_gen, p_real are in [0,1]. Setting both 0 recovers vanilla PR.
    """
    # Distances to TRAIN
    d_gT = torch.cdist(Phi_gen,  Phi_train).amin(dim=1) if Phi_gen.size(0)  else torch.empty(0, device=Phi_gen.device)
    d_rT = torch.cdist(Phi_real, Phi_train).amin(dim=1) if Phi_real.size(0) else torch.empty(0, device=Phi_real.device)

    # Quantile thresholds (upper tail)
    keep_g = torch.ones_like(d_gT, dtype=torch.bool)
    keep_r = torch.ones_like(d_rT, dtype=torch.bool)
    if Phi_gen.size(0) and p_gen > 0.0:
        tau_g = torch.quantile(d_gT, p_gen)
        keep_g = d_gT >= tau_g
    if Phi_real.size(0) and p_real > 0.0:
        tau_r = torch.quantile(d_rT, p_real)
        keep_r = d_rT >= tau_r

    R = Phi_real[keep_r]
    G = Phi_gen [keep_g]

    if R.size(0) == 0 or G.size(0) == 0:
        return {"precision": empty_value, "recall": empty_value,
                "kept_real_frac": R.size(0)/max(1,Phi_real.size(0)),
                "kept_gen_frac":  G.size(0)/max(1,Phi_gen.size(0))}

    out = pr_knn(R, G, k=k)
    out.update({
        "kept_real_frac": R.size(0)/Phi_real.size(0),
        "kept_gen_frac":  G.size(0)/Phi_gen.size(0),
    })
    return out

# Convenience wrappers matching your two examples:
@torch.no_grad()
def pr_on_novel_gen(Phi_real: Tensor, Phi_gen: Tensor, Phi_train: Tensor, *, q: float = 0.95, k: int = 3):
    return pr_knn_conditioned(Phi_real, Phi_gen, Phi_train, p_gen=q, p_real=0.0, k=k)

@torch.no_grad()
def pr_on_novel_real(Phi_real: Tensor, Phi_gen: Tensor, Phi_train: Tensor, *, q: float = 0.50, k: int = 3):
    return pr_knn_conditioned(Phi_real, Phi_gen, Phi_train, p_gen=0.0, p_real=q, k=k)