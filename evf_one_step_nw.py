import torch
from torch import Tensor
import matplotlib.pyplot as plt

from evf import EmpiricalVectorField, sample_rho_t_empirical
from metrics import nn_rmse_to_targets
from evf_circles_images import generate_circle_images, show_image_grid


@torch.no_grad()
def nw_one_step(Y: Tensor, x_t: Tensor, t: float) -> Tensor:
    """
    Closed-form Nadaraya–Watson estimator from the passage:
    x1 = sum_i y_i K((x_t - t y_i)/(1-t)) / sum_i K((x_t - t y_i)/(1-t))
    with K from standard normal density (up to a shared constant that cancels).
    """
    one_minus_t = max(1e-6, 1.0 - float(t))
    # weights proportional to exp(-||x_t - t y_i||^2 / (2 (1-t)^2))
    x = x_t  # [B,d]
    Yd = Y.to(device=x.device, dtype=x.dtype)  # [n,d]
    x2 = (x**2).sum(-1, keepdim=True)               # [B,1]
    yy2 = (Yd**2).sum(-1, keepdim=True).T           # [1,n]
    xy = x @ Yd.T                                    # [B,n]
    num = x2 - 2.0*float(t)*xy + (float(t)**2) * yy2 # [B,n]
    logw = - num / (2.0 * (one_minus_t**2))
    w = torch.softmax(logw, dim=1)                   # [B,n]
    mu = w @ Yd                                      # [B,d]
    return mu


@torch.no_grad()
def euler_one_step(Y: Tensor, x_t: Tensor, t: float) -> Tensor:
    """One Euler step to t=1 using EVF: x1 = x_t + (1-t) v(t, x_t)."""
    field = EmpiricalVectorField(Y)
    t_tensor = x_t.new_full((x_t.size(0), 1), float(t))
    v = field(t_tensor, x_t)
    return x_t + (1.0 - float(t)) * v


@torch.no_grad()
def run_one_step_experiment(
    t: float = 0.8,
    n_target: int = 60,
    n_samp: int = 2048,
    image_size: int = 32,
    noise_std: float = 0.0,
    thickness: float = 2.0,
    show_vis: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Y = generate_circle_images(
        n_target,
        image_size=image_size,
        noise_std=noise_std,
        thickness=thickness,
        device=device,
        dtype=torch.float32,
    )
    # sample x_t ~ ρ_t
    x_t = sample_rho_t_empirical(Y, t, n_samp)
    # compute x1 by both methods
    x1_euler = euler_one_step(Y, x_t, t)
    x1_nw = nw_one_step(Y, x_t, t)
    # numeric agreement
    max_diff = (x1_euler - x1_nw).abs().max().item()

    # novelty (to training) and fit (to fine grid)
    Y_fine = generate_circle_images(
        max(n_samp, 4096),
        image_size=image_size,
        noise_std=noise_std,
        thickness=thickness,
        device=device,
        dtype=torch.float32,
    )
    novelty_euler = nn_rmse_to_targets(x1_euler, Y)
    novelty_nw    = nn_rmse_to_targets(x1_nw, Y)
    fit_euler = nn_rmse_to_targets(x1_euler, Y_fine)
    fit_nw    = nn_rmse_to_targets(x1_nw, Y_fine)

    print(f"t={t:.3f} | max|Euler - NW| = {max_diff:.3e}")
    print(f"Euler: novelty(NN-RMSE to train)={novelty_euler:.4g} | fit(NN-RMSE to fine)={fit_euler:.4g}")
    print(f"   NW: novelty(NN-RMSE to train)={novelty_nw:.4g} | fit(NN-RMSE to fine)={fit_nw:.4g}")

    if show_vis:
        show_image_grid(Y, image_size=image_size, title="Training targets (Y)")
        show_image_grid(x_t, image_size=image_size, title=f"Samples x_t ~ ρ_t, t={t:.2f}")
        show_image_grid(x1_euler, image_size=image_size, title="x1 via one-step Euler")
        show_image_grid(x1_nw, image_size=image_size, title="x1 via Nadaraya–Watson")


if __name__ == "__main__":
    # You can vary t in (0,1). Larger t gives sharper kernels (smaller (1-t)).
    run_one_step_experiment(
        t=0.8,
        n_target=60,
        n_samp=2048,
        image_size=32,
        noise_std=0.0,
        thickness=2.0,
        show_vis=True,
    )


