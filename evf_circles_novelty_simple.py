#%%
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from evf import EmpiricalVectorField, Integrator, uniform_grid, near1_grid, sample_rho_t_empirical
from metrics import nn_rmse_to_targets
from circle_images import generate_circle_images, show_image_grid
import numpy as np


@torch.no_grad()
def run_novelty_simple(
    n_target: int = 12,
    n_samp: int = 2048,
    image_size: int = 32,
    method: str = "rk2",
    grid: str = "uniform",
    n_steps: int = 6,
    use_double: bool = True,
    noise_std: float = 0.0,
    thickness: float = 2.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training samples (targets)
    Y = generate_circle_images(
        n_target,
        image_size=image_size,
        noise_std=noise_std,
        thickness=thickness,
        device=device,
        dtype=torch.float32,
    )

    # Finer grid (proxy for true distribution support)
    Y_fine = generate_circle_images(
        max(n_samp, 4096),
        image_size=image_size,
        noise_std=noise_std,
        thickness=thickness,
        device=device,
        dtype=torch.float32,
    )

    # EVF integration and reference mixture at t*
    Yd = Y.double() if use_double else Y
    field = EmpiricalVectorField(Yd)
    integ = Integrator(field)
    t_grid = uniform_grid(n_steps) if grid == "uniform" else near1_grid(n_steps, gamma=4.0)

    x0 = torch.randn(n_samp, Yd.size(1), device=device, dtype=Yd.dtype)
    xT, mids, _ = integ.integrate(x0, t_grid, method=method, return_traj=True)
    t_star = float(mids[-1])
    ref = sample_rho_t_empirical(Yd, t_star, n_samp).float()
    xT = xT.float()

    # Metrics: novelty (NN RMSE to training), fit (NN RMSE to fine grid)
    novelty_ode = nn_rmse_to_targets(xT, Y)
    novelty_ref = nn_rmse_to_targets(ref, Y)
    fit_ode = nn_rmse_to_targets(xT, Y_fine)
    fit_ref = nn_rmse_to_targets(ref, Y_fine)

    print(f"grid={grid}, method={method}, steps={n_steps}, t*={t_star:.6f}")
    print(f"ODE: novelty(NN-RMSE to train)={novelty_ode:.4g} | fit(NN-RMSE to fine)={fit_ode:.4g}")
    print(f"REF: novelty(NN-RMSE to train)={novelty_ref:.4g} | fit(NN-RMSE to fine)={fit_ref:.4g}")

    # Optional visuals
    show_image_grid(Y, image_size=image_size, title="Training targets (Y)")
    show_image_grid(xT, image_size=image_size, title="ODE samples @ t*")
    show_image_grid(ref, image_size=image_size, title="Reference ρ_t* samples")


if __name__ == "__main__":
    run_novelty_simple(
        n_target=200,
        n_samp=2048,
        image_size=32,
        method="rk2",
        grid="uniform",
        n_steps=2, # try changing to 1
        use_double=True,
        noise_std=0.0,
        thickness=2.0,
    )



# %%
