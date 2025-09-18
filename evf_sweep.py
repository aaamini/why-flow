#%%
import os
import json
import math
from typing import List, Dict, Any

import torch
from torch import Tensor
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evf import EmpiricalVectorField, sample_rho_t_empirical, Integrator, uniform_grid
from metrics import nn_rmse_to_targets
from circle_images import generate_circle_images

import torchvision
from torchvision import models
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def euler_one_step(Y: Tensor, x_t: Tensor, t: float) -> Tensor:
    field = EmpiricalVectorField(Y)
    t_tensor = x_t.new_full((x_t.size(0), 1), float(t))
    v = field(t_tensor, x_t)
    return x_t + (1.0 - float(t)) * v


@torch.no_grad()
def dode(Y: Tensor, n_steps: int, t: float, n_samp: int, method: str) -> Tensor:
    field = EmpiricalVectorField(Y.double())
    integ = Integrator(field)
    t_grid = uniform_grid(n_steps, t1=t)
    x0 = torch.randn(n_samp, Y.size(1), device=Y.device, dtype=Y.dtype)
    xT = integ.integrate(x0, t_grid, method=method, return_traj=False)
    return xT.float()


class LinearDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    @torch.no_grad()
    def fit_ridge(self, X_latent: Tensor, Y_img: Tensor, ridge_lambda: float = 1e-3):
        X = X_latent
        Y = Y_img
        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        Xc = X - X_mean
        Yc = Y - Y_mean
        d = Xc.size(1)
        XtX = Xc.T @ Xc
        reg = ridge_lambda * torch.eye(d, device=X.device, dtype=X.dtype)
        W_t = torch.linalg.solve(XtX + reg, Xc.T @ Yc)
        W = W_t.T
        b = (Y_mean - X_mean @ W_t).squeeze(0)
        self.linear.weight.copy_(W)
        self.linear.bias.copy_(b)

    def forward(self, z: Tensor) -> Tensor:
        return self.linear(z)


@torch.no_grad()
def build_encoder(image_size: int, encoder_batch_size: int):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    resnet.eval()
    resnet.fc = nn.Identity()

    @torch.no_grad()
    def encoder(flat_imgs: Tensor, batch_size: int = None) -> Tensor:
        if batch_size is None:
            batch_size = encoder_batch_size
        outputs = []
        N = flat_imgs.size(0)
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            x = flat_imgs[start:end].to(device)
            n = x.size(0)
            x = x.view(n, 1, image_size, image_size)
            x = x.expand(n, 3, image_size, image_size)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=x.dtype).view(1, 3, 1, 1)
            x = (x - mean) / std
            outputs.append(resnet(x).detach())
        return torch.cat(outputs, dim=0)

    return encoder


def generate_ground_truth(n_samp: int, image_size: int, noise_std: float, thickness: float, dtype: torch.dtype) -> Tensor:
    return generate_circle_images(
        n_samp,
        image_size=image_size,
        noise_std=noise_std,
        thickness=thickness,
        device=device,
        dtype=dtype,
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    # Experiment config
    image_size = 32
    noise_std = 0.0
    thickness = 2.0
    dtype = torch.float32
    encoder_batch_size = 128
    n_samp = 2048

    # Sweep settings
    n_target_list = [300, 600, 1200]
    t_values = np.linspace(0.5, 0.95, 10).tolist()
    n_steps_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    method = "rk2"

    # Outputs
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    plots_dir = os.path.join(results_dir, "plots")
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    encoder = build_encoder(image_size=image_size, encoder_batch_size=encoder_batch_size)

    all_rows: List[Dict[str, Any]] = []

    for n_target in n_target_list:
        with torch.no_grad():
            Y_img = generate_ground_truth(n_target, image_size, noise_std, thickness, dtype)
            D_img = Y_img.size(1)

            # Latent targets
            zY = encoder(Y_img)
            D_lat = zY.size(1)

            # Dense cover of latent manifold and true samples (once per n_target)
            Y_fine = generate_ground_truth(max(n_samp, 8000), image_size, noise_std, thickness, dtype)
            zY_fine = encoder(Y_fine)

            X_true = generate_ground_truth(n_samp, image_size, noise_std, thickness, dtype)
            X_true_lat = encoder(X_true)

            # Methods depending on t (x_t and Euler)
            for t in t_values:
                x_t_lat = sample_rho_t_empirical(zY, float(t), n_samp)
                novelty_xt = nn_rmse_to_targets(x_t_lat, zY)
                fit_xt = nn_rmse_to_targets(x_t_lat, zY_fine)
                all_rows.append({
                    "method": "x_t",
                    "n_target": int(n_target),
                    "t": float(t),
                    "n_steps": None,
                    "novelty": float(novelty_xt),
                    "fit": float(fit_xt),
                    "latent_dim": int(D_lat),
                    "image_dim": int(D_img),
                })

                x1_euler_lat = euler_one_step(zY, x_t_lat, float(t))
                novelty_euler = nn_rmse_to_targets(x1_euler_lat, zY)
                fit_euler = nn_rmse_to_targets(x1_euler_lat, zY_fine)
                all_rows.append({
                    "method": "euler",
                    "n_target": int(n_target),
                    "t": float(t),
                    "n_steps": None,
                    "novelty": float(novelty_euler),
                    "fit": float(fit_euler),
                    "latent_dim": int(D_lat),
                    "image_dim": int(D_img),
                })

            # Dode varying n_steps (t fixed at 1.0)
            for n_steps in n_steps_values:
                x1_dode_lat = dode(zY, n_steps=int(n_steps), t=1.0, n_samp=n_samp, method=method)
                novelty_dode = nn_rmse_to_targets(x1_dode_lat, zY)
                fit_dode = nn_rmse_to_targets(x1_dode_lat, zY_fine)
                all_rows.append({
                    "method": "dode",
                    "n_target": int(n_target),
                    "t": 1.0,
                    "n_steps": int(n_steps),
                    "novelty": float(novelty_dode),
                    "fit": float(fit_dode),
                    "latent_dim": int(D_lat),
                    "image_dim": int(D_img),
                })

    # Save results
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(results_dir, "evf_sweep_results.csv")
    json_path = os.path.join(results_dir, "evf_sweep_results.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f)

    # Plot novelty vs fit per n_target
    for n_target in sorted(df["n_target"].unique()):
        sub = df[df["n_target"] == n_target]
        plt.figure(figsize=(6, 5))
        for method_name, marker in [("x_t", "o"), ("euler", "s"), ("dode", "^")]:
            part = sub[sub["method"] == method_name]
            x = part["novelty"].to_numpy()
            y = part["fit"].to_numpy()
            eps = 1e-8
            x = np.clip(x, eps, None)
            y = np.clip(y, eps, None)
            plt.scatter(x, y, label=method_name, marker=marker)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Novelty (RMSE to training latents)")
        plt.ylabel("Fit (RMSE to dense latents)")
        plt.title(f"Novelty vs Fit (n_target={n_target})")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plot_path = os.path.join(plots_dir, f"evf_sweep_n{n_target}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"Saved results to: {csv_path} and {json_path}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()



# %%
