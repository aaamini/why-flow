# train_and_sweep.py
#%%
import os
import json
from typing import List, Dict, Any

import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project imports
from vae import SimpleVAE, vae_loss
from evf import EmpiricalVectorField, sample_rho_t_empirical, Integrator, uniform_grid
from metrics import nn_rmse_to_targets
from circle_images import generate_circle_images

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# EVF helpers
# -----------------------------
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

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# VAE training (using vae.py)
# -----------------------------
def train_simple_vae(
    image_size=32,
    latent_dim=32,
    num_train_images=5000,
    batch_size=64,
    epochs=200,
    lr=1e-3,
    noise_std=0.0,
    radius_range=(4, 12),  # add explicit default to match circle_images if needed
    thickness=2.0,
    dtype=torch.float32,
    device=device,
) -> SimpleVAE:
    vae = SimpleVAE(image_size=image_size, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()

    # Generate training data (keyword args to avoid positional mismatches)
    train_data_flat = generate_circle_images(
        num_images=num_train_images,
        image_size=image_size,
        noise_std=noise_std,
        radius_range=radius_range,
        thickness=thickness,
        device=device,
        dtype=dtype,
    )
    loader = DataLoader(TensorDataset(train_data_flat), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for (data,) in loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = vae(data)
            loss = vae_loss(recon, data, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader.dataset)
        print(f"[VAE] epoch {epoch+1}/{epochs} loss {avg:.6f}")

    vae.eval()
    return vae

@torch.no_grad()
def build_vae_encoder(vae: SimpleVAE, encoder_batch_size: int):
    def encoder(flat_imgs: Tensor, batch_size: int = None) -> Tensor:
        if batch_size is None:
            batch_size = encoder_batch_size
        outputs = []
        N = flat_imgs.size(0)
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            x = flat_imgs[start:end].to(device)
            mu, log_var = vae.encode(x)  # SimpleVAE.encode handles flat or (N,1,H,W)
            z = mu.view(mu.size(0), -1)
            outputs.append(z.detach())
        return torch.cat(outputs, dim=0)
    return encoder

# -----------------------------
# Main: Train or load VAE, then EVF sweep
# -----------------------------
def main():
    # Core config
    image_size = 32
    latent_dim = 32
    noise_std = 0.0
    radius_range = (4, 12)  # ensure a tuple for circle generator
    thickness = 2.0
    dtype = torch.float32
    encoder_batch_size = 128
    n_samp = 2048

    # VAE training config
    num_train_images = 5000
    batch_size = 64
    epochs = 200         # increase if needed (your original script had 1000)
    lr = 1e-3

    # Sweep settings
    n_target_list = [10, 20, 50, 100, 300, 600, 1200]
    t_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
    n_steps_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    method = "rk2"

    # Paths
    here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    results_dir = os.path.join(here, "results")
    plots_dir = os.path.join(results_dir, "plots")
    ckpt_dir = os.path.join(here, "checkpoints")
    ensure_dir(results_dir)
    ensure_dir(plots_dir)
    ensure_dir(ckpt_dir)

    ckpt_path = os.path.join(ckpt_dir, f"simple_vae_{image_size}px_lat{latent_dim}.pt")
    skip_if_ckpt_exists = True  # set False to always retrain

    # Train or load VAE
    if skip_if_ckpt_exists and os.path.isfile(ckpt_path):
        print(f"Loading VAE from checkpoint: {ckpt_path}")
        vae = SimpleVAE(image_size=image_size, latent_dim=latent_dim).to(device)
        vae.load_state_dict(torch.load(ckpt_path, map_location=device))
        vae.eval()
    else:
        print("Training SimpleVAE...")
        vae = train_simple_vae(
            image_size=image_size,
            latent_dim=latent_dim,
            num_train_images=num_train_images,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            noise_std=noise_std,
            radius_range=radius_range,
            thickness=thickness,
            dtype=dtype,
            device=device,
        )
        torch.save(vae.state_dict(), ckpt_path)
        print(f"Saved VAE checkpoint to: {ckpt_path}")

    # Sanity check recon loss with keyword args
    with torch.no_grad():
        x_val = generate_circle_images(
            num_images=64,
            image_size=image_size,
            noise_std=noise_std,
            radius_range=radius_range,
            thickness=thickness,
            device=device,
            dtype=dtype,
        )
        recon, mu, log_var = vae(x_val)
        bce = torch.nn.functional.binary_cross_entropy(recon, x_val, reduction="mean")
    print(f"Sanity recon BCE (mean): {float(bce):.6f}")

    # Build encoder from trained VAE
    encoder = build_vae_encoder(vae, encoder_batch_size=encoder_batch_size)

    # EVF sweep
    all_rows: List[Dict[str, Any]] = []

    for n_target in n_target_list:
        with torch.no_grad():
            # Training images and latents
            Y_img = generate_circle_images(
                num_images=n_target,
                image_size=image_size,
                noise_std=noise_std,
                radius_range=radius_range,
                thickness=thickness,
                device=device,
                dtype=dtype,
            )
            D_img = Y_img.size(1)
            zY = encoder(Y_img)
            D_lat = zY.size(1)

            # Dense cover and baseline latents
            Y_fine = generate_circle_images(
                num_images=max(n_samp, 8000),
                image_size=image_size,
                noise_std=noise_std,
                radius_range=radius_range,
                thickness=thickness,
                device=device,
                dtype=dtype,
            )
            zY_fine = encoder(Y_fine)

            X_true = generate_circle_images(
                num_images=n_samp,
                image_size=image_size,
                noise_std=noise_std,
                radius_range=radius_range,
                thickness=thickness,
                device=device,
                dtype=dtype,
            )
            X_true_lat = encoder(X_true)

            # x_t and Euler over t
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

            # DODE varying n_steps (t=1.0)
            for n_steps in n_steps_values:
                x1_dode_lat = dode(Y=zY, n_steps=int(n_steps), t=1.0, n_samp=n_samp, method=method)
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