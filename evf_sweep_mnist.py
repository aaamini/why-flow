# train_and_sweep_mnist.py
#%%
import os
import json
from typing import List, Dict, Any, Tuple

import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader, TensorDataset, Subset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

# Project imports
from vae import SimpleVAE, vae_loss
from evf import EmpiricalVectorField, sample_rho_t_empirical, Integrator, uniform_grid
from metrics import nn_rmse_to_targets

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
# Data utilities (MNIST)
# -----------------------------
def load_mnist_flat(image_size: int = 28, dtype=torch.float32) -> Tuple[Tensor, Tensor]:
    """
    Loads MNIST train split (60k), transforms to [0,1] tensors,
    returns flat tensors X of shape (N, image_size*image_size).
    Labels are returned for completeness but unused here.
    """
    assert image_size == 28, "MNIST image size is 28x28."
    transform = transforms.ToTensor()  # produces (C=1, 28, 28) in [0,1]
    ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    # Stack all into memory as a single flat tensor
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=2)
    X_list, y_list = [], []
    for imgs, labels in loader:
        # imgs: (B,1,28,28) -> flatten to (B, 784)
        X_list.append(imgs.view(imgs.size(0), -1).to(dtype))
        y_list.append(labels)
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y


def split_for_experiment(
    X: Tensor,
    num_train_images: int,
    n_target_list: List[int],
    n_fine_cover: int,
    n_eval: int,
    seed: int = 123,
) -> Dict[int, Dict[str, Tensor]]:
    """
    Create disjoint subsets from MNIST for each n_target in n_target_list:
      - train set for VAE training (shared across experiments)
      - Y (targets) of size n_target
      - Y_fine (dense cover) of size n_fine_cover
      - X_true (evaluation samples) of size n_eval
    Ensures no overlap between train, Y, Y_fine, X_true.
    Returns a dict keyed by n_target with tensors: Y_img, Y_fine, X_true.
    Also returns train_img (shared) separately.
    """
    N = X.size(0)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)

    # Reserve the first chunk for VAE training
    assert num_train_images < N, "num_train_images must be < total MNIST train size (60000)."
    train_idx = perm[:num_train_images]
    cursor = num_train_images

    # We’ll allocate the remaining pool per n_target independently (without overlaps within each experiment).
    # To avoid running out, keep sizes conservative.
    results: Dict[int, Dict[str, Tensor]] = {}

    # For each experiment, sample disjoint sets from the remaining pool, but independently per n_target
    # (train set is shared and disjoint from all).
    pool = perm[cursor:]  # remaining indices
    pool_N = pool.size

    # Validate that we have enough for the largest requested split per n_target
    max_need = max(n_target_list) + n_fine_cover + n_eval
    assert max_need <= pool_N, (
        f"Not enough MNIST examples for splits. Need at least {max_need} from pool, have {pool_N}. "
        "Reduce n_fine_cover/n_eval or use fewer n_target or smaller num_train_images."
    )

    for n_target in n_target_list:
        # Randomly sample disjoint sets from pool for this experiment
        pick = rng.choice(pool, size=(n_target + n_fine_cover + n_eval), replace=False)
        y_idx = pick[:n_target]
        y_fine_idx = pick[n_target:n_target + n_fine_cover]
        x_true_idx = pick[n_target + n_fine_cover:]

        results[n_target] = dict(
            Y_img=X[y_idx],
            Y_fine=X[y_fine_idx],
            X_true=X[x_true_idx],
        )

    return dict(
        train_img=X[train_idx],
        per_target=results,
    )


# -----------------------------
# VAE training (using vae.py)
# -----------------------------
def train_simple_vae(
    image_size=28,
    latent_dim=8,
    num_train_images=5000,
    batch_size=128,
    epochs=50,
    lr=1e-3,
    dtype=torch.float32,
    device=device,
    train_img: Tensor = None,
) -> SimpleVAE:
    vae = SimpleVAE(image_size=image_size, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()

    assert train_img is not None and train_img.ndim == 2, "train_img must be flat (N, 784) tensor."
    # Subsample to exactly num_train_images (if more were provided)
    if train_img.size(0) > num_train_images:
        idx = torch.randperm(train_img.size(0))[:num_train_images]
        train_img = train_img[idx]

    loader = DataLoader(TensorDataset(train_img), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for (data,) in loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = vae(data)
            loss = vae_loss(recon, data, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
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
# Main: Train or load VAE, then EVF sweep on MNIST
# -----------------------------
def main():
    # Core config
    image_size = 28
    latent_dim = 8
    dtype = torch.float32
    encoder_batch_size = 256

    # Sizes for each phase
    num_train_images = 5000   # VAE training set size
    n_target_list = [300, 600, 1200]  # number of target images for EVF field
    n_fine_cover = 8000       # dense cover for "fit"
    n_eval = 2048             # X_true sample count for eval/sampling
    n_samp = n_eval           # keep same for consistency

    # VAE training config
    batch_size = 128
    epochs = 50
    lr = 1e-3

    # EVF integration sweep
    t_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
    n_steps_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    method = "rk2"

    # Paths
    here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    results_dir = os.path.join(here, "results_mnist")
    plots_dir = os.path.join(results_dir, "plots")
    ckpt_dir = os.path.join(here, "checkpoints")
    ensure_dir(results_dir)
    ensure_dir(plots_dir)
    ensure_dir(ckpt_dir)

    ckpt_path = os.path.join(ckpt_dir, f"simple_vae_mnist_{image_size}px_lat{latent_dim}.pt")
    skip_if_ckpt_exists = True  # set False to always retrain

    # Load MNIST (train split)
    print("Loading MNIST...")
    X_all, y_all = load_mnist_flat(image_size=image_size, dtype=dtype)  # X_all in [0,1], shape (60000, 784)

    # Create disjoint splits for our experiment
    splits = split_for_experiment(
        X_all,
        num_train_images=num_train_images,
        n_target_list=n_target_list,
        n_fine_cover=n_fine_cover,
        n_eval=n_eval,
        seed=123,
    )
    train_img = splits["train_img"]
    per_target = splits["per_target"]

    # Train or load VAE
    if skip_if_ckpt_exists and os.path.isfile(ckpt_path):
        print(f"Loading VAE from checkpoint: {ckpt_path}")
        vae = SimpleVAE(image_size=image_size, latent_dim=latent_dim).to(device)
        # PyTorch security guidance (untrusted models)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        vae.load_state_dict(state)
        vae.eval()
    else:
        print("Training SimpleVAE on MNIST...")
        vae = train_simple_vae(
            image_size=image_size,
            latent_dim=latent_dim,
            num_train_images=num_train_images,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            dtype=dtype,
            device=device,
            train_img=train_img.to(device),
        )
        torch.save(vae.state_dict(), ckpt_path)
        print(f"Saved VAE checkpoint to: {ckpt_path}")

    # Sanity check recon loss
    with torch.no_grad():
        # take a small batch from training images
        x_val = train_img[:64].to(device)
        recon, mu, log_var = vae(x_val)
        bce = torch.nn.functional.binary_cross_entropy(recon, x_val, reduction="mean")
    print(f"Sanity recon BCE (mean): {float(bce):.6f}")

    # Build encoder from trained VAE
    encoder = build_vae_encoder(vae, encoder_batch_size=encoder_batch_size)

    # EVF sweep
    all_rows: List[Dict[str, Any]] = []

    for n_target in n_target_list:
        with torch.no_grad():
            # Training/target/fine/eval images and latents for this n_target
            Y_img = per_target[n_target]["Y_img"].to(device)
            Y_fine = per_target[n_target]["Y_fine"].to(device)
            X_true = per_target[n_target]["X_true"].to(device)

            D_img = Y_img.size(1)
            zY = encoder(Y_img)
            D_lat = zY.size(1)

            zY_fine = encoder(Y_fine)
            X_true_lat = encoder(X_true)

            # x_t and Euler over t
            for t in t_values:
                x_t_lat = sample_rho_t_empirical(zY, float(t), n_samp)
                novelty_xt = nn_rmse_to_targets(x_t_lat, zY)
                fit_xt = nn_rmse_to_targets(x_t_lat, zY_fine)
                all_rows.append({
                    "dataset": "MNIST",
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
                    "dataset": "MNIST",
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
                    "dataset": "MNIST",
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
    csv_path = os.path.join(results_dir, "evf_sweep_results_mnist.csv")
    json_path = os.path.join(results_dir, "evf_sweep_results_mnist.json")
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
        plt.title(f"MNIST: Novelty vs Fit (n_target={n_target})")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plot_path = os.path.join(plots_dir, f"evf_sweep_mnist_n{n_target}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"Saved results to: {csv_path} and {json_path}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()

# %%