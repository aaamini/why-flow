from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ====== EVF dependencies from the why-flow repo ======
# The repo https://github.com/aaamini/why-flow contains evf.py with:
# - sample_rho_t_empirical(Y_train: Tensor, t: float, n: int) -> Tensor
# - EmpiricalVectorField(Y_train: Tensor) callable: v(t: Tensor, x: Tensor) -> Tensor
# - Integrator(field) with integrate(x0, t_grid, method, return_traj=False) -> Tensor
# - uniform_grid(steps: int, t1: float) -> Tensor
from evf import sample_rho_t_empirical, EmpiricalVectorField, Integrator, uniform_grid


# =========================
# Config
# =========================
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # Data
    data_root = "./data"
    download = False
    batch_size = 256
    num_workers = 0

    # Single digit
    digit = 0

    # VAE settings
    latent_dim = 16
    vae_ckpt = "checkpoints/vae_mnist_digit0_lat16.pt"  # path to your trained digit-0 VAE

    # EVF settings
    n_train_for_evf = 100       # number of training μ used as EVF seeds (Y_train)
    n_gen = 2000                # number of Euler-one-step generated samples
    t_euler = 0.3               # Euler one-step t

    # Histogram settings
    hist_bins = 50
    max_train_embed = None      # e.g., 5000 to subsample for speed
    max_test_embed = None       # e.g., 2000
    max_gen_embed = None        # e.g., 2000

    # IO
    out_dir = "outputs"

    # Optional small visualizations
    n_show = 8


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# Simple Conv VAE
# =========================
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28->14
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14->7
            nn.ReLU(inplace=True),
        )
        self.enc_out_dim = 32 * 7 * 7
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        self.fc_z = nn.Linear(latent_dim, self.enc_out_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7->14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14->28
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.enc_conv(x)
        h = torch.flatten(h, 1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_z(z)
        h = h.view(z.size(0), 32, 7, 7)
        return self.dec_conv(h)

    def forward(self, x: Tensor):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        xrec = self.decode(z)
        return xrec, mu, logvar


# =========================
# EVF generator wrapper
# =========================
class EVFGenerators:
    def __init__(self, Y_train: Tensor):
        assert Y_train.dim() == 2, "Y_train must be [N, D]"
        self.Y_train = Y_train
        self.device = Y_train.device
        self.dtype = Y_train.dtype

    @torch.no_grad()
    def euler_one_step(self, t: float, n_samp: int) -> Tensor:
        # Sample x_t ~ rho_t(Y_train)
        x_t = sample_rho_t_empirical(self.Y_train, float(t), int(n_samp))
        # Empirical vector field v(t, x)
        field = EmpiricalVectorField(self.Y_train.double())
        t_tensor = x_t.new_full((x_t.size(0), 1), float(t))
        v = field(t_tensor, x_t)
        # Euler backward one-step: x_{t-1} ≈ x_t + (1 - t) v(t, x_t)
        return (x_t + (1.0 - float(t)) * v).to(self.dtype)


# =========================
# Data helpers
# =========================
@torch.no_grad()
def get_mnist_digit_subset(digit: int, train: bool, cfg: Config) -> Subset:
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root=cfg.data_root, train=train, transform=tfm, download=cfg.download)
    indices = [i for i, (_, y) in enumerate(ds) if int(y) == int(digit)]
    return Subset(ds, indices)


@torch.no_grad()
def encode_dataset_to_mu(model: VAE, loader: DataLoader, device: str) -> Tuple[Tensor, Tensor]:
    """
    Returns:
      mu_all: [N, D] on CPU
      x_all:  [N, 1, 28, 28] on CPU
    """
    mus, imgs = [], []
    for x, _ in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        mus.append(mu.detach().cpu())
        imgs.append(x.detach().cpu())
    return torch.cat(mus, dim=0), torch.cat(imgs, dim=0)


@torch.no_grad()
def encode_mu(model: nn.Module, x: Tensor, device: str) -> Tensor:
    model.eval()
    mu, _ = model.encode(x.to(device))
    return mu.detach().cpu()


def maybe_subsample(t: Tensor, k: int | None) -> Tensor:
    if k is None or t.size(0) <= k:
        return t
    idx = torch.randperm(t.size(0))[:k]
    return t[idx]


@torch.no_grad()
def compute_nn_min_dist(queries: Tensor, refs: Tensor, chunk_q: int = 1024, chunk_r: int = 4096) -> Tensor:
    """
    Min Euclidean distance from each row of queries to the set refs, with batching.
    queries: [Nq, D] CPU
    refs:    [Nr, D] CPU
    returns: [Nq] CPU float32
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refs_d = refs.to(device)
    out = torch.empty(queries.size(0), dtype=torch.float32)
    q_ptr = 0
    while q_ptr < queries.size(0):
        q_end = min(q_ptr + chunk_q, queries.size(0))
        q_chunk = queries[q_ptr:q_end].to(device)  # [bq, D]
        min_d2 = torch.full((q_chunk.size(0),), float("inf"), device=device)
        r_ptr = 0
        while r_ptr < refs_d.size(0):
            r_end = min(r_ptr + chunk_r, refs_d.size(0))
            r_chunk = refs_d[r_ptr:r_end]  # [br, D]
            q2 = (q_chunk**2).sum(dim=1, keepdim=True)         # [bq,1]
            r2 = (r_chunk**2).sum(dim=1).unsqueeze(0)          # [1,br]
            cross = q_chunk @ r_chunk.t()                      # [bq,br]
            d2 = q2 + r2 - 2.0 * cross
            d2.clamp_(min=0.0)
            min_d2 = torch.minimum(min_d2, d2.min(dim=1).values)
            r_ptr = r_end
        out[q_ptr:q_end] = min_d2.sqrt().detach().cpu()
        q_ptr = q_end
    return out


def make_square_grid_count(n: int) -> Tuple[int, int]:
    rows = cols = int(math.sqrt(n))
    if rows * cols < n:
        cols += 1
        if rows * cols < n:
            rows += 1
    return rows, cols


# =========================
# Main
# =========================
def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)
    print("Using device:", cfg.device)

    # Load VAE
    vae = VAE(latent_dim=cfg.latent_dim).to(cfg.device)
    assert os.path.exists(cfg.vae_ckpt), f"VAE checkpoint not found: {cfg.vae_ckpt}"
    vae.load_state_dict(torch.load(cfg.vae_ckpt, map_location=cfg.device))
    vae.eval()

    # Prepare MNIST digit-0 subsets
    train_subset = get_mnist_digit_subset(cfg.digit, train=True, cfg=cfg)
    test_subset = get_mnist_digit_subset(cfg.digit, train=False, cfg=cfg)
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Encode train/test to μ
    with torch.no_grad():
        mu_train_all, x_train_all = encode_dataset_to_mu(vae, train_loader, cfg.device)
        mu_test_all, x_test_all = encode_dataset_to_mu(vae, test_loader, cfg.device)

    N_tr = mu_train_all.size(0)
    D = mu_train_all.size(1)
    N_te = mu_test_all.size(0)
    print(f"Encoded μ | train: {N_tr}x{D} | test: {N_te}x{D}")

    # Pick EVF seeds from training μ
    assert cfg.n_train_for_evf <= N_tr, "n_train_for_evf exceeds available training samples"
    perm = torch.randperm(N_tr)
    idx_train_evf = perm[:cfg.n_train_for_evf]
    mu_train_seeds = mu_train_all[idx_train_evf]  # [n_seed, D]
    print(f"Selected {cfg.n_train_for_evf} EVF seeds from training set.")

    # Build EVF generator and sample with Euler one step at t
    evf = EVFGenerators(mu_train_seeds.to(cfg.device))
    t_euler = float(cfg.t_euler)
    with torch.no_grad():
        z_gen = evf.euler_one_step(t=t_euler, n_samp=cfg.n_gen)  # in latent space of μ (D-dim)
    z_gen_cpu = z_gen.detach().cpu()
    print(f"Generated {z_gen_cpu.size(0)} latent samples via Euler one step at t={t_euler}.")

    # Decode generated latents to images, then re-encode to μ for fair comparison
    with torch.no_grad():
        x_gen = vae.decode(z_gen.to(cfg.device)).detach().cpu()
        mu_gen_all = encode_mu(vae, x_gen, cfg.device)

    # Optional: quick grids
    n_grid = min(64, x_gen.size(0))
    rows, cols = make_square_grid_count(n_grid)
    fig = plt.figure(figsize=(cols, rows))
    for i in range(n_grid):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(x_gen[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    t_tag = f"{t_euler:.2f}".replace(".", "")
    grid_path = os.path.join(cfg.out_dir, f"digit{cfg.digit}_gen_grid_t{t_tag}.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=160)
    plt.close()
    print(f"Saved generated grid to {grid_path}")

    # Optional: example triplets (generated vs nearest train/test in μ-space, shown as reconstructions)
    n_show = min(cfg.n_show, mu_gen_all.size(0))
    if n_show > 0:
        pick_idx = torch.linspace(0, mu_gen_all.size(0) - 1, steps=n_show).long()
        with torch.no_grad():
            dist_tr = torch.cdist(mu_gen_all[pick_idx], mu_train_all)  # [n_show, N_tr]
            nn_tr_idx = dist_tr.argmin(dim=1)
            dist_te = torch.cdist(mu_gen_all[pick_idx], mu_test_all)   # [n_show, N_te]
            nn_te_idx = dist_te.argmin(dim=1)

            x_tr_nn = x_train_all[nn_tr_idx]
            x_te_nn = x_test_all[nn_te_idx]

            # Reconstruct via μ to keep consistency
            x_gen_show = vae.decode(mu_gen_all[pick_idx].to(cfg.device)).detach().cpu()
            mu_tr, _ = vae.encode(x_tr_nn.to(cfg.device))
            x_tr_recon = vae.decode(mu_tr).detach().cpu()
            mu_te, _ = vae.encode(x_te_nn.to(cfg.device))
            x_te_recon = vae.decode(mu_te).detach().cpu()

        plt.figure(figsize=(6.5, 2.1 * n_show))
        for i in range(n_show):
            ax = plt.subplot(n_show, 3, i * 3 + 1)
            ax.imshow(x_gen_show[i, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title("generated (recon)", fontsize=9)
            ax.axis("off")

            ax = plt.subplot(n_show, 3, i * 3 + 2)
            ax.imshow(x_tr_recon[i, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title("nearest train (recon)", fontsize=9)
            ax.axis("off")

            ax = plt.subplot(n_show, 3, i * 3 + 3)
            ax.imshow(x_te_recon[i, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title("nearest test (recon)", fontsize=9)
            ax.axis("off")

        ex_path = os.path.join(cfg.out_dir, f"digit{cfg.digit}_examples_recon_t{t_tag}.png")
        plt.tight_layout()
        plt.savefig(ex_path, dpi=160)
        plt.close()
        print(f"Saved example triplets to {ex_path}")

    # Subsample for speed if desired
    mu_train_ref = maybe_subsample(mu_train_all, cfg.max_train_embed)
    mu_test_q = maybe_subsample(mu_test_all, cfg.max_test_embed)
    mu_gen_q = maybe_subsample(mu_gen_all, cfg.max_gen_embed)

    # Compute NN distances to training μ
    print("Computing NN distances (test→train, gen→train) in μ-space ...")
    nn_test_to_train = compute_nn_min_dist(mu_test_q, mu_train_ref) if mu_test_q.numel() > 0 else None
    nn_gen_to_train = compute_nn_min_dist(mu_gen_q, mu_train_ref)

    # Plot overlapped histograms
    plt.figure(figsize=(7.0, 4.5))
    if nn_test_to_train is not None:
        plt.hist(nn_test_to_train.numpy(), bins=cfg.hist_bins, alpha=0.6, density=True,
                 label="Test → Train (NN dist)", color="#1f77b4")
    plt.hist(nn_gen_to_train.numpy(), bins=cfg.hist_bins, alpha=0.6, density=True,
             label="Generated (Euler 1-step) → Train (NN dist)", color="#ff7f0e")
    plt.xlabel("Nearest neighbor distance in encoder μ-space")
    plt.ylabel("Density")
    plt.title(f"Digit {cfg.digit} | NN distance histograms (t={cfg.t_euler})")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(cfg.out_dir, f"digit{cfg.digit}_nn_hist_t{t_tag}.png")
    plt.savefig(hist_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved NN distance histograms to {hist_path}")

    # Optional numeric summaries
    if nn_test_to_train is not None:
        gen_med = float(nn_gen_to_train.median())
        test_med = float(nn_test_to_train.median())
        print(f"Median NN dist | Generated: {gen_med:.4f} | Test: {test_med:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()