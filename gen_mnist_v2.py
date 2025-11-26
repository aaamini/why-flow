from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ====== EVF dependencies (ensure evf.py is importable in the same folder) ======
# evf.py must define:
# - sample_rho_t_empirical(Y_train: Tensor, t: float, n: int) -> Tensor
# - EmpiricalVectorField(Y_train: Tensor) callable: v(t: Tensor, x: Tensor) -> Tensor
# - Integrator(field) with integrate(x0, t_grid, method, return_traj=False) -> Tensor
# - uniform_grid(steps: int, t1: float) -> Tensor
from evf import sample_rho_t_empirical, EmpiricalVectorField, Integrator, uniform_grid

GeneratorFn = Callable[[Any, int], Tensor]


@dataclass
class MethodSpec:
    name: str
    params: Sequence[Any]
    generator: GeneratorFn
    n_samples: int


class EVFGenerators:
    def __init__(self, Y_train: Tensor):
        assert Y_train.dim() == 2, "Y_train must be [N, D]"
        self.Y_train = Y_train
        self.device = Y_train.device
        self.dtype = Y_train.dtype

    @torch.no_grad()
    def exact_xt(self, t: float, n_samp: int) -> Tensor:
        return sample_rho_t_empirical(self.Y_train, float(t), int(n_samp))

    @torch.no_grad()
    def euler_one_step(self, t: float, n_samp: int) -> Tensor:
        # Sample x_t ~ rho_t(Y_train)
        x_t = sample_rho_t_empirical(self.Y_train, float(t), int(n_samp))
        # Empirical vector field
        field = EmpiricalVectorField(self.Y_train.double())
        t_tensor = x_t.new_full((x_t.size(0), 1), float(t))
        v = field(t_tensor, x_t)
        # Euler backward one step: x_{t-1} ≈ x_t + (1 - t) v(t, x_t)
        return (x_t + (1.0 - float(t)) * v).to(self.dtype)

    @torch.no_grad()
    def dode(self, steps: int, n_samp: int, *, t1: float = 1.0, method: str = "rk2") -> Tensor:
        field = EmpiricalVectorField(self.Y_train.double())
        integ = Integrator(field)
        t_grid = uniform_grid(steps, t1=float(t1))
        x0 = torch.randn(n_samp, self.Y_train.size(1), device=self.device, dtype=self.dtype)
        xT = integ.integrate(x0, t_grid, method=method, return_traj=False)
        return xT.to(self.dtype)


# ====== Config ======
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    data_root = "./data"
    download = False  # set True if MNIST is not present locally
    batch_size = 256
    num_workers = 0

    # single digit
    digit = 0
    latent_dim = 10  # must match your VAE
    vae_ckpt = "checkpoints/vae_mnist_digit0_lat10.pt"  # TODO: set your checkpoint path

    # how many training samples to use as EVF seeds (Y_train)
    n_train_for_evf = 100

    # how many to generate
    n_gen = 1000

    # Global Euler step time parameter; change here to propagate everywhere
    t_euler = 0.5

    # how many triplets to show
    n_show = 8

    out_dir = "outputs"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ====== Simple VAE (replace with your own if needed) ======
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
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
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


@torch.no_grad()
def get_mnist_digit_subset(digit: int, train: bool, cfg: Config):
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root=cfg.data_root, train=train, transform=tfm, download=cfg.download)
    indices = [i for i, (_, y) in enumerate(ds) if int(y) == int(digit)]
    return Subset(ds, indices)


@torch.no_grad()
def encode_dataset_to_latents(model: VAE, loader: DataLoader, device: str) -> Tuple[Tensor, Tensor]:
    zs, imgs = [], []
    for x, _ in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        zs.append(mu.detach().cpu())
        imgs.append(x.detach().cpu())
    return torch.cat(zs, dim=0), torch.cat(imgs, dim=0)


def pairwise_min_dist(A: Tensor, B: Tensor, chunk: int = 1000) -> Tensor:
    # return min euclidean distance from each row of A to set B
    mins = []
    for i in range(0, A.size(0), chunk):
        Ai = A[i:i + chunk]
        dist = torch.cdist(Ai, B)
        mins.append(dist.min(dim=1).values)
    return torch.cat(mins, dim=0)


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

    # Prepare MNIST digit subset
    train_subset = get_mnist_digit_subset(cfg.digit, train=True, cfg=cfg)
    test_subset = get_mnist_digit_subset(cfg.digit, train=False, cfg=cfg)
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Encode to latents
    with torch.no_grad():
        z_train_all, x_train_all = encode_dataset_to_latents(vae, train_loader, cfg.device)  # include seeds
        z_test, x_test = encode_dataset_to_latents(vae, test_loader, cfg.device)

    N_tr = z_train_all.size(0)
    D = z_train_all.size(1)
    N_te = z_test.size(0)
    print(f"Encoded latents | train: {N_tr}x{D} | test: {N_te}x{D}")

    # Choose EVF seeds from training latents
    assert cfg.n_train_for_evf <= N_tr
    perm = torch.randperm(N_tr)
    idx_train_evf = perm[:cfg.n_train_for_evf]
    z_train_evf = z_train_all[idx_train_evf]  # [n_seed, D]
    print(f"Selected {cfg.n_train_for_evf} EVF seeds from the training set.")

    # Build EVF generator
    evf = EVFGenerators(z_train_evf.to(cfg.device))

    # Generate with Euler one-step at t_euler
    t_euler = float(cfg.t_euler)  # single source of truth
    with torch.no_grad():
        z_gen = evf.euler_one_step(t=t_euler, n_samp=cfg.n_gen)
    z_gen_cpu = z_gen.detach().cpu()
    print(f"Generated {z_gen_cpu.size(0)} latent samples via Euler one step at t={t_euler}.")

    # Distances to full training (including seeds) and test
    with torch.no_grad():
        d_train_min = pairwise_min_dist(z_gen_cpu, z_train_all)
        d_test_min = pairwise_min_dist(z_gen_cpu, z_test)

    # Scatter plot: nearest distance to train (x) vs test (y)
    plt.figure(figsize=(6, 5))
    plt.scatter(d_train_min.numpy(), d_test_min.numpy(), s=10, alpha=0.6)
    plt.xlabel("Nearest distance to training (including seeds)")
    plt.ylabel("Nearest distance to test")
    plt.title(f"EVF (digit {cfg.digit}) Euler one-step at t={t_euler}")
    plt.grid(True, alpha=0.2)
    t_tag = f"{t_euler:.2f}".replace(".", "")
    scatter_path = os.path.join(cfg.out_dir, f"evf_digit{cfg.digit}_scatter_t{t_tag}.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=160)
    plt.close()
    print(f"Saved scatter to {scatter_path}")

    # Decode generated latents, save grid
    with torch.no_grad():
        x_gen = vae.decode(z_gen.to(cfg.device)).detach().cpu()

    n_grid = min(64, x_gen.size(0))
    rows = cols = int(math.sqrt(n_grid)) if int(math.sqrt(n_grid))**2 == n_grid else int(math.floor(math.sqrt(n_grid)))
    if rows * cols < n_grid:
        cols = rows + 1
        if rows * cols < n_grid:
            rows += 1

    fig = plt.figure(figsize=(cols, rows))
    for i in range(n_grid):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(x_gen[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    grid_path = os.path.join(cfg.out_dir, f"evf_digit{cfg.digit}_gen_grid_t{t_tag}.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=160)
    plt.close()
    print(f"Saved generated grid to {grid_path}")

    # Triplets: gen / nearest train / nearest test
    n_show = min(cfg.n_show, z_gen_cpu.size(0))
    pick_idx = torch.linspace(0, z_gen_cpu.size(0) - 1, steps=n_show).long() if n_show > 1 else torch.tensor([0])

    with torch.no_grad():
        dist_tr = torch.cdist(z_gen_cpu[pick_idx], z_train_all)  # [n_show, N_tr]
        nn_tr_idx = dist_tr.argmin(dim=1)                        # nearest train (full set)
        dist_te = torch.cdist(z_gen_cpu[pick_idx], z_test)       # [n_show, N_te]
        nn_te_idx = dist_te.argmin(dim=1)

    with torch.no_grad():
        x_gen_show = vae.decode(z_gen[pick_idx].to(cfg.device)).detach().cpu()
        x_tr_show = x_train_all[nn_tr_idx]
        x_te_show = x_test[nn_te_idx]

    plt.figure(figsize=(6.5, 2.1 * n_show))
    for i in range(n_show):
        # generated
        ax = plt.subplot(n_show, 3, i * 3 + 1)
        ax.imshow(x_gen_show[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("generated", fontsize=9)
        ax.axis("off")

        # nearest train
        ax = plt.subplot(n_show, 3, i * 3 + 2)
        ax.imshow(x_tr_show[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("nearest train", fontsize=9)
        ax.axis("off")

        # nearest test
        ax = plt.subplot(n_show, 3, i * 3 + 3)
        ax.imshow(x_te_show[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("nearest test", fontsize=9)
        ax.axis("off")

    ex_path = os.path.join(cfg.out_dir, f"evf_digit{cfg.digit}_examples_t{t_tag}.png")
    plt.tight_layout()
    plt.savefig(ex_path, dpi=160)
    plt.close()
    print(f"Saved example triplets to {ex_path}")

    # Optional: what fraction of nearest-train belong to the EVF seed subset
    in_seed = torch.isin(nn_tr_idx.cpu(), idx_train_evf.cpu()).float().mean().item()
    print(f"Fraction of picked examples whose nearest training is among EVF seeds: {in_seed:.3f}")

    print("Done.")


if __name__ == "__main__":
    main()