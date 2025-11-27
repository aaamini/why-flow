import os
import math
import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # Data
    data_root = "./data"
    download = False     # Set to True if you need to auto-download MNIST
    img_size = 28
    channels = 1

    # Per-class training cap
    train_cap_per_class = 6000  # MNIST has about 5923/6742 (train/test) per class; 6000 covers all of class 0

    # Training
    batch_size = 128
    epochs = 300
    lr = 1e-3
    latent_dim = 16
    beta = 1.0                 # beta-VAE coefficient (1.0 = standard VAE)
    use_early_stop = False
    patience = 10

    # IO
    ckpt_dir = "checkpoints"
    out_dir = "outputs"

    # Visualization
    viz_batch = 8

    # Num workers
    num_workers = 0


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


# =========================
# Model: Small Conv VAE
# =========================
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 14 -> 7
            nn.ReLU(inplace=True),
        )
        self.enc_out_dim = 32 * 7 * 7
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        self.fc_z = nn.Linear(latent_dim, self.enc_out_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 14 -> 28
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_z(z)
        h = h.view(z.size(0), 32, 7, 7)
        return self.dec_conv(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


# =========================
# Loss & Metrics
# =========================
def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    # Reconstruction loss: BCE over [0,1]
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # KL divergence
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Average per batch
    return (recon_loss + beta * kl) / x.size(0)


@torch.no_grad()
def mse_metric(x, y):
    # Per-pixel MSE (mean over all dims)
    return F.mse_loss(y, x, reduction="mean").item()


@torch.no_grad()
def psnr_metric(x, y, data_range=1.0, eps=1e-10):
    mse = F.mse_loss(y, x, reduction="mean").item()
    mse = max(mse, eps)
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


@torch.no_grad()
def ssim_metric(x, y, data_range=1.0, K1=0.01, K2=0.03, window_size=7):
    # Simple SSIM for single-channel images
    pad = window_size // 2
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    w = torch.ones((1, 1, window_size, window_size), device=x.device) / (window_size * window_size)
    mu_x = F.conv2d(x, w, padding=pad)
    mu_y = F.conv2d(y, w, padding=pad)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, w, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return ssim_map.mean().item()


# =========================
# Data helpers
# =========================
def build_per_class_indices(dataset, cap_per_class: int) -> Dict[int, List[int]]:
    indices = {d: [] for d in range(10)}
    # One pass to collect up to cap_per_class samples per label
    for i, (_, y) in enumerate(dataset):
        if len(indices[y]) < cap_per_class:
            indices[y].append(i)
    return indices


# =========================
# Train / Eval loop
# =========================
def train_one_class(digit: int, train_loader: DataLoader, model: VAE, cfg: Config):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_state = None
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(cfg.epochs):
        total, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(cfg.device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, beta=cfg.beta)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

        epoch_loss = total / max(n, 1)
        print(f"  Epoch {epoch+1:03d} | loss {epoch_loss:.4f}")

        # Early stopping on training loss (simple heuristic)
        if cfg.use_early_stop:
            if epoch_loss + 1e-6 < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    print(f"  Early stopping at epoch {epoch+1} (no improve {cfg.patience}).")
                    break

    if cfg.use_early_stop and best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def evaluate_one_class(digit: int, test_loader: DataLoader, model: VAE, cfg: Config):
    model.eval()
    mse_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    n = 0
    for x, _ in test_loader:
        x = x.to(cfg.device)
        recon, _, _ = model(x)
        # Aggregate
        mse_sum += F.mse_loss(recon, x, reduction="sum").item()
        psnr_sum += psnr_metric(x, recon) * x.size(0)
        ssim_sum += ssim_metric(x, recon) * x.size(0)
        n += x.size(0)
    mse_mean = mse_sum / (n * cfg.img_size * cfg.img_size)
    psnr_mean = psnr_sum / n
    ssim_mean = ssim_sum / n
    print(f"Metrics(d={digit}) | MSE {mse_mean:.6f} | PSNR {psnr_mean:.2f} dB | SSIM {ssim_mean:.4f}")
    return mse_mean, psnr_mean, ssim_mean


@torch.no_grad()
def visualize_one_class(digit: int, test_subset: Subset, model: VAE, cfg: Config):
    model.eval()
    loader = DataLoader(test_subset, batch_size=cfg.viz_batch, shuffle=True, num_workers=cfg.num_workers)
    x, _ = next(iter(loader))
    x = x.to(cfg.device)
    recon, _, _ = model(x)
    x = x.detach().cpu()
    recon = recon.detach().cpu()

    n = x.size(0)
    plt.figure(figsize=(8, 2.7))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        if i == 0:
            plt.title(f"Digit {digit} - Orig")
        plt.subplot(2, n, n + i + 1)
        plt.imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        if i == 0:
            plt.title("Recon")
    plt.tight_layout()
    path = os.path.join(cfg.out_dir, f"digit{digit}_recon_grid.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved grid to {path}")


# =========================
# Main
# =========================
def main():
    cfg = Config()
    print("Using device:", cfg.device)
    set_seed(cfg.seed)
    ensure_dirs(cfg.ckpt_dir, cfg.out_dir)

    tfm = transforms.ToTensor()
    train_full = datasets.MNIST(root=cfg.data_root, train=True, transform=tfm, download=cfg.download)
    test_full = datasets.MNIST(root=cfg.data_root, train=False, transform=tfm, download=cfg.download)

    # Build per-class indices (cap for train; collect all for test)
    train_indices = build_per_class_indices(train_full, cfg.train_cap_per_class)
    test_indices = {d: [] for d in range(10)}
    for i, (_, y) in enumerate(test_full):
        test_indices[y].append(i)

    # Loop over 10 classes
    for d in range(10):
        print(f"\n=== Digit {d} ===")

        train_ds_d = Subset(train_full, train_indices[d])
        test_ds_d = Subset(test_full, test_indices[d])

        print(f"Training VAE(d={d}) with {len(train_ds_d)} samples...")

        train_loader = DataLoader(train_ds_d, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        test_loader = DataLoader(test_ds_d, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        model = VAE(latent_dim=cfg.latent_dim).to(cfg.device)
        ckpt_path = os.path.join(cfg.ckpt_dir, f"vae_mnist_digit{d}_lat{cfg.latent_dim}.pt")

        if os.path.exists(ckpt_path):
            print(f"Loading VAE(d={d}) from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=cfg.device)
            model.load_state_dict(state)
        else:
            train_one_class(d, train_loader, model, cfg)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved VAE(d={d}) to {ckpt_path}")

        # Evaluate
        evaluate_one_class(d, test_loader, model, cfg)
        # Visualize
        visualize_one_class(d, test_ds_d, model, cfg)

    print("\nDone training/evaluating 10 VAEs (digits 0–9).")


if __name__ == "__main__":
    main()