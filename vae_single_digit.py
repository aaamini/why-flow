import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------------------
# VAE definition (same as before)
# ---------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
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

def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + kl) / x.size(0)

@torch.no_grad()
def mse_metric(x, y):
    return F.mse_loss(y, x, reduction="mean").item()

@torch.no_grad()
def psnr_metric(x, y, data_range=1.0, eps=1e-10):
    mse = F.mse_loss(y, x, reduction="mean").item()
    mse = max(mse, eps)
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)

@torch.no_grad()
def ssim_metric(x, y, data_range=1.0, K1=0.01, K2=0.03, window_size=7):
    pad = window_size // 2
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    w = torch.ones((1, 1, window_size, window_size), device=x.device) / (window_size * window_size)
    mu_x = F.conv2d(x, w, padding=pad)
    mu_y = F.conv2d(y, w, padding=pad)
    mu_x2 = mu_x.pow(2); mu_y2 = mu_y.pow(2); mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, w, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return ssim_map.mean().item()

def filter_indices_by_label(dataset, label):
    idxs = [i for i, (_, y) in enumerate(dataset) if y == label]
    return idxs

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    batch_size = 128
    epochs = 50
    latent_dim = 32
    lr = 1e-3
    train_cap_per_class = 6000   # adjust if you want to limit training per class

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    transform = transforms.ToTensor()
    train_full = datasets.MNIST(root="./data", train=True, transform=transform, download=False)
    test_full  = datasets.MNIST(root="./data", train=False, transform=transform, download=False)

    # Precompute per-class indices
    train_indices = {d: [] for d in range(10)}
    test_indices  = {d: [] for d in range(10)}
    for i, (_, y) in enumerate(train_full):
        if len(train_indices[y]) < train_cap_per_class:
            train_indices[y].append(i)
    for i, (_, y) in enumerate(test_full):
        test_indices[y].append(i)

    # Train or load 10 VAEs
    vaes = {}
    for d in range(10):
        print(f"\n=== Digit {d} ===")
        train_ds_d = Subset(train_full, train_indices[d])
        test_ds_d  = Subset(test_full,  test_indices[d])

        train_loader = DataLoader(train_ds_d, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader  = DataLoader(test_ds_d,  batch_size=batch_size, shuffle=False, num_workers=0)

        model_path = os.path.join("checkpoints", f"vae_mnist_digit{d}_lat{latent_dim}.pt")
        vae = VAE(latent_dim=latent_dim).to(device)
        opt = torch.optim.Adam(vae.parameters(), lr=lr)

        if os.path.exists(model_path):
            print(f"Loading VAE(d={d}) from {model_path}")
            vae.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Training VAE(d={d}) with {len(train_ds_d)} samples...")
            vae.train()
            for epoch in range(epochs):
                total, n = 0.0, 0
                for x, _ in train_loader:
                    x = x.to(device)
                    opt.zero_grad()
                    recon, mu, logvar = vae(x)
                    loss = vae_loss(recon, x, mu, logvar)
                    loss.backward()
                    opt.step()
                    total += loss.item() * x.size(0)
                    n += x.size(0)
                print(f"  Epoch {epoch+1:03d} | loss {total/max(n,1):.4f}")
            torch.save(vae.state_dict(), model_path)
            print(f"Saved VAE(d={d}) to {model_path}")

        # Evaluate metrics for this class
        vae.eval()
        mse_sum = psnr_sum = ssim_sum = 0.0
        n_samples = 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                recon, _, _ = vae(x)
                mse_sum += F.mse_loss(recon, x, reduction="sum").item()
                psnr_sum += psnr_metric(x, recon) * x.size(0)
                ssim_sum += ssim_metric(x, recon) * x.size(0)
                n_samples += x.size(0)
        mse_mean = mse_sum / (n_samples * 28 * 28)
        psnr_mean = psnr_sum / n_samples
        ssim_mean = ssim_sum / n_samples
        print(f"Metrics(d={d}) | MSE {mse_mean:.6f} | PSNR {psnr_mean:.2f} dB | SSIM {ssim_mean:.4f}")

        # Visualize a small grid for this class
        sample_loader = DataLoader(test_ds_d, batch_size=8, shuffle=True)
        x, _ = next(iter(sample_loader))
        x = x.to(device)
        recon, _, _ = vae(x)
        x = x.cpu(); recon = recon.cpu()
        n = x.size(0)
        plt.figure(figsize=(8, 2.5))
        for i in range(n):
            plt.subplot(2, n, i+1)
            plt.imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            if i == 0: plt.title(f"Digit {d} - Orig")
            plt.subplot(2, n, n+i+1)
            plt.imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            if i == 0: plt.title("Recon")
        plt.tight_layout()
        fig_path = os.path.join("outputs", f"digit{d}_recon_grid.png")
        plt.savefig(fig_path, dpi=160, bbox_inches="tight")
        print(f"Saved grid to {fig_path}")
        plt.close()

        vaes[d] = vae  # keep in memory if you need later

    print("\nDone training/evaluating 10 VAEs (digits 0–9).")

if __name__ == "__main__":
    main()