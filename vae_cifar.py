import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ---------------------------
# 1. Convolutional VAE for CIFAR-10
# ---------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: [B,3,32,32] -> [B,64,16,16] -> [B,128,8,8] -> [B,256,4,4]
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 8 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        enc_out_c, enc_out_h, enc_out_w = 256, 4, 4
        self.enc_out_dim = enc_out_c * enc_out_h * enc_out_w  # 256*4*4 = 4096

        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder: z -> [B,256,4,4] -> [B,128,8,8] -> [B,64,16,16] -> [B,3,32,32]
        self.fc_z = nn.Linear(latent_dim, self.enc_out_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 4 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 8 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 16 -> 32
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
        h = h.view(z.size(0), 256, 4, 4)
        x_recon = self.dec_conv(h)
        return x_recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


# ---------------------------
# 2. Loss
# ---------------------------
def vae_loss(recon_x, x, mu, log_var):
    """VAE loss = reconstruction loss + KL divergence"""
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + kl) / x.size(0)


# ---------------------------
# 3. Train / load + visualize
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    batch_size = 128
    epochs = 100
    latent_dim = 64          # latent vector dimension (size of z)
    lr = 1e-3

    # save / load from checkpoints/ as .pt
    os.makedirs("checkpoints", exist_ok=True)
    model_path = os.path.join("checkpoints", f"vae_cifar10_lat{latent_dim}.pt")

    transform = transforms.ToTensor()

    full_train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=transform, download=True
    )

    # Use full training set (50k samples)
    train_dataset = full_train_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    vae = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # ----- load if exists, otherwise train -----
    if os.path.exists(model_path):
        print(f"Loading pretrained VAE from '{model_path}'")
        vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print("No saved model found, training from scratch...")
        vae.train()
        for epoch in range(epochs):
            total_loss, n = 0.0, 0
            for x, _ in train_loader:
                x = x.to(device)
                optimizer.zero_grad()
                recon_x, mu, log_var = vae(x)
                loss = vae_loss(recon_x, x, mu, log_var)
                loss.backward()
                optimizer.step()

                b = x.size(0)
                total_loss += loss.item() * b
                n += b

            avg_loss = total_loss / n
            print(f"Epoch {epoch+1}/{epochs} - train loss: {avg_loss:.4f}")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(vae.state_dict(), model_path)
                print(f"  Checkpoint saved to '{model_path}'")

        # Final save
        torch.save(vae.state_dict(), model_path)
        print(f"Saved trained VAE to '{model_path}'")

    # ----- evaluation: show 16 test reconstructions -----
    vae.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)
        recon_x, _, _ = vae(x)
        x = x.cpu()
        recon_x = recon_x.cpu()

    n = x.size(0)
    plt.figure(figsize=(12, 6))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        img = x[i].permute(1, 2, 0).numpy()  # CHW -> HWC
        plt.imshow(img)
        plt.axis("off")
        if i == 0:
            plt.title("Original")

        # Reconstruction
        plt.subplot(2, n, n + i + 1)
        img_recon = recon_x[i].permute(1, 2, 0).numpy()
        plt.imshow(img_recon)
        plt.axis("off")
        if i == 0:
            plt.title("Recon")

    plt.tight_layout()
    plt.savefig(f"vae_cifar10_lat{latent_dim}_reconstructions.png", dpi=150)
    print(f"Saved reconstruction visualization to 'vae_cifar10_lat{latent_dim}_reconstructions.png'")
    plt.show()


if __name__ == "__main__":
    main()
