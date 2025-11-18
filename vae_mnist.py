import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ---------------------------
# 1. Small Conv VAE
# ---------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim  # latent vector size

        # Encoder: [B,1,28,28] -> [B,16,14,14] -> [B,32,7,7]
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
        )
        enc_out_c, enc_out_h, enc_out_w = 32, 7, 7
        self.enc_out_dim = enc_out_c * enc_out_h * enc_out_w  # 32*7*7 = 1568

        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder: z -> [B,32,7,7] -> [B,16,14,14] -> [B,1,28,28]
        self.fc_z = nn.Linear(latent_dim, self.enc_out_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 14 -> 28
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
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + kl) / x.size(0)


# ---------------------------
# 3. Train / load + visualize
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    batch_size = 64
    epochs = 500
    latent_dim = 32          # latent vector dimension (size of z)
    lr = 1e-3

    # save / load from checkpoints/ as .pt
    os.makedirs("checkpoints", exist_ok=True)
    model_path = os.path.join("checkpoints", f"vae_mnist_lat{latent_dim}.pt")

    transform = transforms.ToTensor()

    full_train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=False
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=False
    )

    # Use only 10,000 for training
    train_size = 20000
    train_dataset = Subset(full_train_dataset, list(range(train_size)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    vae = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # ----- load if exists, otherwise train -----
    if os.path.exists(model_path):
        print(f"Loading pretrained VAE from '{model_path}'")
        vae.load_state_dict(torch.load(model_path, map_location=device))
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

            print(f"Epoch {epoch+1}/{epochs} - train loss: {total_loss / n:.4f}")

        # save model as .pt inside checkpoints
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
    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i, 0], cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Original")

        plt.subplot(2, n, n + i + 1)
        plt.imshow(recon_x[i, 0], cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Recon")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()