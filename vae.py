#%%
import torch
from torch import nn, optim, Tensor  # <-- CORRECTED: Added Tensor import
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
# from tqdm import tqdm # Removing tqdm for simplicity

# Import necessary functions from your custom modules
from circle_images import generate_circle_images, show_image_grid
# from metrics import nn_rmse_to_targets # Removing for simplicity, not strictly needed for VAE training/gen


# --- VAE Model Definition (Convolutional with 2 conv layers per side) ---
class SimpleVAE(nn.Module):
    def __init__(self, image_size=32, latent_dim=32):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.c = 1  # single channel images (grayscale)

        # Encoder: exactly 2 Conv2d layers
        # [B,1,32,32] -> [B,32,16,16] -> [B,64,8,8]
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.ReLU(inplace=True),
        )
        self.enc_feat_h = self.image_size // 4  # 32 -> 8
        self.enc_feat_w = self.image_size // 4  # 32 -> 8
        self.enc_feat_c = 64
        self.enc_out_dim = self.enc_feat_c * self.enc_feat_h * self.enc_feat_w  # 64*8*8 = 4096

        # Latent heads (2 linear layers; these are standard for VAEs)
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder: 1 linear + exactly 2 ConvTranspose2d layers
        # latent -> [B,64,8,8] -> [B,32,16,16] -> [B,1,32,32]
        self.fc_z = nn.Linear(latent_dim, self.enc_out_dim)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # -> 32x32
            nn.Sigmoid(),  # keep outputs in [0,1] to match BCE loss
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Accept flattened [B, 1024] or [B,1,32,32]
        if x.dim() == 2:
            x = x.view(-1, 1, self.image_size, self.image_size)
        h = self.encoder_cnn(x)  # [B,64,8,8]
        h = h.view(h.size(0), -1)  # [B, 4096]
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_z(z)  # [B, 4096]
        h = h.view(z.size(0), self.enc_feat_c, self.enc_feat_h, self.enc_feat_w)  # [B,64,8,8]
        x_rec = self.decoder_cnn(h)  # [B,1,32,32]
        return x_rec.view(z.size(0), -1)  # flattened [B,1024]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


# --- VAE Loss Function ---
def vae_loss(recon_x: Tensor, x: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
    """Calculates the VAE loss (Reconstruction + KL Divergence)."""
    # Reconstruction Loss (Binary Cross-Entropy for 0-1 pixel values)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL Divergence Loss: D_KL(N(mu, sigma^2) || N(0, 1))
    kl_div_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return (recon_loss + kl_div_loss) / x.size(0)  # Average loss per sample


#%%
# --- Training and Evaluation Logic ---

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration Parameters ---
    image_size = 32         # Width and height of the square images
    latent_dim = 32         # Dimension of the latent space (as requested)
    num_train_images = 5000 # Number of synthetic images for training
    num_test_images = 500   # Number of synthetic images for testing
    batch_size = 64         # Batch size for training
    epochs = 1000           # Reduced epochs for simplicity, adjust if needed
    learning_rate = 1e-3    # Learning rate for Adam optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available

    print(f"Using device: {device}")

    # --- 1. Generate Datasets (Circles) ---
    print("Generating training data...")
    train_data_flat = generate_circle_images(
        num_train_images,
        image_size=image_size,
        noise_std=0.0,
        thickness=2.0,
        device=device,
        dtype=torch.float32,
    )
    train_dataset = TensorDataset(train_data_flat)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Generated {num_train_images} training images.")

    print("Generating test data...")
    test_data_flat = generate_circle_images(
        num_test_images,
        image_size=image_size,
        noise_std=0.0,
        thickness=2.0,
        device=device,
        dtype=torch.float32,
    )
    print(f"Generated {num_test_images} test images.")

    # Optional: show some example training images
    print("\n--- Example Training Images ---")
    show_image_grid(
        train_data_flat,
        image_size=image_size,
        title="Example Training Images",
        max_images=16
    )

    # --- 2. Initialize and Train the Simple VAE Model ---
    vae = SimpleVAE(image_size=image_size, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    vae.train()

    print(f"\nStarting Simple VAE training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for data, in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = vae(data)
            loss = vae_loss(recon_batch, data, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    print("Simple VAE training finished.")

    # --- 3. Evaluate Reconstruction Quality and Generate New Samples ---
    vae.eval() # Set model to evaluation mode
    # --- Original vs. Reconstructed Images (Test Set) ---
    print("\n--- Original vs. Reconstructed Images (Test Set) ---")
    display_data = test_data_flat[:8] # Show 8 original/recon pairs
    original_images_np = display_data.cpu().numpy()
    with torch.no_grad(): # No gradients needed for evaluation
        recon_images_tensor, _, _ = vae(display_data.to(device))
    recon_images_np = recon_images_tensor.cpu().numpy()
    combined_images = np.concatenate((original_images_np, recon_images_np), axis=0)
    show_image_grid(
        torch.from_numpy(combined_images),
        image_size=image_size,
        title="Top: Originals | Bottom: Reconstructions (Test Set)",
        max_images=16 # 8 originals + 8 reconstructions
    )

    # --- Randomly Generated Images from Latent Space ---
    print("\n--- Randomly Generated Images from Latent Space ---")
    num_random_generations = 16
    with torch.no_grad(): # No gradients needed for generation
        random_latent_vectors = torch.randn(num_random_generations, vae.latent_dim).to(device)
        generated_images = vae.decode(random_latent_vectors)
    show_image_grid(
        generated_images,
        image_size=image_size,
        title="Randomly Generated Images by Simple VAE",
        max_images=num_random_generations
    )
# %%
