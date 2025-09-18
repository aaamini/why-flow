import torch
from torch import nn, optim, Tensor # <-- CORRECTED: Added Tensor import
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
# from tqdm import tqdm # Removing tqdm for simplicity

# Import necessary functions from your custom modules
from circle_images import generate_circle_images, show_image_grid
# from metrics import nn_rmse_to_targets # Removing for simplicity, not strictly needed for VAE training/gen


# --- VAE Model Definition (Simplified) ---
class SimpleVAE(nn.Module):
    def __init__(self, image_size=32, latent_dim=32):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        input_dim = image_size * image_size # 32*32 = 1024

        # Encoder: Simple 2-layer MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # Reduced hidden dimension
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim) # Outputs mu and log_var
        )

        # Decoder: Simple 2-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), # Reduced hidden dimension
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Output layer
            nn.Sigmoid()                # Pixel values 0-1
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encodes input x into latent mean (mu) and log-variance (log_var)."""
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1) # Split into mu and log_var
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """Decodes a latent vector z into an image."""
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick: samples z from N(mu, exp(log_var))."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the VAE."""
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

    return (recon_loss + kl_div_loss) / x.size(0) # Average loss per sample

# --- Training and Evaluation Logic ---

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration Parameters ---
    image_size = 32         # Width and height of the square images
    latent_dim = 16         # Dimension of the latent space (as requested)
    num_train_images = 5000 # Number of synthetic images for training
    num_test_images = 500   # Number of synthetic images for testing
    batch_size = 64         # Batch size for training
    epochs = 100             # Reduced epochs for simplicity, adjust if needed
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