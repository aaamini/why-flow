import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import os
import time
import math

# Import the model definition (assuming we are in flow_matching/examples/image/)
from models.unet import UNetModel
from models.ema import EMA

# --- Configuration ---
CONFIG = {
    "dataset": "cifar10",
    "batch_size": 128,  # Adjusted for single GPU
    "epochs": 20,      # Reduced for demonstration, original is 3000
    "lr": 1e-4,
    "data_path": "./data",
    "output_dir": "./simple_output",
    "use_ema": True,
    "skewed_timesteps": True,
    "save_every": 10,
}

# --- Model Config for CIFAR-10 (from models/model_configs.py) ---
MODEL_CONFIG = {
    "in_channels": 3,
    "model_channels": 128,
    "out_channels": 3,
    "num_res_blocks": 4,
    "attention_resolutions": [2],
    "dropout": 0.3,
    "channel_mult": [2, 2, 2],
    "conv_resample": False,
    "dims": 2,
    "num_classes": None, # Unconditional
    "use_checkpoint": False,
    "num_heads": 1,
    "num_head_channels": -1,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "resblock_updown": False,
    "use_new_attention_order": True,
    "with_fourier_features": False,
}

# --- Helper Functions ---

def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    """Samples timesteps with a skewed distribution (more near 0 and 1)."""
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time

def get_transforms():
    return v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
    ])

# --- Main Training Loop ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["data_path"], exist_ok=True)

    # 1. Data
    print("Loading data...")
    dataset = datasets.CIFAR10(
        root=CONFIG["data_path"],
        train=True,
        download=True,
        transform=get_transforms()
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )

    # 2. Model
    print("Initializing model...")
    model = UNetModel(**MODEL_CONFIG)
    model.to(device)

    if CONFIG["use_ema"]:
        model = EMA(model)
        print("Using EMA")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    
    # 3. Training
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        model.train(True)
        epoch_loss = 0.0
        
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Scale images to [-1, 1]
            x_1 = images * 2.0 - 1.0
            
            # Sample noise x_0 ~ N(0, I)
            x_0 = torch.randn_like(x_1)
            
            # Sample timesteps t
            if CONFIG["skewed_timesteps"]:
                t = skewed_timestep_sample(x_1.shape[0], device)
            else:
                t = torch.rand(x_1.shape[0], device=device)
            
            # Reshape t for broadcasting: [B] -> [B, 1, 1, 1]
            t_reshaped = t.view(-1, 1, 1, 1)
            
            # Compute interpolant x_t = (1-t)x_0 + t*x_1 (Conditional Optimal Transport)
            x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1
            
            # Target velocity u_t = x_1 - x_0
            u_t = x_1 - x_0
            
            # Predict velocity
            # Note: model expects t as [B] tensor, and an 'extra' dict for labels (empty for unconditional)
            if isinstance(model, EMA):
                # EMA wrapper forward calls model forward
                prediction = model(x_t, t, extra={})
            else:
                prediction = model(x_t, t, extra={})
            
            # Loss: MSE(prediction, u_t)
            loss = torch.mean((prediction - u_t) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if CONFIG["use_ema"]:
                model.update_ema()
                
            epoch_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch} [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = os.path.join(CONFIG["output_dir"], f"checkpoint_{epoch+1}.pth")
            
            # Handle EMA vs regular model saving
            state_dict = model.model.state_dict() if isinstance(model, EMA) else model.state_dict()
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": CONFIG,
                "model_config": MODEL_CONFIG,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()
