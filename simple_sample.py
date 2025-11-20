import torch
import os
import argparse
from torchvision.utils import save_image
from models.unet import UNetModel

# --- Configuration ---
CONFIG = {
    "batch_size": 16,
    "steps": 50,        # Number of ODE steps (NFE)
    "method": "euler",  # 'euler' or 'heun'
    "output_dir": "./simple_samples",
    "checkpoint_path": "./simple_output/checkpoint_10.pth", # Default checkpoint
}

# --- ODE Solvers ---

def euler_solve(model, x, steps, device):
    """
    Simple Euler integration: x_{t+dt} = x_t + v(x_t, t) * dt
    """
    dt = 1.0 / steps
    t_grid = torch.linspace(0, 1, steps + 1, device=device)
    
    print(f"Sampling with Euler method ({steps} steps)...")
    
    for i in range(steps):
        t = t_grid[i]
        # Broadcast t to [B]
        t_batch = torch.ones(x.shape[0], device=device) * t
        
        # Get velocity
        with torch.no_grad():
            v = model(x, t_batch, extra={})
            
        # Update x
        x = x + v * dt
        
    return x

def heun_solve(model, x, steps, device):
    """
    Heun's method (2nd order):
    d1 = v(x_t, t)
    x_mid = x_t + d1 * dt
    d2 = v(x_mid, t + dt)
    x_{t+dt} = x_t + (d1 + d2) / 2 * dt
    """
    dt = 1.0 / steps
    t_grid = torch.linspace(0, 1, steps + 1, device=device)
    
    print(f"Sampling with Heun method ({steps} steps)...")
    
    for i in range(steps):
        t = t_grid[i]
        t_next = t_grid[i+1]
        t_batch = torch.ones(x.shape[0], device=device) * t
        t_next_batch = torch.ones(x.shape[0], device=device) * t_next
        
        # d1
        with torch.no_grad():
            d1 = model(x, t_batch, extra={})
            
        # x_mid
        x_mid = x + d1 * dt
        
        # d2
        with torch.no_grad():
            d2 = model(x_mid, t_next_batch, extra={})
            
        # Update x
        x = x + (d1 + d2) * 0.5 * dt
        
    return x

# --- Main Sampling Loop ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CONFIG["checkpoint_path"], help="Path to model checkpoint")
    parser.add_argument("--steps", type=int, default=CONFIG["steps"], help="Number of sampling steps")
    parser.add_argument("--method", type=str, default=CONFIG["method"], choices=["euler", "heun"], help="ODE solver method")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found at {args.checkpoint}. Please train first or specify path.")
        return

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint["model_config"]
    
    # 2. Initialize Model
    model = UNetModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # 3. Sample
    # Start from Gaussian noise x_0 ~ N(0, I)
    x_0 = torch.randn(CONFIG["batch_size"], 3, 32, 32, device=device)
    
    if args.method == "euler":
        samples = euler_solve(model, x_0, args.steps, device)
    else:
        samples = heun_solve(model, x_0, args.steps, device)
        
    # 4. Post-process and Save
    # Scale from [-1, 1] to [0, 1]
    samples = torch.clamp(samples * 0.5 + 0.5, 0, 1)
    
    output_path = os.path.join(CONFIG["output_dir"], f"samples_{args.method}_{args.steps}.png")
    save_image(samples, output_path, nrow=4)
    print(f"Saved samples to {output_path}")

if __name__ == "__main__":
    main()
