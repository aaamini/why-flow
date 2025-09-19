#%%
import torch
from torch import Tensor
from torch import nn

from evf import EmpiricalVectorField, sample_rho_t_empirical, Integrator, uniform_grid
from metrics import nn_rmse_to_targets
from circle_images import generate_circle_images, show_image_grid

import torchvision
from torchvision import models
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def euler_one_step(Y: Tensor, x_t: Tensor, t: float) -> Tensor:
    """One Euler step to t=1 using EVF in the current feature space."""
    field = EmpiricalVectorField(Y)
    t_tensor = x_t.new_full((x_t.size(0), 1), float(t))
    v = field(t_tensor, x_t)
    return x_t + (1.0 - float(t)) * v

def dode(Y: Tensor, n_steps: int, t: float, n_samp: int, method: str, x0: Tensor = None) -> Tensor:
    """Integrate ODE from time t to 1 starting from x0 (or noise if x0 is None). Returns terminal state only."""
    field = EmpiricalVectorField(Y.double())
    integ = Integrator(field)
    t_grid = uniform_grid(n_steps, t1=t)
    if x0 is None:
        x0 = torch.randn(n_samp, Y.size(1), device=Y.device, dtype=Y.dtype)
    else:
        x0 = x0.to(device=Y.device, dtype=Y.dtype)
    xT, _, _ = integ.integrate(x0, t_grid, method=method, return_traj=True)
    return xT.float()

class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return x

class LinearDecoder(nn.Module):
    """Simple linear decoder mapping latent -> image space."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    @torch.no_grad()
    def fit_ridge(self, X_latent: Tensor, Y_img: Tensor, ridge_lambda: float = 1e-3):
        X = X_latent
        Y = Y_img
        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        Xc = X - X_mean
        Yc = Y - Y_mean
        d = Xc.size(1)
        XtX = Xc.T @ Xc
        reg = ridge_lambda * torch.eye(d, device=X.device, dtype=X.dtype)
        W_t = torch.linalg.solve(XtX + reg, Xc.T @ Yc)  # (d x out)
        W = W_t.T  # (out x d)
        b = (Y_mean - X_mean @ W_t).squeeze(0)  # (out)
        self.linear.weight.copy_(W)
        self.linear.bias.copy_(b)

    def forward(self, z: Tensor) -> Tensor:
        return self.linear(z)

# ResNet18 encoder (penultimate layer features)
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
resnet.eval()
feature_dim = resnet.fc.in_features
resnet.fc = nn.Identity()

@torch.no_grad()
def encoder(flat_imgs: Tensor, batch_size: int = None) -> Tensor:
    if batch_size is None:
        batch_size = encoder_batch_size
    outputs = []
    N = flat_imgs.size(0)
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        x = flat_imgs[start:end].to(device)
        n = x.size(0)
        x = x.view(n, 1, image_size, image_size)
        x = x.expand(n, 3, image_size, image_size)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        outputs.append(resnet(x).detach())
    return torch.cat(outputs, dim=0)

# experiment parameters
t = 0.99
n_target = 600
n_samp = 2048
image_size = 32
noise_std = 0.0
thickness = 2.0
show_vis = True
fit_decoder = True
ridge_lambda = 1e-2
n_steps = 4
encoder_batch_size = 128

generate_ground_truth = lambda n: generate_circle_images(
    n,
    image_size=image_size,
    noise_std=noise_std,
    thickness=thickness,
    device=device,
    dtype=torch.float32,
)

with torch.no_grad():
    # 1) Training images (Y_img)
    Y_img = generate_ground_truth(n_target)
    D_img = Y_img.size(1)

    # 2) Encode training images to latent zY, fit linear decoder back to images
    zY = encoder(Y_img)  # [n, D_lat]
    D_lat = zY.size(1)
    decoder = LinearDecoder(D_lat, D_img).to(device)
    if fit_decoder:
        decoder.fit_ridge(zY, Y_img, ridge_lambda=ridge_lambda)

    # 3) Sample x_t in latent space directly: x_t_lat ~ ρ_t(zY)
    x_t_lat = sample_rho_t_empirical(zY, t, n_samp)  # [m, D_lat]

    # 4) One-step Euler in latent space
    x1_euler_lat = euler_one_step(zY, x_t_lat, t)

    # 5) ODE integration in latent space from the SAME x_t_lat, from t -> 1
    x1_dode_lat = dode(zY, n_steps=n_steps, t=t, n_samp=n_samp, method="rk2", x0=x_t_lat)

    # 6) Decode to image space (for visualization)
    x_t = decoder(x_t_lat)
    x1_euler = decoder(x1_euler_lat)
    x1_dode = decoder(x1_dode_lat)

    # 7) Evaluate novelty/fit in LATENT space (avoids decoder bias)
    novelty_xt = nn_rmse_to_targets(x_t_lat, zY)
    novelty_euler = nn_rmse_to_targets(x1_euler_lat, zY)
    novelty_dode = nn_rmse_to_targets(x1_dode_lat, zY)

    # Dense latent cover using fresh images
    Y_fine = generate_ground_truth(max(n_samp, 8000))
    zY_fine = encoder(Y_fine)

    fit_xt = nn_rmse_to_targets(x_t_lat, zY_fine)
    fit_euler = nn_rmse_to_targets(x1_euler_lat, zY_fine)
    fit_dode = nn_rmse_to_targets(x1_dode_lat, zY_fine)

    # 8) True baseline (same generator, encoded to latent)
    X_true = generate_ground_truth(n_samp)
    X_true_lat = encoder(X_true)
    true_novelty = nn_rmse_to_targets(X_true_lat, zY)
    true_fit = nn_rmse_to_targets(X_true_lat, zY_fine)

    print(f"t={t:.3f} | latent_dim={D_lat} | image_dim={D_img}")
    print(f"True:  novelty={true_novelty:.4g} | fit={true_fit:.4g}")
    print(f"x_t:   novelty={novelty_xt:.4g} | fit={fit_xt:.4g}")
    print(f"Euler: novelty={novelty_euler:.4g} | fit={fit_euler:.4g}")
    print(f"Dode:  novelty={novelty_dode:.4g} | fit={fit_dode:.4g}")

    if show_vis:
        show_image_grid(Y_img, image_size=image_size, title="Training targets (Y)")
        show_image_grid(X_true, image_size=image_size, title="True baseline (fresh samples)")
        show_image_grid(x_t, image_size=image_size, title=f"x_t ~ rho_t, t={t:.2f}")
        show_image_grid(x1_euler, image_size=image_size, title="x1 via one-step Euler (latent)")
        show_image_grid(x1_dode, image_size=image_size, title="x1 via Dode (latent)")

# %%