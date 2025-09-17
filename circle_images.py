#%%
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

# %%

@torch.no_grad()
def generate_circle_images(
    num_images: int,
    image_size: int = 32,
    radius_range=(6.0, 12.0),
    noise_std: float = 0.0,
    thickness: float = 1.0,
    device: str | torch.device = None,
    dtype: torch.dtype = torch.float32,
    return_params: bool = False,
) -> Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    H = W = int(image_size)
    assert H == W, "Only square images are supported"

    # Sample circle parameters
    radii = torch.empty(num_images).uniform_(radius_range[0], radius_range[1])
    centers = torch.empty(num_images, 2)
    # Center shifts in pixels; keep within bounds
    max_shift = (image_size // 4)
    centers[:, 0].uniform_(W//2 - max_shift, W//2 + max_shift)  # cx
    centers[:, 1].uniform_(H//2 - max_shift, H//2 + max_shift)  # cy

    # Coordinate grid
    ys = torch.arange(H).view(H, 1).expand(H, W)
    xs = torch.arange(W).view(1, W).expand(H, W)
    grid = torch.stack([xs, ys], dim=0).float()  # [2,H,W]

    images = []
    for i in range(num_images):
        cx, cy = centers[i]
        r = radii[i]
        dx = grid[0] - cx
        dy = grid[1] - cy
        dist = torch.sqrt(dx * dx + dy * dy)
        ring = torch.exp(-0.5 * ((dist - r) / max(thickness, 1e-3)) ** 2)
        img = ring
        if noise_std > 0:
            img = img + noise_std * torch.randn_like(img)
        img = img.clamp(0.0, 1.0)
        images.append(img.unsqueeze(0))  # [1,H,W]

    imgs = torch.cat(images, dim=0)  # [N, H, W]
    imgs = imgs.to(device=device, dtype=dtype)
    X = imgs.view(num_images, -1)  # flatten to [N, H*W]
    if return_params:
        params = torch.stack([centers[:,0], centers[:,1], radii], dim=1).to(device=device, dtype=dtype)
        return X, params
    return X


@torch.no_grad()
def show_image_grid(flat: Tensor, image_size: int = 32, title: str = "", max_images: int = 36):
    N = min(flat.size(0), max_images)
    H = W = image_size
    imgs = flat[:N].detach().cpu().view(N, H, W)
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis('off')
        if idx < N:
            ax.imshow(imgs[idx].numpy(), cmap='gray', vmin=0.0, vmax=1.0)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()