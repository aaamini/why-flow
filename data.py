
from __future__ import annotations
from typing import Tuple
import math
import torch
from torch import Tensor

def _to_tensor(x, device, dtype) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def make_2moons(n: int, *, noise: float, device, dtype) -> Tensor:
    from sklearn.datasets import make_moons
    X, _ = make_moons(n, noise=noise)
    return _to_tensor(X, device, dtype)

def make_2circles(n: int, *, noise: float, factor: float, device, dtype) -> Tensor:
    from sklearn.datasets import make_circles
    X, _ = make_circles(n, noise=noise, factor=factor)
    return _to_tensor(X, device, dtype)

def make_line(n: int, *, width: float, xspan: float, device, dtype) -> Tensor:
    x = (torch.rand(n, device=device) * 2 - 1) * xspan
    y = torch.randn(n, device=device) * width
    return torch.stack([x, y], dim=1).to(dtype)

def make_sine(n: int, *, amp: float, noise: float, xspan: float, device, dtype) -> Tensor:
    x = (torch.rand(n, device=device) * 2 - 1) * xspan
    y = amp * torch.sin(2 * math.pi * x) + noise * torch.randn(n, device=device)
    return torch.stack([x, y], dim=1).to(dtype)

def make_circles_pixels(n: int, *, image_size: int, noise_std: float, thickness: float, vary_center: bool, device, dtype) -> Tensor:
    from circle_images import generate_circle_images
    X = generate_circle_images(
        n, image_size=image_size, noise_std=noise_std, thickness=thickness,
        device=device, dtype=dtype, vary_center=vary_center, flatten=True
    )
    return X

def load_dataset(
    name: str, n_train: int, n_real: int, *,
    device=None, dtype=torch.float32,
    moons_noise: float = 0.0,
    circles_noise: float = 0.0,
    circles_factor: float = 0.5,
    line_width: float = 0.1,
    line_xspan: float = 1.0,
    sine_amp: float = 0.5,
    sine_noise: float = 0.0,
    sine_xspan: float = 1.0,
    image_size: int = 32,
    pix_noise_std: float = 0.0,
    pix_thickness: float = 2.0,
    pix_vary_center: bool = True,
) -> Tuple[Tensor, Tensor]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "2moons":
        make = lambda n: make_2moons(n, noise=moons_noise, device=device, dtype=dtype)
    elif name == "2circles":
        make = lambda n: make_2circles(n, noise=circles_noise, factor=circles_factor, device=device, dtype=dtype)
    elif name == "line":
        make = lambda n: make_line(n, width=line_width, xspan=line_xspan, device=device, dtype=dtype)
    elif name == "sine":
        make = lambda n: make_sine(n, amp=sine_amp, noise=sine_noise, xspan=sine_xspan, device=device, dtype=dtype)
    elif name == "circles_pixels":
        make = lambda n: make_circles_pixels(n, image_size=image_size, noise_std=pix_noise_std,
                                             thickness=pix_thickness, vary_center=pix_vary_center,
                                             device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return make(n_train), make(n_real)
