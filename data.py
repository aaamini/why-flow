
"""
data.py
-------
Dataset helpers for EVF PR evaluation.

Available names for `load_dataset`:
- "2moons", "2circles", "line", "sine"      → returns [N, 2]
- "circles_pixels"                           → flattened circle images [N, H*W] in [0,1]
- "mnist_pixels"                             → flattened MNIST images  [N, 28*28] in [0,1]
"""

from __future__ import annotations
from typing import Tuple, Optional, Sequence
import math
import torch
from torch import Tensor

def _to_tensor(x, device, dtype) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

# --------------------- 2D toy datasets ---------------------

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

# --------------------- Pixel datasets ---------------------

def make_circles_pixels(n: int, *, image_size: int, noise_std: float, thickness: float, vary_center: bool, device, dtype) -> Tensor:
    """
    Flattened circle images in [0,1]. Returns [N, H*W].
    """
    from circle_images import generate_circle_images
    X = generate_circle_images(
        n, image_size=image_size, noise_std=noise_std, thickness=thickness,
        device=device, dtype=dtype, vary_center=vary_center, flatten=True
    )
    return X

def _load_mnist_split(n: int, *, split: str, data_root: str, classes: Optional[Sequence[int]],
                      shuffle: bool, seed: Optional[int], device, dtype) -> Tensor:
    """
    Load an MNIST split ("train" or "test"), optionally filter by `classes`,
    take (shuffled) first n, flatten to [N, 784], scale to [0,1], move to device/dtype.
    """
    try:
        import torchvision
    except Exception as e:
        raise ImportError("torchvision is required for 'mnist_pixels'") from e

    is_train = (split.lower() == "train")
    ds = torchvision.datasets.MNIST(root=data_root, train=is_train, download=True)
    X = ds.data.float() / 255.0   # [N,28,28] in [0,1], on CPU
    y = ds.targets                # [N]

    if classes is not None and len(classes) > 0:
        mask = torch.zeros_like(y, dtype=torch.bool)
        for c in classes:
            mask |= (y == int(c))
        X = X[mask]
        y = y[mask]

    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        idx = torch.randperm(X.size(0), generator=g)
        X = X[idx]

    X = X[:min(n, X.size(0))]
    X = X.view(X.size(0), -1).to(device=device, dtype=dtype)
    return X

def make_mnist_pixels(n: int, *, split: str, data_root: str, classes: Optional[Sequence[int]],
                      shuffle: bool, seed: Optional[int], device, dtype) -> Tensor:
    return _load_mnist_split(n, split=split, data_root=data_root, classes=classes,
                             shuffle=shuffle, seed=seed, device=device, dtype=dtype)

# --------------------- Public API ---------------------

def load_dataset(
    name: str, n_train: int, n_real: int, *,
    device=None, dtype=torch.float32,
    # 2D toy params
    moons_noise: float = 0.0,
    circles_noise: float = 0.0,
    circles_factor: float = 0.5,
    line_width: float = 0.1,
    line_xspan: float = 1.0,
    sine_amp: float = 0.5,
    sine_noise: float = 0.0,
    sine_xspan: float = 1.0,
    # circles_pixels params
    image_size: int = 32,
    pix_noise_std: float = 0.0,
    pix_thickness: float = 2.0,
    pix_vary_center: bool = True,
    # mnist params
    mnist_data_root: str = "./data",
    mnist_classes: Optional[Sequence[int]] = None,  # e.g., [0,1] to restrict
    mnist_shuffle: bool = True,
    mnist_seed: Optional[int] = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Returns (Y_train, Y_real) as [N,D] on device/dtype.

    For "mnist_pixels": Y_train is drawn from the MNIST training split,
    Y_real is drawn from the MNIST test split (both flattened to [N, 784]).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "2moons":
        make = lambda n: make_2moons(n, noise=moons_noise, device=device, dtype=dtype)
        return make(n_train), make(n_real)
    elif name == "2circles":
        make = lambda n: make_2circles(n, noise=circles_noise, factor=circles_factor, device=device, dtype=dtype)
        return make(n_train), make(n_real)
    elif name == "line":
        make = lambda n: make_line(n, width=line_width, xspan=line_xspan, device=device, dtype=dtype)
        return make(n_train), make(n_real)
    elif name == "sine":
        make = lambda n: make_sine(n, amp=sine_amp, noise=sine_noise, xspan=sine_xspan, device=device, dtype=dtype)
        return make(n_train), make(n_real)
    elif name == "circles_pixels":
        make = lambda n: make_circles_pixels(n, image_size=image_size, noise_std=pix_noise_std,
                                             thickness=pix_thickness, vary_center=pix_vary_center,
                                             device=device, dtype=dtype)
        return make(n_train), make(n_real)
    elif name == "mnist_pixels":
        Y_tr = make_mnist_pixels(n_train, split="train", data_root=mnist_data_root,
                                 classes=mnist_classes, shuffle=mnist_shuffle, seed=mnist_seed,
                                 device=device, dtype=dtype)
        Y_re = make_mnist_pixels(n_real,  split="test",  data_root=mnist_data_root,
                                 classes=mnist_classes, shuffle=mnist_shuffle, seed=mnist_seed,
                                 device=device, dtype=dtype)
        return Y_tr, Y_re
    else:
        raise ValueError(f"Unknown dataset: {name}")
