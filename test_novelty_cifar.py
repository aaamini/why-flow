#!/usr/bin/env python
#%%
"""
check_memorization_cifar.py

Empirical EVF on CIFAR-10 in pixel space:
- "Training" = store Y_train (flattened)
- Generation = solve probability-flow ODE with EVF
- Memorization check = distance from each generated sample to its nearest neighbor
  in the training set (in pixel space), plus visual side-by-side examples.
"""

from __future__ import annotations
import math
from typing import Tuple

import torch
from torch import Tensor
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from evf import EmpiricalVectorField, Integrator, uniform_grid
from novelty_metrics import (
    nearest_neighbor_distances,
    estimate_train_pairwise_distance,
    plot_distance_histogram,
    show_side_by_side_examples,
)


# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

N_TRAIN = 50_000         # use full CIFAR-10 train
N_GEN   = 1_000          # number of generated samples for the histogram
N_SHOW  = 8              # number of side-by-side examples to show

ODE_STEPS = 16           # number of integration steps
T1        = 1.0          # final time
INTEG_METHOD = "rk2"     # or "euler" etc., as supported by Integrator

RANDOM_SEED = 123


# -----------------------------
# Data loading (pixel space)
# -----------------------------
def load_cifar10_train_flat(n_train: int | None = None) -> Tuple[Tensor, Tuple[int,int,int]]:
    """
    Load CIFAR-10 train set, return flattened tensors in [0,1].
    """
    transform = transforms.ToTensor()  # -> [0,1], shape (3,32,32)
    ds = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    if n_train is None or n_train > len(ds):
        n_train = len(ds)

    imgs = []
    for i in range(n_train):
        x, _ = ds[i]
        imgs.append(x)

    X = torch.stack(imgs, dim=0)              # [N,3,32,32]
    img_shape = X.shape[1:]                   # (3,32,32)
    X_flat = X.view(X.size(0), -1)            # [N, 3*32*32]
    return X_flat.to(DEVICE, DTYPE), img_shape


# -----------------------------
# Generator: EVF + ODE in pixel space
# -----------------------------
@torch.no_grad()
def generate_evf_ode(
    Y_train_flat: Tensor,
    n_samples: int,
    steps: int = ODE_STEPS,
    t1: float = T1,
    method: str = INTEG_METHOD,
) -> Tensor:
    """
    Probability flow ODE solver in pixel space with empirical EVF.

    Y_train_flat: [N_train, D] tensor (doubles inside EVF)
    Returns X_gen_flat: [n_samples, D]
    """
    field = EmpiricalVectorField(Y_train_flat.double())
    integ = Integrator(field)
    t_grid = uniform_grid(steps, t0=0.0, t1=t1)  # list of time points

    D = Y_train_flat.size(1)
    x0 = torch.randn(n_samples, D, device=DEVICE, dtype=torch.double)

    xT = integ.integrate(
        x0,
        t_grid,
        method=method,
        return_traj=False,
    )  # double

    return xT.float()


# -----------------------------
# Memorization diagnostics (imported from novelty_metrics)
# -----------------------------



#%%

# -----------------------------
# Main
# -----------------------------

torch.manual_seed(RANDOM_SEED)

print(f"Using device: {DEVICE}")

# 1. Load training data in pixel space
print("Loading CIFAR-10 training data...")
Y_train_flat, img_shape = load_cifar10_train_flat(N_TRAIN)
print(f"Train data shape: {Y_train_flat.shape}, image shape: {img_shape}")

# 1b. Estimate average distance between *training* samples
print("Estimating average pairwise distance between training samples...")
mean_train_dist, train_pair_dists = estimate_train_pairwise_distance(
    Y_train_flat,
    n_pairs=200_000,   # adjust if you want more/less accuracy
)
D = math.prod(img_shape)
mean_train_rms = mean_train_dist / math.sqrt(D)
print(f"Estimated mean train-train distance (L2):  {mean_train_dist:.4f}")
print(f"Estimated mean train-train distance (RMS): {mean_train_rms:.4f}")


# 2. Generate samples via EVF probability-flow ODE
print(f"Generating {N_GEN} samples with EVF ODE (steps={ODE_STEPS}, method={INTEG_METHOD})...")
X_gen_flat = generate_evf_ode(Y_train_flat, N_GEN)

# 3. Nearest-neighbor distances in pixel space
print("Computing nearest-neighbor distances to training set...")
nn_dists, nn_idx = nearest_neighbor_distances(X_gen_flat, Y_train_flat)

# 4. Histogram
plot_distance_histogram(
    nn_dists,
    img_shape,
    title="CIFAR-10 EVF ODE: Nearest-neighbor distances",
    filename="memorization_hist_cifar10.png",
)

# 5. Side-by-side examples (most different)
show_side_by_side_examples(
    X_gen_flat,
    Y_train_flat,
    nn_idx,
    nn_dists,
    img_shape,
    n_show=N_SHOW,
    filename="memorization_examples_cifar10_most_different.png",
)
