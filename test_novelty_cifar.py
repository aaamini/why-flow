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
# Memorization diagnostics
# -----------------------------
@torch.no_grad()
def nearest_neighbor_distances(
    X_gen_flat: Tensor,
    Y_train_flat: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Compute nearest-neighbor distances from generated samples to training set
    in pixel space (Euclidean).
    Returns:
      nn_dists: [N_gen]
      nn_idx:   [N_gen] indices in training set.
    """
    # X_gen_flat: [M,D], Y_train_flat: [N,D]
    # torch.cdist can be memory-heavy, but M~1k and N~50k is fine.
    # dist_mat: [M, N]
    dist_mat = torch.cdist(X_gen_flat, Y_train_flat)
    nn_dists, nn_idx = dist_mat.min(dim=1)
    return nn_dists, nn_idx


def plot_distance_histogram(nn_dists: Tensor, img_shape: Tuple[int,int,int]):
    """
    Plot histogram of per-pixel RMS distances to nearest training neighbor.
    """
    D = math.prod(img_shape)
    nn_rms = nn_dists.cpu() / math.sqrt(D)

    plt.figure(figsize=(6,4))
    plt.hist(nn_rms.numpy(), bins=40, alpha=0.8, edgecolor="black")
    plt.xlabel("RMS pixel-wise distance to nearest training image")
    plt.ylabel("Count")
    plt.title("CIFAR-10 EVF ODE: Nearest-neighbor distances")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("memorization_hist_cifar10.png", dpi=150)
    print("Saved histogram: memorization_hist_cifar10.png")

    print("Summary stats (RMS distances):")
    for q in [0, 25, 50, 75, 90, 95, 99]:
        val = torch.quantile(nn_rms, q/100.0).item()
        print(f"  {q:2d}%-quantile: {val:.4f}")
    print(f"  mean: {nn_rms.mean().item():.4f}")
    print(f"  min:  {nn_rms.min().item():.4f}")
    print(f"  max:  {nn_rms.max().item():.4f}")


def show_side_by_side_examples(
    X_gen_flat: Tensor,
    Y_train_flat: Tensor,
    nn_idx: Tensor,
    nn_dists: Tensor,
    img_shape: Tuple[int,int,int],
    n_show: int = N_SHOW,
):
    """
    Show generated images with the *largest* nearest-neighbor distance
    (i.e., most different from training set) side-by-side with their
    closest training neighbors. Saves to PNG.
    """
    C, H, W = img_shape
    X_gen = X_gen_flat.view(-1, C, H, W).cpu().clamp(0.0, 1.0)
    Y_train = Y_train_flat.view(-1, C, H, W).cpu().clamp(0.0, 1.0)

    n_show = min(n_show, X_gen.size(0))

    # indices of the n_show samples with *largest* distance to training set
    top_dists, top_idx = torch.topk(nn_dists, k=n_show, largest=True)

    fig, axes = plt.subplots(n_show, 2, figsize=(4, 2*n_show))
    if n_show == 1:
        axes = axes[None, :]  # make 2D for consistency

    for row, gidx in enumerate(top_idx):
        gidx = gidx.item()
        tidx = nn_idx[gidx].item()
        gen_img = X_gen[gidx]
        train_img = Y_train[tidx]

        # generated
        ax = axes[row, 0]
        ax.imshow(gen_img.permute(1, 2, 0).numpy())
        ax.set_title(f"Generated\n(d={top_dists[row].item():.4f})")
        ax.axis("off")

        # nearest neighbor
        ax = axes[row, 1]
        ax.imshow(train_img.permute(1, 2, 0).numpy())
        ax.set_title(f"Nearest train (idx={tidx})")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("memorization_examples_cifar10_most_different.png", dpi=150)
    print("Saved side-by-side examples: memorization_examples_cifar10_most_different.png")

@torch.no_grad()
def estimate_train_pairwise_distance(
    Y_train_flat: Tensor,
    n_pairs: int = 200_000,
) -> float:
    """
    Estimate the average Euclidean distance between *distinct* training samples
    in pixel space by sampling random pairs (i != j).

    Returns:
        mean_dist: scalar (in the same units as nn_dists, i.e. raw L2 distance).
    """
    N, D = Y_train_flat.shape
    device = Y_train_flat.device

    # sample indices with replacement; enforce i != j
    i_idx = torch.randint(0, N, (n_pairs,), device=device)
    j_idx = torch.randint(0, N, (n_pairs,), device=device)
    same = (i_idx == j_idx)
    while same.any():
        j_idx[same] = torch.randint(0, N, (same.sum(),), device=device)
        same = (i_idx == j_idx)

    x_i = Y_train_flat[i_idx]   # [n_pairs, D]
    x_j = Y_train_flat[j_idx]   # [n_pairs, D]

    # Euclidean distances
    dists = torch.norm(x_i - x_j, dim=1)  # [n_pairs]
    return dists.mean().item(), dists

#%%

# -----------------------------
# Main
# -----------------------------
def main():
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
    plot_distance_histogram(nn_dists, img_shape)

    # 5. Side-by-side examples (most different)
    show_side_by_side_examples(
        X_gen_flat,
        Y_train_flat,
        nn_idx,
        nn_dists,
        img_shape,
        n_show=N_SHOW,
    )

if __name__ == "__main__":
    main()