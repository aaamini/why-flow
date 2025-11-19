#!/usr/bin/env python
"""
novelty_metrics.py

Shared utilities for novelty/memorization testing:
- Nearest-neighbor distance computations
- Training set pairwise distance estimation
- Visualization functions (histograms, side-by-side comparisons)

These functions are used across multiple novelty testing scripts
(MNIST, CIFAR-10, etc.) to evaluate whether generated samples
are memorized from the training set.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional

import torch
from torch import Tensor
import matplotlib.pyplot as plt


@torch.no_grad()
def nearest_neighbor_distances(
    X_gen_flat: Tensor,
    Y_train_flat: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Compute nearest-neighbor distances from generated samples to training set
    in pixel space (Euclidean).
    
    Args:
        X_gen_flat: [N_gen, D] generated samples (flattened)
        Y_train_flat: [N_train, D] training samples (flattened)
    
    Returns:
        nn_dists: [N_gen] L2 distances to nearest training sample
        nn_idx: [N_gen] indices of nearest training samples
    """
    dist_mat = torch.cdist(X_gen_flat, Y_train_flat)
    nn_dists, nn_idx = dist_mat.min(dim=1)
    return nn_dists, nn_idx


@torch.no_grad()
def estimate_train_pairwise_distance(
    Y_train_flat: Tensor,
    n_pairs: int = 200_000,
) -> Tuple[float, Tensor]:
    """
    Estimate the average Euclidean distance between *distinct* training samples
    in pixel space by sampling random pairs (i != j).
    
    Args:
        Y_train_flat: [N, D] training samples (flattened)
        n_pairs: number of random pairs to sample
    
    Returns:
        mean_dist: average L2 distance between training pairs
        dists: [n_pairs] tensor of sampled pairwise distances
    """
    N, D = Y_train_flat.shape
    device = Y_train_flat.device

    i_idx = torch.randint(0, N, (n_pairs,), device=device)
    j_idx = torch.randint(0, N, (n_pairs,), device=device)
    same = (i_idx == j_idx)
    while same.any():
        j_idx[same] = torch.randint(0, N, (same.sum(),), device=device)
        same = (i_idx == j_idx)

    x_i = Y_train_flat[i_idx]
    x_j = Y_train_flat[j_idx]
    dists = torch.norm(x_i - x_j, dim=1)
    return dists.mean().item(), dists


@torch.no_grad()
def train_self_nearest_neighbor_distances(
    Y_train_flat: Tensor,
) -> Tensor:
    """
    For each training point, compute the distance to its nearest *other*
    training point (exclude self-distance).
    
    Args:
        Y_train_flat: [N, D] training samples (flattened)
    
    Returns:
        nn_dists: [N] tensor of nearest-neighbor L2 distances within training set
    """
    dist_mat = torch.cdist(Y_train_flat, Y_train_flat)  # [N, N]
    N = dist_mat.size(0)
    idx = torch.arange(N, device=Y_train_flat.device)
    dist_mat[idx, idx] = float("inf")  # exclude self by setting diagonal to +inf
    nn_dists, _ = dist_mat.min(dim=1)
    return nn_dists


def plot_distance_histogram(
    nn_dists: Tensor,
    img_shape: Tuple[int, int, int],
    title: str = "Nearest-neighbor distances",
    filename: str = "memorization_hist.png",
):
    """
    Plot histogram of per-pixel RMS distances to nearest training neighbor.
    
    Args:
        nn_dists: [N] L2 distances to nearest training samples
        img_shape: (C, H, W) image shape for computing RMS normalization
        title: plot title
        filename: output filename
    """
    D = math.prod(img_shape)
    nn_rms = nn_dists.cpu() / math.sqrt(D)

    plt.figure(figsize=(6, 4))
    plt.hist(nn_rms.numpy(), bins=40, alpha=0.8, edgecolor="black")
    plt.xlabel("RMS pixel-wise distance to nearest training image")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved histogram: {filename}")

    print("Summary stats (RMS distances):")
    for q in [0, 25, 50, 75, 90, 95, 99]:
        val = torch.quantile(nn_rms, q / 100.0).item()
        print(f"  {q:2d}%-quantile: {val:.4f}")
    print(f"  mean: {nn_rms.mean().item():.4f}")
    print(f"  min:  {nn_rms.min().item():.4f}")
    print(f"  max:  {nn_rms.max().item():.4f}")


def plot_comparison_histogram(
    nn_dists: Tensor,
    train_pair_dists: Tensor,
    img_shape: Tuple[int, int, int],
    title: str = "Distance comparison",
    filename: str = "memorization_hist_comparison.png",
):
    """
    Plot overlaid histograms comparing gen-train vs train-train distances.
    
    Args:
        nn_dists: [N_gen] L2 distances from generated to nearest training samples
        train_pair_dists: [N_pairs] L2 distances between random training pairs
        img_shape: (C, H, W) image shape for computing RMS normalization
        title: plot title
        filename: output filename
    """
    D = math.prod(img_shape)
    gen_rms = nn_dists.cpu() / math.sqrt(D)
    train_rms = train_pair_dists.cpu() / math.sqrt(D)

    plt.figure(figsize=(6, 4))
    plt.hist(
        train_rms.numpy(),
        bins=40,
        alpha=0.5,
        edgecolor="black",
        label="train–train (random pairs)",
    )
    plt.hist(
        gen_rms.numpy(),
        bins=40,
        alpha=0.5,
        edgecolor="black",
        label="gen–train (nearest)",
    )
    plt.xlabel("RMS pixel-wise distance")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved comparison histogram: {filename}")


def show_side_by_side_examples(
    X_gen_flat: Tensor,
    Y_train_flat: Tensor,
    nn_idx: Tensor,
    nn_dists: Tensor,
    img_shape: Tuple[int, int, int],
    n_show: int = 8,
    filename: str = "memorization_examples_most_different.png",
    cmap: Optional[str] = None,
):
    """
    Show generated images with the *largest* nearest-neighbor distance
    (most different from training set) side-by-side with their
    closest training neighbors. Saves to PNG.
    
    Args:
        X_gen_flat: [N_gen, D] generated samples (flattened)
        Y_train_flat: [N_train, D] training samples (flattened)
        nn_idx: [N_gen] indices of nearest training samples
        nn_dists: [N_gen] L2 distances to nearest training samples
        img_shape: (C, H, W) image shape
        n_show: number of examples to show
        filename: output filename
        cmap: colormap for grayscale images (e.g., 'gray' for MNIST), None for RGB
    """
    C, H, W = img_shape
    X_gen = X_gen_flat.view(-1, C, H, W).cpu().clamp(0.0, 1.0)
    Y_train = Y_train_flat.view(-1, C, H, W).cpu().clamp(0.0, 1.0)

    n_show = min(n_show, X_gen.size(0))
    top_dists, top_idx = torch.topk(nn_dists, k=n_show, largest=True)

    fig, axes = plt.subplots(n_show, 2, figsize=(4, 2 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for row, gidx in enumerate(top_idx):
        gidx = gidx.item()
        tidx = nn_idx[gidx].item()
        gen_img = X_gen[gidx]
        train_img = Y_train[tidx]

        # generated
        ax = axes[row, 0]
        if C == 1:  # grayscale
            ax.imshow(gen_img.permute(1, 2, 0).squeeze(-1), cmap=cmap)
        else:  # RGB
            ax.imshow(gen_img.permute(1, 2, 0).numpy())
        ax.set_title(f"Generated\n(d={top_dists[row].item():.4f})")
        ax.axis("off")

        # nearest neighbor
        ax = axes[row, 1]
        if C == 1:  # grayscale
            ax.imshow(train_img.permute(1, 2, 0).squeeze(-1), cmap=cmap)
        else:  # RGB
            ax.imshow(train_img.permute(1, 2, 0).numpy())
        ax.set_title(f"Nearest train (idx={tidx})")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved side-by-side examples: {filename}")
