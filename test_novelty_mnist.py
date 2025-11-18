#!/usr/bin/env python
# check_memorization_mnist_euler1.py

"""
Generate MNIST samples with Euler‑1 at T=0.8 in *latent space* using your EVF
implementation (make_evf_generators), then:

- decode them with a VAE,
- compute nearest‑neighbor distances to the MNIST training set in *pixel space*,
- plot a histogram of distances,
- show side‑by‑side examples of generated images and their nearest neighbors.

This mirrors the CIFAR script's behavior but uses Euler‑1 instead of the ODE
integrator, and MNIST instead of CIFAR.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional

import torch
from torch import Tensor
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---- import your EVF + VAE utilities (adjust module names if needed) ----
from methods import make_evf_generators
from vae_mnist import VAE

# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# MNIST
N_TRAIN = 4000           # full MNIST train
IMAGE_SIZE = 28
IMG_SHAPE = (1, IMAGE_SIZE, IMAGE_SIZE)

# Generation
LATENT_DIM = 32
VAE_CKPT   = f"checkpoints/vae_mnist_lat{LATENT_DIM}.pt"

N_GEN   = 4000            # number of generated samples for histogram
N_SHOW  = 8                # side‑by‑side examples

EULER_T = 0.8              # Euler‑1 time T=0.8

RANDOM_SEED = 12


# -----------------------------
# Data loading (MNIST, pixel space)
# -----------------------------
def load_mnist_train_flat(n_train: Optional[int] = None) -> Tuple[Tensor, Tuple[int,int,int]]:
    """
    Load MNIST train set, return flattened tensors in [0,1].
    Returns:
        Y_train_flat: [N, 1*28*28] on DEVICE, DTYPE
        img_shape:    (1, 28, 28)
    """
    transform = transforms.ToTensor()  # -> [0,1], shape (1,28,28)
    ds = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    if n_train is None or n_train > len(ds):
        n_train = len(ds)

    imgs = []
    for i in range(n_train):
        x, _ = ds[i]       # x: [1, 28, 28]
        imgs.append(x)

    X = torch.stack(imgs, dim=0)               # [N,1,28,28]
    img_shape = X.shape[1:]                    # (1,28,28)
    X_flat = X.view(X.size(0), -1)             # [N, 784]
    return X_flat.to(DEVICE, DTYPE), img_shape


# -----------------------------
# VAE helpers
# -----------------------------
def flat_to_img(flat: Tensor) -> Tensor:
    """
    Convert [N, 784] tensor in [0,1] to [N,1,28,28].
    """
    return flat.view(flat.size(0), 1, IMAGE_SIZE, IMAGE_SIZE).clamp(0.0, 1.0)


@torch.no_grad()
def encode_pixels_to_latents(vae: VAE, flat: Tensor) -> Tensor:
    """
    Encode [N, 784] pixels into [N, LATENT_DIM] latent mean vectors.
    """
    imgs = flat_to_img(flat).to(DEVICE)
    mu, logvar = vae.encode(imgs)
    return mu  # keep on DEVICE


@torch.no_grad()
def decode_latents_to_pixels(vae: VAE, z: Tensor) -> Tensor:
    """
    Decode [N, LATENT_DIM] latent vectors into [N, 784] pixels.
    """
    z = z.to(DEVICE)
    x = vae.decode(z)
    return x.view(z.size(0), -1)


# -----------------------------
# Memorization diagnostics
# (same API as CIFAR script)
# -----------------------------
@torch.no_grad()
def nearest_neighbor_distances(
    X_gen_flat: Tensor,
    Y_train_flat: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Compute nearest‑neighbor distances from generated samples to training set
    in pixel space (Euclidean).
    Returns:
      nn_dists: [N_gen]
      nn_idx:   [N_gen] indices in training set.
    """
    # X_gen_flat: [M,D], Y_train_flat: [N,D]
    dist_mat = torch.cdist(X_gen_flat, Y_train_flat)
    nn_dists, nn_idx = dist_mat.min(dim=1)
    return nn_dists, nn_idx


def plot_distance_histogram(nn_dists: Tensor, img_shape: Tuple[int,int,int]):
    """
    Plot histogram of per‑pixel RMS distances to nearest training neighbor.
    """
    D = math.prod(img_shape)
    nn_rms = nn_dists.cpu() / math.sqrt(D)

    plt.figure(figsize=(6, 4))
    plt.hist(nn_rms.numpy(), bins=40, alpha=0.8, edgecolor="black")
    plt.xlabel("RMS pixel-wise distance to nearest training image")
    plt.ylabel("Count")
    plt.title("MNIST Euler-1 (t=0.8): Nearest-neighbor distances")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("memorization_hist_mnist_euler1_t0.8.png", dpi=150)
    print("Saved histogram: memorization_hist_mnist_euler1_t0.8.png")

    print("Summary stats (RMS distances):")
    for q in [5, 25, 50, 75, 90, 95, 99]:
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
    Show generated images with the *largest* nearest‑neighbor distance (most novel)
    side‑by‑side with their closest training neighbors. Saves to PNG.
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
        ax.imshow(gen_img.permute(1, 2, 0).squeeze(-1), cmap="gray")
        ax.set_title(f"Generated\n(d={top_dists[row].item():.4f})")
        ax.axis("off")

        # nearest neighbor
        ax = axes[row, 1]
        ax.imshow(train_img.permute(1, 2, 0).squeeze(-1), cmap="gray")
        ax.set_title(f"Nearest train (idx={tidx})")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("memorization_examples_mnist_euler1_t0.8_most_different.png", dpi=150)
    print("Saved side-by-side examples: memorization_examples_mnist_euler1_t0.8_most_different.png")


@torch.no_grad()
def estimate_train_pairwise_distance(
    Y_train_flat: Tensor,
    n_pairs: int = 200_000,
) -> Tuple[float, Tensor]:
    """
    Estimate the average Euclidean distance between *distinct* training samples
    in pixel space by sampling random pairs (i != j).
    """
    N, D = Y_train_flat.shape
    device = Y_train_flat.device

    i_idx = torch.randint(0, N, (n_pairs,), device=device)
    j_idx = torch.randint(0, N, (n_pairs,), device=device)
    same = (i_idx == j_idx)
    while same.any():
        j_idx[same] = torch.randint(0, N, (same.sum(),), device=device)
        same = (i_idx == j_idx)

    x_i = Y_train_flat[i_idx]   # [n_pairs, D]
    x_j = Y_train_flat[j_idx]   # [n_pairs, D]

    dists = torch.norm(x_i - x_j, dim=1)  # [n_pairs]
    return dists.mean().item(), dists


@torch.no_grad()
def train_self_nearest_neighbor_distances(
    Y_train_flat: Tensor,
) -> Tensor:
    """
    For each training point, compute the distance to its nearest *other*
    training point (exclude self‑distance).
    Returns:
        nn_dists: [N] tensor of nearest‑neighbor L2 distances in pixel space.
    """
    # Y_train_flat: [N, D]
    dist_mat = torch.cdist(Y_train_flat, Y_train_flat)  # [N, N]
    N = dist_mat.size(0)
    idx = torch.arange(N, device=Y_train_flat.device)
    dist_mat[idx, idx] = float("inf")  # exclude self by setting diagonal to +inf
    nn_dists, _ = dist_mat.min(dim=1)
    return nn_dists


# -----------------------------
# Main
# -----------------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")

    # 1. Load MNIST training data in pixel space
    print("Loading MNIST training data...")
    Y_train_flat, img_shape = load_mnist_train_flat(N_TRAIN)
    print(f"Train data shape: {Y_train_flat.shape}, image shape: {img_shape}")

    # 1b. Nearest‑neighbor distances *within* the training set
    print("Computing nearest-neighbor distances within training set...")
    train_nn_dists = train_self_nearest_neighbor_distances(Y_train_flat)  # [N_TRAIN]
    D = math.prod(img_shape)
    train_nn_rms = train_nn_dists / math.sqrt(D)

    print("Training-set self nearest-neighbor distance quantiles:")
    for q in [5, 25, 50, 75, 90, 95, 99]:
        val_l2 = torch.quantile(train_nn_dists, q / 100.0).item()
        val_rms = torch.quantile(train_nn_rms, q / 100.0).item()
        print(f"  {q:2d}%-quantile: L2={val_l2:.4f}, RMS={val_rms:.4f}")

    # 2. Load VAE
    print(f"Loading VAE with latent_dim={LATENT_DIM} from {VAE_CKPT}...")
    vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    state_dict = torch.load(VAE_CKPT, map_location=DEVICE, weights_only=True)
    vae.load_state_dict(state_dict)
    vae.eval()

    # 3. Encode training images into latent space
    print("Encoding training images into latent space...")
    Z_train = encode_pixels_to_latents(vae, Y_train_flat)  # [N_TRAIN, LATENT_DIM]

    # 4. Build Euler‑1 generator in latent space and generate samples at T=0.8
    print(f"Constructing EVF generators and generating {N_GEN} samples with Euler-1 at t={EULER_T}...")
    gens_latent = make_evf_generators(Z_train)

    @torch.no_grad()
    def generate_euler_one(t: float, n_samples: int) -> Tensor:
        return gens_latent.euler_one_step(t, n_samples)

    Z_gen = generate_euler_one(EULER_T, N_GEN)      # [N_GEN, LATENT_DIM]
    X_gen_flat = decode_latents_to_pixels(vae, Z_gen)  # [N_GEN, 784]

    # 5. Nearest‑neighbor distances in pixel space (like CIFAR script)
    print("Computing nearest-neighbor distances to training set (pixel space)...")
    nn_dists, nn_idx = nearest_neighbor_distances(X_gen_flat, Y_train_flat)

    # 6. Histogram
    print("Plotting histogram of nearest-neighbor distances...")
    plot_distance_histogram(nn_dists, img_shape)

    # 7. Side‑by‑side examples (most different)
    print("Saving side-by-side examples (most different)...")
    show_side_by_side_examples(
        X_gen_flat,
        Y_train_flat,
        nn_idx,
        nn_dists,
        img_shape,
        n_show=N_SHOW,
    )

    print("Done.")


if __name__ == "__main__":
    main()