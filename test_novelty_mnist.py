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
from novelty_metrics import (
    nearest_neighbor_distances,
    estimate_train_pairwise_distance,
    train_self_nearest_neighbor_distances,
    plot_distance_histogram,
    show_side_by_side_examples,
)

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
# Memorization diagnostics (imported from novelty_metrics)
# -----------------------------


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
    plot_distance_histogram(
        nn_dists,
        img_shape,
        title="MNIST Euler-1 (t=0.8): Nearest-neighbor distances",
        filename="memorization_hist_mnist_euler1_t0.8.png",
    )

    # 7. Side‑by‑side examples (most different)
    print("Saving side-by-side examples (most different)...")
    show_side_by_side_examples(
        X_gen_flat,
        Y_train_flat,
        nn_idx,
        nn_dists,
        img_shape,
        n_show=N_SHOW,
        filename="memorization_examples_mnist_euler1_t0.8_most_different.png",
        cmap="gray",
    )

    print("Done.")


if __name__ == "__main__":
    main()