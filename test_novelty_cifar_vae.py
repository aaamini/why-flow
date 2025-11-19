#!/usr/bin/env python
# test_novelty_cifar_vae.py

"""
Generate CIFAR-10 samples with Euler-1 at T=0.8 in *latent space* using the EVF
implementation (make_evf_generators) with a custom-trained CIFAR-10 VAE, then:

- decode them with the VAE,
- compute nearest-neighbor distances to the CIFAR-10 training set in *pixel space*,
- plot a histogram of distances,
- show side-by-side examples of generated images and their nearest neighbors.

This is the CIFAR-10 equivalent of test_novelty_mnist.py, using a custom-trained
convolutional VAE instead of the pretrained FLUX VAE.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional

import torch
from torch import Tensor
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---- import your EVF + VAE utilities ----
from methods import make_evf_generators
from vae_cifar import VAE
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

# CIFAR-10
N_TRAIN = 50000           # full CIFAR-10 train
IMAGE_SIZE = 32
IMG_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)

# Generation
LATENT_DIM = 64
VAE_CKPT   = f"checkpoints/vae_cifar10_lat{LATENT_DIM}.pt"

N_GEN   = 1000            # number of generated samples for histogram
N_SHOW  = 8               # side-by-side examples

EULER_T = 0.8             # Euler-1 time T=0.8

RANDOM_SEED = 12


# -----------------------------
# Data loading (CIFAR-10, pixel space)
# -----------------------------
def load_cifar10_train_flat(n_train: Optional[int] = None) -> Tuple[Tensor, Tuple[int,int,int]]:
    """
    Load CIFAR-10 train set, return flattened tensors in [0,1].
    Returns:
        Y_train_flat: [N, 3*32*32] on DEVICE, DTYPE
        img_shape:    (3, 32, 32)
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
        x, _ = ds[i]       # x: [3, 32, 32]
        imgs.append(x)

    X = torch.stack(imgs, dim=0)               # [N,3,32,32]
    img_shape = X.shape[1:]                    # (3,32,32)
    X_flat = X.view(X.size(0), -1)             # [N, 3*32*32]
    return X_flat.to(DEVICE, DTYPE), img_shape


# -----------------------------
# VAE helpers
# -----------------------------
def flat_to_img(flat: Tensor) -> Tensor:
    """
    Convert [N, 3*32*32] tensor in [0,1] to [N,3,32,32].
    """
    return flat.view(flat.size(0), 3, IMAGE_SIZE, IMAGE_SIZE).clamp(0.0, 1.0)


@torch.no_grad()
def encode_pixels_to_latents(vae: VAE, flat: Tensor) -> Tensor:
    """
    Encode [N, 3*32*32] pixels into [N, LATENT_DIM] latent mean vectors.
    """
    imgs = flat_to_img(flat).to(DEVICE)
    mu, logvar = vae.encode(imgs)
    return mu  # keep on DEVICE


@torch.no_grad()
def decode_latents_to_pixels(vae: VAE, z: Tensor) -> Tensor:
    """
    Decode [N, LATENT_DIM] latent vectors into [N, 3*32*32] pixels.
    """
    z = z.to(DEVICE)
    x = vae.decode(z)
    return x.view(z.size(0), -1)


# -----------------------------
# Main
# -----------------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")

    # 1. Load CIFAR-10 training data in pixel space
    print("Loading CIFAR-10 training data...")
    Y_train_flat, img_shape = load_cifar10_train_flat(N_TRAIN)
    print(f"Train data shape: {Y_train_flat.shape}, image shape: {img_shape}")

    # 1b. Nearest-neighbor distances *within* the training set
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

    # 4. Build Euler-1 generator in latent space and generate samples at T=0.8
    print(f"Constructing EVF generators and generating {N_GEN} samples with Euler-1 at t={EULER_T}...")
    gens_latent = make_evf_generators(Z_train)

    @torch.no_grad()
    def generate_euler_one(t: float, n_samples: int) -> Tensor:
        return gens_latent.euler_one_step(t, n_samples)

    Z_gen = gens_latent.dode(16, N_GEN)      # [N_GEN, LATENT_DIM]
    X_gen_flat = decode_latents_to_pixels(vae, Z_gen)  # [N_GEN, 3*32*32]

    # 5. Nearest-neighbor distances in pixel space
    print("Computing nearest-neighbor distances to training set (pixel space)...")
    nn_dists, nn_idx = nearest_neighbor_distances(X_gen_flat, Y_train_flat)

    # 6. Histogram
    print("Plotting histogram of nearest-neighbor distances...")
    plot_distance_histogram(
        nn_dists,
        img_shape,
        title=f"CIFAR-10 VAE Euler-1 (t={EULER_T}): Nearest-neighbor distances",
        filename=f"memorization_hist_cifar10_vae_euler1_t{EULER_T}.png",
    )

    # 7. Side-by-side examples (most different)
    print("Saving side-by-side examples (most different)...")
    show_side_by_side_examples(
        X_gen_flat,
        Y_train_flat,
        nn_idx,
        nn_dists,
        img_shape,
        n_show=N_SHOW,
        filename=f"memorization_examples_cifar10_vae_euler1_t{EULER_T}_most_different.png",
    )

    print("Done.")


if __name__ == "__main__":
    main()
