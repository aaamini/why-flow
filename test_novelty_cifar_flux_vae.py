#!/usr/bin/env python
# test_novelty_cifar_flux_vae.py

"""
Generate CIFAR-10 samples with Euler-1 at T=0.8 in *latent space* using the EVF
implementation (make_evf_generators) with a pretrained FLUX VAE, then:

- decode them with the FLUX VAE,
- compute nearest-neighbor distances to the CIFAR-10 training set in *pixel space*,
- plot a histogram of distances,
- show side-by-side examples of generated images and their nearest neighbors.

This mirrors the MNIST script's behavior but uses CIFAR-10 and the pretrained FLUX VAE
instead of a custom-trained VAE.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional, Callable

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL

# ---- import your EVF + novelty metrics utilities ----
from methods import make_evf_generators
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

# FLUX VAE settings
VAE_IN_SIZE = 256         # FLUX VAE expects larger images, we'll upsample
ENCODER_BATCH_SIZE = 64   # batch size for encoding/decoding

# Generation
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
# FLUX VAE helpers
# -----------------------------
def vae_downscale_factor(vae):
    """Compute the downscale factor of the VAE encoder."""
    n = len(vae.config.down_block_types)
    return 2 ** max(0, n - 1)


def build_flux_vae_encoder(
    image_size: int,
    encoder_batch_size: int,
    vae_in_size: int = 256,
) -> Tuple[AutoencoderKL, Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """
    Loads the pretrained FLUX VAE and returns the VAE plus two callables:
      - encode(flat_imgs) -> flattened latents [N, D_lat]
      - decode(flat_latents) -> flattened images matching original image_size [N, image_size*image_size*3]
    
    Note: Expects the FLUX VAE to be available in ./flux_vae directory.
    """
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        "./flux_vae",
        torch_dtype=dtype,
    ).to(DEVICE)
    vae.eval()

    scaling_factor = vae.config.scaling_factor
    shift_factor = vae.config.shift_factor
    latent_channels = vae.config.latent_channels
    down_factor = vae_downscale_factor(vae)
    z_h = vae_in_size // down_factor
    z_w = vae_in_size // down_factor
    model_dtype = next(vae.parameters()).dtype

    @torch.no_grad()
    def encode(flat_imgs: Tensor) -> Tensor:
        """
        Encode flattened images [N, 3*image_size*image_size] to flattened latents [N, D_lat].
        """
        outputs = []
        N = flat_imgs.size(0)
        for start in range(0, N, encoder_batch_size):
            end = min(N, start + encoder_batch_size)
            x = flat_imgs[start:end].to(device=DEVICE, dtype=model_dtype)
            n = x.size(0)
            # Reshape to [n, 3, image_size, image_size]
            x = x.view(n, 3, image_size, image_size)
            # Upsample to VAE input size
            x = F.interpolate(x, size=(vae_in_size, vae_in_size), mode="bilinear", align_corners=False)
            # Normalize to [-1, 1]
            x = x * 2.0 - 1.0
            # Encode
            latents = vae.encode(x).latent_dist.sample()
            # Apply FLUX scaling and shift
            latents = latents * scaling_factor + shift_factor
            # Flatten and store
            outputs.append(latents.detach().to(torch.float32).view(n, -1))
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def decode(flat_latents: Tensor) -> Tensor:
        """
        Decode flattened latents [N, D_lat] to flattened images [N, 3*image_size*image_size].
        """
        outputs = []
        N = flat_latents.size(0)
        for start in range(0, N, encoder_batch_size):
            end = min(N, start + encoder_batch_size)
            z_flat = flat_latents[start:end].to(DEVICE)
            n = z_flat.size(0)
            # Reshape to [n, latent_channels, z_h, z_w]
            z = z_flat.view(n, latent_channels, z_h, z_w).to(device=DEVICE, dtype=model_dtype)
            # Inverse FLUX scaling and shift
            z = (z - shift_factor) / scaling_factor
            # Decode
            imgs = vae.decode(z).sample
            # Normalize to [0, 1]
            imgs = (imgs + 1.0) / 2.0
            # Downsample to original image size
            imgs_small = F.interpolate(imgs, size=(image_size, image_size), mode="bilinear", align_corners=False)
            # Flatten and store
            outputs.append(imgs_small.view(n, -1).detach().to(torch.float32))
        return torch.cat(outputs, dim=0)

    return vae, encode, decode


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

    # 2. Load FLUX VAE
    print(f"Loading FLUX VAE from ./flux_vae...")
    vae, encode_fn, decode_fn = build_flux_vae_encoder(
        image_size=IMAGE_SIZE,
        encoder_batch_size=ENCODER_BATCH_SIZE,
        vae_in_size=VAE_IN_SIZE,
    )

    # 3. Encode training images into latent space
    print("Encoding training images into latent space...")
    Z_train = encode_fn(Y_train_flat)  # [N_TRAIN, D_lat]
    print(f"Latent shape: {Z_train.shape}")

    # 4. Build Euler-1 generator in latent space and generate samples at T=0.8
    print(f"Constructing EVF generators and generating {N_GEN} samples with Euler-1 at t={EULER_T}...")
    gens_latent = make_evf_generators(Z_train)

    # @torch.no_grad()
    # def generate_euler_one(t: float, n_samples: int) -> Tensor:
    #     return gens_latent.euler_one_step(t, n_samples)
    
    Z_gen = gens_latent.dode(16, N_GEN)
    # Z_gen = generate_euler_one(EULER_T, N_GEN)      # [N_GEN, D_lat]
    X_gen_flat = decode_fn(Z_gen)  # [N_GEN, 3*32*32]

    # 5. Nearest-neighbor distances in pixel space
    print("Computing nearest-neighbor distances to training set (pixel space)...")
    nn_dists, nn_idx = nearest_neighbor_distances(X_gen_flat, Y_train_flat)

    # 6. Histogram
    print("Plotting histogram of nearest-neighbor distances...")
    plot_distance_histogram(
        nn_dists,
        img_shape,
        title=f"CIFAR-10 FLUX VAE Euler-1 (t={EULER_T}): Nearest-neighbor distances",
        filename=f"memorization_hist_cifar10_flux_vae_euler1_t{EULER_T}.png",
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
        filename=f"memorization_examples_cifar10_flux_vae_euler1_t{EULER_T}_most_different.png",
    )

    print("Done.")


if __name__ == "__main__":
    main()
