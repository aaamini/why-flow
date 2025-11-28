from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ====== EVF dependencies from the why-flow repo ======
# Requires evf.py from https://github.com/aaamini/why-flow
# - sample_rho_t_empirical(Y_train: Tensor, t: float, n: int) -> Tensor
# - EmpiricalVectorField(Y_train: Tensor) callable: v(t: Tensor, x: Tensor) -> Tensor
# - Integrator(field) with integrate(x0, t_grid, method, return_traj=False) -> Tensor
# - uniform_grid(steps: int, t1: float) -> Tensor
from evf import sample_rho_t_empirical, EmpiricalVectorField, Integrator, uniform_grid


# =========================
# Config
# =========================
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # Data
    data_root = "./data"
    download = False
    batch_size = 512
    num_workers = 0

    # Single digit
    digit = 0

    # EVF settings (pixel space)
    n_train_for_evf = 100     # number of training samples used as EVF seeds (Y_train)
    n_gen = 2000               # number of Euler-one-step generated samples
    t_euler = 0.1              # Euler one-step t in (0,1)

    # Histogram settings
    hist_bins = 50
    max_train_embed = 5000     # subsample refs for speed if needed (None = no limit)
    max_test_embed = 2000
    max_gen_embed = 2000

    # IO
    out_dir = "outputs_pixel"

    # Visualization
    n_show = 8                 # number of example triplets to show (gen vs NN train/test)
    grid_max = 64              # number of generated images to show in grid


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_flatten(x: Tensor) -> Tensor:
    # x: [N,1,28,28] -> [N,784], float32
    return x.view(x.size(0), -1)


def to_images(x_flat: Tensor) -> Tensor:
    # x_flat: [N,784] -> [N,1,28,28]
    return x_flat.view(x_flat.size(0), 1, 28, 28)


@torch.no_grad()
def get_mnist_digit_subset(digit: int, train: bool, cfg: Config) -> Subset:
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root=cfg.data_root, train=train, transform=tfm, download=cfg.download)
    indices = [i for i, (_, y) in enumerate(ds) if int(y) == int(digit)]
    return Subset(ds, indices)


def make_square_grid_count(n: int) -> Tuple[int, int]:
    rows = cols = int(math.sqrt(n))
    if rows * cols < n:
        cols += 1
        if rows * cols < n:
            rows += 1
    return rows, cols


def maybe_subsample(t: Tensor, k: Optional[int]) -> Tensor:
    if k is None or t.size(0) <= k:
        return t
    idx = torch.randperm(t.size(0))[:k]
    return t[idx]


@torch.no_grad()
def compute_nn_min_dist(queries: Tensor, refs: Tensor, chunk_q: int = 512, chunk_r: int = 4096) -> Tensor:
    """
    Min Euclidean distance from each row of queries to the set refs, with batching.
    queries: [Nq, D] CPU float32
    refs:    [Nr, D] CPU float32
    returns: [Nq] CPU float32
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refs_d = refs.to(device)
    out = torch.empty(queries.size(0), dtype=torch.float32)
    q_ptr = 0
    while q_ptr < queries.size(0):
        q_end = min(q_ptr + chunk_q, queries.size(0))
        q_chunk = queries[q_ptr:q_end].to(device)  # [bq, D]
        min_d2 = torch.full((q_chunk.size(0),), float("inf"), device=device)
        r_ptr = 0
        # Precompute q2 once per q_chunk
        q2 = (q_chunk**2).sum(dim=1, keepdim=True)  # [bq,1]
        while r_ptr < refs_d.size(0):
            r_end = min(r_ptr + chunk_r, refs_d.size(0))
            r_chunk = refs_d[r_ptr:r_end]  # [br, D]
            r2 = (r_chunk**2).sum(dim=1).unsqueeze(0)  # [1,br]
            cross = q_chunk @ r_chunk.t()              # [bq,br]
            d2 = q2 + r2 - 2.0 * cross
            d2.clamp_(min=0.0)
            min_d2 = torch.minimum(min_d2, d2.min(dim=1).values)
            r_ptr = r_end
        out[q_ptr:q_end] = min_d2.sqrt().detach().cpu()
        q_ptr = q_end
    return out


# =========================
# EVF generator in pixel space
# =========================
class EVFPixelGenerator:
    def __init__(self, Y_train_pix: Tensor):
        """
        Y_train_pix: [N, D] pixel vectors in [0,1], float32/float64
        Internally, EmpiricalVectorField expects float64 for numerical stability.
        """
        assert Y_train_pix.dim() == 2, "Y_train must be [N, D]"
        self.Y_train64 = Y_train_pix.double().contiguous()
        self.device = self.Y_train64.device

    @torch.no_grad()
    def euler_one_step(self, t: float, n_samp: int) -> Tensor:
        """
        Returns x_gen in pixel space [n_samp, D], clamped to [0,1], float32
        """
        # Sample x_t ~ rho_t(Y_train)
        x_t = sample_rho_t_empirical(self.Y_train64, float(t), int(n_samp))  # float64
        # Build EVF and evaluate v(t, x_t)
        field = EmpiricalVectorField(self.Y_train64)
        t_tensor = x_t.new_full((x_t.size(0), 1), float(t))  # float64
        v = field(t_tensor, x_t)                             # float64
        # Euler backward one-step: x_{t-1} ≈ x_t + (1 - t) v(t, x_t)
        x_prev = x_t + (1.0 - float(t)) * v                  # float64
        x_prev = x_prev.to(dtype=torch.float32)
        # Clamp to valid pixel range
        x_prev.clamp_(0.0, 1.0)
        return x_prev


# =========================
# Main
# =========================
def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)
    print("Using device:", cfg.device)

    # Prepare MNIST digit subset
    train_subset = get_mnist_digit_subset(cfg.digit, train=True, cfg=cfg)
    test_subset = get_mnist_digit_subset(cfg.digit, train=False, cfg=cfg)

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Collect all train/test images as pixel vectors in [0,1]
    def collect_pixels(loader: DataLoader) -> Tensor:
        xs = []
        for x, _ in loader:
            xs.append(to_flatten(x))  # float32
        return torch.cat(xs, dim=0)   # [N,784]

    with torch.no_grad():
        x_train_flat = collect_pixels(train_loader).cpu()  # [N_tr, 784], float32
        x_test_flat = collect_pixels(test_loader).cpu()    # [N_te, 784], float32

    N_tr = x_train_flat.size(0)
    D = x_train_flat.size(1)
    N_te = x_test_flat.size(0)
    print(f"Pixel data | train: {N_tr}x{D} | test: {N_te}x{D}")

    # Pick EVF seeds from training pixels
    assert cfg.n_train_for_evf <= N_tr, "n_train_for_evf exceeds available training samples"
    perm = torch.randperm(N_tr)
    idx_train_evf = perm[:cfg.n_train_for_evf]
    Y_train = x_train_flat[idx_train_evf].to(cfg.device)  # [n_seed, D], float32
    print(f"Selected {cfg.n_train_for_evf} EVF seeds from training set (pixel space).")

    # Build EVF pixel generator and sample with Euler one step at t
    evf_pix = EVFPixelGenerator(Y_train)
    t_euler = float(cfg.t_euler)
    with torch.no_grad():
        x_gen_flat = evf_pix.euler_one_step(t=t_euler, n_samp=cfg.n_gen).cpu()  # [n_gen, D], float32 in [0,1]
    print(f"Generated {x_gen_flat.size(0)} pixel samples via Euler one step at t={t_euler}.")

    # Save generated grid
    n_grid = min(cfg.grid_max, x_gen_flat.size(0))
    rows, cols = make_square_grid_count(n_grid)
    fig = plt.figure(figsize=(cols, rows))
    x_gen_imgs = to_images(x_gen_flat[:n_grid])
    for i in range(n_grid):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(x_gen_imgs[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    t_tag = f"{t_euler:.2f}".replace(".", "")
    grid_path = os.path.join(cfg.out_dir, f"digit{cfg.digit}_gen_grid_pixel_t{t_tag}.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=160)
    plt.close()
    print(f"Saved generated pixel grid to {grid_path}")

    # Example triplets: generated vs nearest train/test in pixel-space
    n_show = min(cfg.n_show, x_gen_flat.size(0))
    if n_show > 0:
        pick_idx = torch.linspace(0, x_gen_flat.size(0) - 1, steps=n_show).long()
        with torch.no_grad():
            # Compute pairwise to get argmin (small n_show; OK to do directly)
            dist_tr = torch.cdist(x_gen_flat[pick_idx], x_train_flat)  # [n_show, N_tr]
            nn_tr_idx = dist_tr.argmin(dim=1)
            dist_te = torch.cdist(x_gen_flat[pick_idx], x_test_flat)   # [n_show, N_te]
            nn_te_idx = dist_te.argmin(dim=1)

            x_tr_nn = x_train_flat[nn_tr_idx]
            x_te_nn = x_test_flat[nn_te_idx]

            x_gen_show = to_images(x_gen_flat[pick_idx])
            x_tr_show = to_images(x_tr_nn)
            x_te_show = to_images(x_te_nn)

        plt.figure(figsize=(6.5, 2.1 * n_show))
        for i in range(n_show):
            ax = plt.subplot(n_show, 3, i * 3 + 1)
            ax.imshow(x_gen_show[i, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title("generated", fontsize=9)
            ax.axis("off")

            ax = plt.subplot(n_show, 3, i * 3 + 2)
            ax.imshow(x_tr_show[i, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title("nearest train", fontsize=9)
            ax.axis("off")

            ax = plt.subplot(n_show, 3, i * 3 + 3)
            ax.imshow(x_te_show[i, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title("nearest test", fontsize=9)
            ax.axis("off")

        ex_path = os.path.join(cfg.out_dir, f"digit{cfg.digit}_examples_pixel_t{t_tag}.png")
        plt.tight_layout()
        plt.savefig(ex_path, dpi=160)
        plt.close()
        print(f"Saved example triplets to {ex_path}")

    # Subsample for speed if desired
    x_train_ref = maybe_subsample(x_train_flat, cfg.max_train_embed)
    x_test_q = maybe_subsample(x_test_flat, cfg.max_test_embed)
    x_gen_q = maybe_subsample(x_gen_flat, cfg.max_gen_embed)

    # Compute NN distances to training pixels
    print("Computing NN distances (test→train, gen→train) in pixel space ...")
    nn_test_to_train = compute_nn_min_dist(x_test_q, x_train_ref) if x_test_q.numel() > 0 else None
    nn_gen_to_train = compute_nn_min_dist(x_gen_q, x_train_ref)

    # Plot overlapped histograms
    plt.figure(figsize=(7.0, 4.5))
    if nn_test_to_train is not None:
        plt.hist(nn_test_to_train.numpy(), bins=cfg.hist_bins, alpha=0.6, density=True,
                 label="Test → Train (NN dist)", color="#1f77b4")
    plt.hist(nn_gen_to_train.numpy(), bins=cfg.hist_bins, alpha=0.6, density=True,
             label="Generated (Euler 1-step) → Train (NN dist)", color="#ff7f0e")
    plt.xlabel("Nearest neighbor distance in pixel space")
    plt.ylabel("Density")
    plt.title(f"Digit {cfg.digit} | NN distance histograms (t={cfg.t_euler})")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(cfg.out_dir, f"digit{cfg.digit}_nn_hist_pixel_t{t_tag}.png")
    plt.savefig(hist_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved NN distance histograms to {hist_path}")

    # Numeric summaries
    if nn_test_to_train is not None:
        gen_med = float(nn_gen_to_train.median())
        test_med = float(nn_test_to_train.median())
        print(f"Median NN dist (pixel) | Generated: {gen_med:.4f} | Test: {test_med:.4f}")

    print("Done (pixel space EVF).")


if __name__ == "__main__":
    main()