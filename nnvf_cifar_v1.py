#!/usr/bin/env python
"""
check_memorization_cifar10_flowmatch_nn.py

Train a neural velocity field with Flow Matching (Lipman et al.)
on CIFAR-10 in pixel space, then run a memorization check:

- Train: regress v_theta(x_t, t) to the true velocity (x1 - x0)
  along a linear interpolation between x0 ~ N(0, I) and x1 ~ p_data.
- Sampling: integrate dx/dt = v_theta(x, t) from t=0 to t=1 starting
  from Gaussian noise.
- Memorization: for each generated image, compute the L2 distance
  to its nearest training image in pixel space; plot histogram,
  show most-different pairs, and estimate average train–train distance.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # data
    n_train: int | None = None      # None -> full CIFAR-10 train (50k)
    img_size: int = 32
    num_channels: int = 3

    # flow-matching training
    batch_size: int = 128
    num_epochs: int = 40            # adjust as needed
    lr: float = 2e-4
    weight_decay: float = 0.0

    # sampling
    n_gen: int = 1000
    ode_steps: int = 64

    # memorization visualization
    n_show: int = 8
    n_pairs_for_train_dist: int = 200_000

    # randomness
    seed: int = 123


C = Config()


# ============================================================
# Data loading (pixel space)
# ============================================================

def load_cifar10_train(c: Config):
    """
    Returns:
      dataset: a torch Dataset of CIFAR-10 train, ToTensor() in [0,1]
      img_shape: (C,H,W)
    """
    tfm = transforms.ToTensor()
    ds_full = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=tfm,
    )
    if c.n_train is not None and c.n_train < len(ds_full):
        ds = Subset(ds_full, range(c.n_train))
    else:
        ds = ds_full
    # Peek one sample to get shape
    x0, _ = ds[0]
    img_shape = x0.shape  # (3,32,32)
    return ds, img_shape


def dataset_to_flat_tensor(ds, c: Config) -> Tensor:
    """
    Load entire dataset into a [N, D] tensor on device.
    """
    xs = []
    for i in range(len(ds)):
        x, _ = ds[i]  # [C,H,W] in [0,1]
        xs.append(x)
    X = torch.stack(xs, dim=0)  # [N,C,H,W]
    X_flat = X.view(X.size(0), -1)
    return X_flat.to(c.device, c.dtype)


# ============================================================
# Neural velocity field: v_theta(x_t, t)
# ============================================================

class TimeEmbedding(nn.Module):
    """
    Simple sinusoidal time embedding (like diffusion models).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        # frequencies
        self.register_buffer(
            "freqs",
            torch.exp(
                torch.linspace(
                    math.log(1.0),
                    math.log(1000.0),
                    half
                )
            ),
            persistent=False,
        )

        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        t: [B] in [0,1]
        returns: [B, dim]
        """
        # shape [B, 1]
        t = t.unsqueeze(-1)
        # [B, half]
        arg = t * self.freqs.unsqueeze(0) * 2.0 * math.pi
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return self.proj(emb)


class SimpleFMUNet(nn.Module):
    """
    Minimal UNet-ish CNN for CIFAR-10 flow matching.
    Takes x in [-1,1], t in [0,1], predicts velocity field v(x,t)
    of same shape as x.
    """

    def __init__(self, in_channels=3, base_channels=128, time_dim=256):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.GroupNorm(8, cout),
                nn.SiLU(),
            )

        self.down1 = block(in_channels, base_channels)
        self.down2 = block(base_channels, base_channels * 2)
        self.down3 = block(base_channels * 2, base_channels * 2)

        self.mid = block(base_channels * 2, base_channels * 2)

        self.up3 = block(base_channels * 2 + base_channels * 2, base_channels * 2)
        self.up2 = block(base_channels * 2 + base_channels, base_channels)
        self.up1 = block(base_channels + in_channels, base_channels)

        self.out = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

        # time embeddings projected into feature dims
        self.to_t1 = nn.Linear(time_dim, base_channels + in_channels)
        self.to_t2 = nn.Linear(time_dim, base_channels * 2)
        self.to_t3 = nn.Linear(time_dim, base_channels * 2)

    def add_time(self, h: Tensor, t_emb: Tensor, proj: nn.Linear):
        """
        Add projected time embedding to feature map h.
        h: [B,C,H,W], t_emb: [B,time_dim]
        proj: Linear(time_dim->C)
        """
        B, C, H, W = h.shape
        t_feat = proj(t_emb).view(B, C, 1, 1)
        return h + t_feat

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        x: [B,3,32,32] in [-1,1]
        t: [B] in [0,1]
        """
        t_emb = self.time_emb(t)  # [B,time_dim]

        # Encoder
        h1 = self.down1(x)                              # [B,C,32,32]
        h2 = nn.functional.avg_pool2d(h1, 2)           # [B,C,16,16]
        h2 = self.down2(h2)                            # [B,2C,16,16]
        h2 = self.add_time(h2, t_emb, self.to_t2)
        h3 = nn.functional.avg_pool2d(h2, 2)           # [B,2C,8,8]
        h3 = self.down3(h3)                            # [B,2C,8,8]
        h3 = self.add_time(h3, t_emb, self.to_t3)

        # Mid
        m = self.mid(h3)                               # [B,2C,8,8]

        # Decoder
        u3 = nn.functional.interpolate(m, scale_factor=2, mode="nearest")   # [B,2C,16,16]
        u3 = torch.cat([u3, h2], dim=1)                                   # [B,4C,16,16]
        u3 = self.up3(u3)                                                 # [B,2C,16,16]

        u2 = nn.functional.interpolate(u3, scale_factor=2, mode="nearest")  # [B,2C,32,32]
        u2 = torch.cat([u2, h1], dim=1)                                    # [B,3C,32,32]
        u2 = self.up2(u2)                                                  # [B,C,32,32]

        u1 = torch.cat([u2, x], dim=1)                                     # [B,C+3,32,32]
        u1 = self.add_time(u1, t_emb, self.to_t1)
        u1 = self.up1(u1)                                                  # [B,C,32,32]

        v = self.out(u1)                                                   # [B,3,32,32]
        return v


# ============================================================
# Flow Matching training utilities
# ============================================================

def preprocess_x(x: Tensor) -> Tensor:
    """
    Map x from [0,1] to [-1,1] for the model.
    """
    return x * 2.0 - 1.0


def postprocess_x(x: Tensor) -> Tensor:
    """
    Map x from [-1,1] back to [0,1] for visualization / distances.
    """
    return (x + 1.0) / 2.0


def sample_fm_batch(
    X_train_flat: Tensor,
    batch_size: int,
    img_shape: Tuple[int, int, int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Sample a Flow Matching training batch.

    - x1 ~ data
    - x0 ~ N(0, I)
    - t ~ Uniform[0,1]

    Path: x_t = (1 - t) x0 + t x1
    Velocity: v(x_t) = x1 - x0 (constant in t for linear path)
    """
    N, D = X_train_flat.shape
    idx = torch.randint(0, N, (batch_size,), device=device)
    x1 = X_train_flat[idx]                         # [B,D] in [0,1]
    x1 = x1.view(-1, *img_shape)                   # [B,3,32,32]
    x1 = preprocess_x(x1)                          # [-1,1]

    # base noise
    x0 = torch.randn_like(x1)

    t = torch.rand(batch_size, device=device)      # [B]

    # broadcast t to spatial dims
    t_view = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * x1        # [B,3,32,32]
    v_target = x1 - x0                             # [B,3,32,32]

    return x_t, t, v_target


def train_flow_matching(
    model: nn.Module,
    X_train_flat: Tensor,
    img_shape: Tuple[int, int, int],
    c: Config,
):
    device = torch.device(c.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=c.lr,
        weight_decay=c.weight_decay,
    )

    N = X_train_flat.size(0)
    steps_per_epoch = math.ceil(N / c.batch_size)
    total_steps = c.num_epochs * steps_per_epoch
    print(f"Training flow-matching model for ~{total_steps} steps.")

    step = 0
    for epoch in range(c.num_epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            x_t, t, v_target = sample_fm_batch(
                X_train_flat, c.batch_size, img_shape, device
            )

            v_pred = model(x_t, t)
            loss = ((v_pred - v_target) ** 2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        epoch_loss /= steps_per_epoch
        print(f"[Epoch {epoch+1}/{c.num_epochs}] loss = {epoch_loss:.6f}")

    print("Training complete.")


# ============================================================
# Sampling via ODE integration
# ============================================================

@torch.no_grad()
def generate_flow_matching(
    model: nn.Module,
    n_samples: int,
    img_shape: Tuple[int, int, int],
    c: Config,
) -> Tensor:
    """
    Integrate dx/dt = v_theta(x,t) from t=0 to t=1 with Euler steps.
    Returns flattened [N, D] in [0,1].
    """
    device = torch.device(c.device)
    model.eval()

    Cx, H, W = img_shape
    D = Cx * H * W

    # x0 ~ N(0, I) in model space [-1,1] is fine (we'll clamp later)
    x = torch.randn(n_samples, Cx, H, W, device=device)

    n_steps = c.ode_steps
    dt = 1.0 / n_steps

    for k in range(n_steps):
        t = torch.full((n_samples,), (k + 0.5) * dt, device=device)
        v = model(x, t)
        x = x + dt * v

    # map back to [0,1]
    x = postprocess_x(x).clamp(0.0, 1.0)
    X_flat = x.view(n_samples, D)
    return X_flat


# ============================================================
# Memorization diagnostics (same as before, plus train-train)
# ============================================================

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
    dist_mat = torch.cdist(X_gen_flat, Y_train_flat)
    nn_dists, nn_idx = dist_mat.min(dim=1)
    return nn_dists, nn_idx


@torch.no_grad()
def estimate_train_pairwise_distance(
    Y_train_flat: Tensor,
    n_pairs: int,
) -> Tuple[float, Tensor]:
    """
    Estimate the average Euclidean distance between *distinct* training samples.
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


def plot_distance_histogram(nn_dists: Tensor, img_shape: Tuple[int,int,int]):
    D = math.prod(img_shape)
    nn_rms = nn_dists.cpu() / math.sqrt(D)

    plt.figure(figsize=(6,4))
    plt.hist(nn_rms.numpy(), bins=40, alpha=0.8, edgecolor="black")
    plt.xlabel("RMS pixel-wise distance to nearest training image")
    plt.ylabel("Count")
    plt.title("CIFAR-10 Flow Matching NN distances")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("fm_memorization_hist_cifar10.png", dpi=150)
    print("Saved histogram: fm_memorization_hist_cifar10.png")

    print("Summary stats (RMS distances):")
    for q in [0, 25, 50, 75, 90, 95, 99]:
        val = torch.quantile(nn_rms, q/100.0).item()
        print(f"  {q:2d}%-quantile: {val:.4f}")
    print(f"  mean: {nn_rms.mean().item():.4f}")
    print(f"  min:  {nn_rms.min().item():.4f}")
    print(f"  max:  {nn_rms.max().item():.4f}")


def plot_comparison_histogram(
    nn_dists: Tensor,
    train_pair_dists: Tensor,
    img_shape: Tuple[int,int,int],
):
    D = math.prod(img_shape)
    gen_rms = nn_dists.cpu() / math.sqrt(D)
    train_rms = train_pair_dists.cpu() / math.sqrt(D)

    plt.figure(figsize=(6,4))
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
    plt.title("CIFAR-10: Flow Matching distance comparison")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("fm_memorization_hist_comparison_cifar10.png", dpi=150)
    print("Saved comparison histogram: fm_memorization_hist_comparison_cifar10.png")


def show_side_by_side_examples(
    X_gen_flat: Tensor,
    Y_train_flat: Tensor,
    nn_idx: Tensor,
    nn_dists: Tensor,
    img_shape: Tuple[int,int,int],
    n_show: int,
):
    """
    Show generated images with the *largest* nearest-neighbor distance
    (most different from training set) side-by-side with their
    closest training neighbors.
    """
    Cx, H, W = img_shape
    X_gen = X_gen_flat.view(-1, Cx, H, W).cpu().clamp(0.0, 1.0)
    Y_train = Y_train_flat.view(-1, Cx, H, W).cpu().clamp(0.0, 1.0)

    n_show = min(n_show, X_gen.size(0))
    top_dists, top_idx = torch.topk(nn_dists, k=n_show, largest=True)

    fig, axes = plt.subplots(n_show, 2, figsize=(4, 2*n_show))
    if n_show == 1:
        axes = axes[None, :]

    for row, gidx in enumerate(top_idx):
        gidx = gidx.item()
        tidx = nn_idx[gidx].item()
        gen_img = X_gen[gidx]
        train_img = Y_train[tidx]

        ax = axes[row, 0]
        ax.imshow(gen_img.permute(1, 2, 0).numpy())
        ax.set_title(f"Generated\n(d={top_dists[row].item():.4f})")
        ax.axis("off")

        ax = axes[row, 1]
        ax.imshow(train_img.permute(1, 2, 0).numpy())
        ax.set_title(f"Nearest train (idx={tidx})")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("fm_memorization_examples_cifar10_most_different.png", dpi=150)
    print("Saved side-by-side examples: fm_memorization_examples_cifar10_most_different.png")


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(C.seed)
    device = torch.device(C.device)
    print(f"Using device: {device}")

    # 1. Load CIFAR-10
    print("Loading CIFAR-10 training data...")
    ds_train, img_shape = load_cifar10_train(C)
    print(f"Train set size: {len(ds_train)}, image shape: {img_shape}")

    # 2. Move training data to flat tensor (for sampling batches & NN search)
    print("Converting train dataset to flat tensor...")
    Y_train_flat = dataset_to_flat_tensor(ds_train, C)   # [N,D]

    # 3. Initialize Flow Matching model
    print("Initializing flow-matching neural velocity field...")
    model = SimpleFMUNet(
        in_channels=img_shape[0],
        base_channels=128,
        time_dim=256,
    )

    # 4. Train Flow Matching model
    train_flow_matching(model, Y_train_flat, img_shape, C)

    # 4b. Estimate average train–train distance (baseline)
    print("Estimating average pairwise distance between training samples...")
    mean_train_dist, train_pair_dists = estimate_train_pairwise_distance(
        Y_train_flat,
        n_pairs=C.n_pairs_for_train_dist,
    )
    D = math.prod(img_shape)
    mean_train_rms = mean_train_dist / math.sqrt(D)
    print(f"Estimated mean train–train distance (L2):  {mean_train_dist:.4f}")
    print(f"Estimated mean train–train distance (RMS): {mean_train_rms:.4f}")

    # 5. Generate samples via learned flow
    print(f"Generating {C.n_gen} samples with learned flow (steps={C.ode_steps})...")
    X_gen_flat = generate_flow_matching(
        model,
        n_samples=C.n_gen,
        img_shape=img_shape,
        c=C,
    )

    # 6. Nearest-neighbor distances
    print("Computing nearest-neighbor distances to training set...")
    nn_dists, nn_idx = nearest_neighbor_distances(X_gen_flat, Y_train_flat)

    # 7. Histograms
    plot_distance_histogram(nn_dists, img_shape)
    plot_comparison_histogram(nn_dists, train_pair_dists, img_shape)

    # 8. Side-by-side most-different examples
    show_side_by_side_examples(
        X_gen_flat,
        Y_train_flat,
        nn_idx,
        nn_dists,
        img_shape,
        n_show=C.n_show,
    )


if __name__ == "__main__":
    main()