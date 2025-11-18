from __future__ import annotations
from typing import Any, Sequence, List, Dict, Tuple, Optional
import math, os

import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from new_metrics import pr_knn, pr_knn_conditioned
from methods import MethodSpec, make_evf_generators
from data import load_dataset
from feature_extractor import InceptionPool3, Whitener

# -------------------------------------------------
# Config
# -------------------------------------------------
DATASET = "mnist_pixels"    # flattened images
N_TRAIN = 2000
N_REAL  = 5000
K       = 3

P_GEN  = 0.95
P_REAL = 0.5

T_VALUES     = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
STEP_VALUES  = [2, 4, 6, 8, 10, 12, 16]
N_EVAL       = N_REAL

# feature-space PR toggles:
MEASURE_IN_FEATURES = True
WHITEN_FEATURES     = False
IMAGE_SIZE          = 28
ENC_BATCH_SIZE      = 128

# VAE config
LATENT_DIM = 32
VAE_CKPT   = f"checkpoints/vae_mnist_lat{LATENT_DIM}.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 1. Load pixel data (Y_train, Y_real)
# -------------------------------------------------
Y_train, Y_real = load_dataset(
    DATASET,
    n_train=N_TRAIN,
    n_real=N_REAL,
    device=device,
    mnist_classes=None,
    mnist_data_root="./data",
)
print("Y_train:", Y_train.shape, "Y_real:", Y_real.shape)  # e.g., [N, 784]

# -------------------------------------------------
# 2. VAE: load and define helpers
# -------------------------------------------------
from vae_mnist import VAE   # <-- change path if needed

vae = VAE(latent_dim=LATENT_DIM).to(device)
vae.load_state_dict(torch.load(VAE_CKPT, map_location=device))
vae.eval()
print(f"Loaded VAE(latent_dim={LATENT_DIM}) from {VAE_CKPT}")

@torch.no_grad()
def flat_pixels_to_imgs(flat: Tensor) -> Tensor:
    # [N, 784] -> [N,1,28,28]
    return flat.view(flat.size(0), 1, IMAGE_SIZE, IMAGE_SIZE).clamp(0, 1)

@torch.no_grad()
def pixels_to_latents(flat: Tensor) -> Tensor:
    """
    flat: [N, 784] in [0,1], on same device as VAE input
    returns mu: [N, LATENT_DIM] on CPU (deterministic embedding)
    """
    imgs = flat_pixels_to_imgs(flat).to(device)
    mu, log_var = vae.encode(imgs)
    return mu.cpu()

@torch.no_grad()
def latents_to_pixels(z: Tensor) -> Tensor:
    """
    z: [N, LATENT_DIM] latent vectors (CPU or GPU)
    returns flat pixels [N, 784] in [0,1] on CPU
    """
    z = z.to(device)
    x_recon = vae.decode(z)              # [N, 1, 28, 28]
    return x_recon.view(z.size(0), -1).cpu()

# -------------------------------------------------
# 3. Encode train/real into latent space
# -------------------------------------------------
print("Encoding Y_train, Y_real to latent space...")
Z_train = pixels_to_latents(Y_train)   # [N_TRAIN, LATENT_DIM]
Z_real  = pixels_to_latents(Y_real)    # [N_REAL,  LATENT_DIM]
print("Z_train:", Z_train.shape, "Z_real:", Z_real.shape)

# -------------------------------------------------
# 4. Feature extractor (Inception) or pixel space features
# -------------------------------------------------
if MEASURE_IN_FEATURES:
    enc = InceptionPool3(device=device)

    @torch.no_grad()
    def to_features_from_pixels(flat: Tensor) -> Tensor:
        # flat [N, 784] -> images -> inception features [N, 2048]
        imgs = flat_pixels_to_imgs(flat)
        return enc(imgs, batch_size=ENC_BATCH_SIZE)   # returns CPU tensor

    # Precompute feature-space Train/Real
    Phi_train = to_features_from_pixels(Y_train)
    Phi_real  = to_features_from_pixels(Y_real)

    if WHITEN_FEATURES:
        wh = Whitener().fit(Phi_real)
        Phi_train = wh.transform(Phi_train)
        Phi_real  = wh.transform(Phi_real)

        @torch.no_grad()
        def gen_to_features(flat: Tensor) -> Tensor:
            Phi = to_features_from_pixels(flat)
            return wh.transform(Phi)
    else:
        @torch.no_grad()
        def gen_to_features(flat: Tensor) -> Tensor:
            return to_features_from_pixels(flat)
else:
    # Use pixel vectors directly as "features"
    Phi_train, Phi_real = Y_train, Y_real

    def gen_to_features(flat: Tensor) -> Tensor:
        return flat

# -------------------------------------------------
# 5. Generators in latent space
# -------------------------------------------------
gens_latent = make_evf_generators(Z_train)
methods_latent: List[MethodSpec] = [
    MethodSpec(
        name="Exact x_t",
        params=T_VALUES,
        generator=gens_latent.exact_xt,
        n_samples=N_EVAL,
    ),
    MethodSpec(
        name="Euler-1",
        params=T_VALUES,
        generator=gens_latent.euler_one_step,
        n_samples=N_EVAL,
    ),
    MethodSpec(
        name="D-ODE (rk2)",
        params=STEP_VALUES,
        generator=lambda s, n: gens_latent.dode(int(s), n, t1=1.0, method="rk2"),
        n_samples=N_EVAL,
    ),
]

VARIANTS = [
    ("vanilla",                      0.0,   0.0),
    (f"gen_novel_{P_GEN}",           P_GEN, 0.0),
    (f"real_novel_{P_REAL}",         0.0,   P_REAL),
    (f"both_gen{P_GEN}_real{P_REAL}", P_GEN, P_REAL),
]

# -------------------------------------------------
# 6. eval_methods: now using latent generators + decoder
# -------------------------------------------------
def eval_methods_latent(methods: List[MethodSpec], k: int) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    # Feature-space tensors for PR
    Train = Phi_train        # features of Y_train
    Real  = Phi_real         # features of Y_real

    # Baseline: Train vs Real for all variants
    for vname, pg, prc in VARIANTS:
        out = pr_knn_conditioned(Real, Train, Train, p_gen=pg, p_real=prc, k=k)
        records.append({
            "method": "Train",
            "param": math.nan,
            "variant": vname,
            "precision": out["precision"],
            "recall": out["recall"],
            "kept_real_frac": out["kept_real_frac"],
            "kept_gen_frac": out["kept_gen_frac"],
        })

    # Methods in latent space
    for m in methods:
        for p in m.params:
            # 1) generate in latent space
            Z_gen = m.generator(p, m.n_samples)   # [N_EVAL, LATENT_DIM]

            # 2) decode to pixels (flattened)
            X_flat = latents_to_pixels(Z_gen)     # [N_EVAL, 784]

            # 3) map to feature space (Inception or identity)
            X_feat = gen_to_features(X_flat)      # [N_EVAL, D_feat]

            # 4) evaluate PR for all variants
            for vname, pg, prc in VARIANTS:
                out = pr_knn_conditioned(Real, X_feat, Train, p_gen=pg, p_real=prc, k=k)
                records.append({
                    "method": m.name,
                    "param": float(p),
                    "variant": vname,
                    "precision": out["precision"],
                    "recall": out["recall"],
                    "kept_real_frac": out["kept_real_frac"],
                    "kept_gen_frac": out["kept_gen_frac"],
                })

    return pd.DataFrame.from_records(records)

# -------------------------------------------------
# 7. Run evaluation, save CSV
# -------------------------------------------------
df = eval_methods_latent(methods_latent, K)

os.makedirs("./pr_outputs", exist_ok=True)
csv_path = "./pr_outputs/pr_sweep_results_latent.csv"
df.to_csv(csv_path, index=False)
print(f"Saved PR sweep CSV: {csv_path}")

# -------------------------------------------------
# 8. Plotting (unchanged logic)
# -------------------------------------------------
COLORS = {
    "Exact x_t": "#1f77b4",
    "Euler-1": "#2ca02c",
    "D-ODE (rk2)": "#d62728",
    "Train": "#7f7f7f",
}

def plot_variant(df: pd.DataFrame, variant: str):
    sub = df[df["variant"] == variant].copy()
    plt.figure(figsize=(5.5, 5.5))
    plt.title(f"PR — {variant}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)

    for m, g in sub.groupby("method"):
        color = COLORS.get(m, None)
        gg = g.sort_values("param")
        if m != "Train" and len(gg) > 1:
            plt.plot(gg["recall"], gg["precision"], "-", lw=1.5, color=color, alpha=0.8)
        plt.scatter(gg["recall"], gg["precision"], s=45, color=color, label=m, alpha=0.9)
        for _, row in gg.iterrows():
            p = row["param"]
            if not pd.isna(p):
                plt.annotate(
                    f"{p:g}",
                    (row["recall"], row["precision"]),
                    textcoords="offset points",
                    xytext=(5, 3),
                    fontsize=8,
                    color=color,
                )

    plt.legend(loc="upper left", frameon=False)
    plt.grid(alpha=0.25)
    out = f"./pr_outputs/pr_curve_{variant}_latent.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved PR curve: {out}")

for vname, _, _ in VARIANTS:
    plot_variant(df, vname)

plt.show()