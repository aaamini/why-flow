from __future__ import annotations
from typing import Any, Sequence, List, Dict, Tuple, Optional
import math, os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

from new_metrics import pr_knn, pr_knn_conditioned
from methods import MethodSpec, make_evf_generators
# from data import load_dataset  # not used

from torch import Tensor
from feature_extractor import InceptionPool3, Whitener
from torchvision import datasets, transforms

# Config
DATASET = "cifar10"
N_TRAIN = 2048         # you can adjust; CIFAR-10 has 50k train images
N_REAL  = 5000
K       = 3

P_GEN  = 0.95
P_REAL = 0.5

T_VALUES     = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
STEP_VALUES  = [2, 4, 6, 8, 10, 12, 16]
N_EVAL       = N_REAL

# >>> feature-space PR toggles:
MEASURE_IN_FEATURES = True
WHITEN_FEATURES     = False
IMAGE_SIZE          = 32        # CIFAR-10 native size
ENC_BATCH_SIZE      = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# CIFAR-10 data loading with "use local if exists, else download"
# Produces flattened tensors in [0,1]:
#   Y_train: [N_TRAIN, 32*32*3]
#   Y_real:  [N_REAL,  32*32*3]
# ---------------------------------------------------------------------
def exists_cifar10(root: str) -> bool:
    # torchvision stores CIFAR-10 under <root>/cifar-10-batches-py after download
    path = os.path.join(root, "cifar-10-batches-py")
    return os.path.isdir(path)

DATA_ROOT = "./data"
download_flag = not exists_cifar10(DATA_ROOT)
if download_flag:
    print("CIFAR-10 not found locally. Downloading to ./data ...")
else:
    print("CIFAR-10 found locally. Using existing files.")

# ToTensor converts to [0,1], shape [3,32,32]
cifar = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=download_flag,
    transform=transforms.ToTensor(),
)

# Load all train images into memory once
loader = torch.utils.data.DataLoader(cifar, batch_size=len(cifar), shuffle=False, num_workers=0)
imgs, labels = next(iter(loader))  # imgs: [50000, 3, 32, 32], [0,1]

# Flatten and move to device: [N, 3*32*32]
X_flat = imgs.view(imgs.size(0), -1).to(device)

# Sample disjoint subsets
rng = np.random.RandomState(123)
N_total = X_flat.size(0)
need = N_TRAIN + N_REAL
if need > N_total:
    raise ValueError(f"Requested N_TRAIN+N_REAL={need} exceeds CIFAR-10 train size {N_total}.")

perm = rng.permutation(N_total)
idx_train = perm[:N_TRAIN]
idx_real  = perm[N_TRAIN:N_TRAIN+N_REAL]

Y_train = X_flat[idx_train].contiguous()
Y_real  = X_flat[idx_real].contiguous()
print(f"CIFAR-10 loaded: Y_train {tuple(Y_train.shape)}, Y_real {tuple(Y_real.shape)}, device={Y_train.device}")

# >>> Map flattened pixels -> images -> features (Inception pool3), then optional whitening
if MEASURE_IN_FEATURES:
    enc = InceptionPool3(device=device)

    def flat_to_imgs(flat: Tensor) -> Tensor:
        # [N, 3*H*W] -> [N,3,H,W] in [0,1]
        return flat.view(flat.size(0), 3, IMAGE_SIZE, IMAGE_SIZE).clamp(0,1)

    @torch.no_grad()
    def to_features(flat: Tensor) -> Tensor:
        imgs = flat_to_imgs(flat)
        return enc(imgs, batch_size=ENC_BATCH_SIZE)   # returns [N, 2048] (on CPU)

    # Precompute train/real in feature space
    Phi_train = to_features(Y_train)
    Phi_real  = to_features(Y_real)

    if WHITEN_FEATURES:
        wh = Whitener().fit(Phi_real)
        Phi_train = wh.transform(Phi_train)
        Phi_real  = wh.transform(Phi_real)

    # Transform generated batches the same way
    @torch.no_grad()
    def gen_to_features(flat: Tensor) -> Tensor:
        Phi = to_features(flat)
        return wh.transform(Phi) if WHITEN_FEATURES else Phi
else:
    # Fall back to pixel feature space (no encoder)
    Phi_train, Phi_real = Y_train, Y_real
    gen_to_features = lambda x: x

# EVF generators based on flattened TRAIN set
gens = make_evf_generators(Y_train)
methods: List[MethodSpec] = [
    MethodSpec(name="Exact x_t",    params=T_VALUES,    generator=gens.exact_xt,         n_samples=N_EVAL),
    MethodSpec(name="Euler-1",      params=T_VALUES,    generator=gens.euler_one_step,   n_samples=N_EVAL),
    MethodSpec(name="D-ODE (rk2)",  params=STEP_VALUES, generator=lambda s,n: gens.dode(int(s), n, t1=1.0, method="rk2"), n_samples=N_EVAL),
]

VARIANTS = [
    ("vanilla",                      0.0,   0.0),
    (f"gen_novel_{P_GEN}",           P_GEN, 0.0),
    (f"real_novel_{P_REAL}",         0.0,   P_REAL),
    (f"both_gen{P_GEN}_real{P_REAL}", P_GEN, P_REAL),
]

def eval_methods(methods: List[MethodSpec], Y_train: torch.Tensor, Y_real: torch.Tensor, k: int) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    # Use feature-space tensors for PR
    Train = Phi_train
    Real  = Phi_real

    # Train baseline across all variants (including "vanilla" if present)
    for vname, pg, prc in VARIANTS:
        out = pr_knn_conditioned(Real, Train, Train, p_gen=pg, p_real=prc, k=k)
        records.append({"method": "Train", "param": math.nan, "variant": vname,
                        "precision": out["precision"], "recall": out["recall"],
                        "kept_real_frac": out["kept_real_frac"], "kept_gen_frac": out["kept_gen_frac"]})

    # Methods
    for m in methods:
        for p in m.params:
            X_flat = m.generator(p, m.n_samples)     # [N, 3*H*W] pixels
            X_feat = gen_to_features(X_flat)         # [N, 2048] (or identity if pixel space)
            for vname, pg, prc in VARIANTS:
                out = pr_knn_conditioned(Real, X_feat, Train, p_gen=pg, p_real=prc, k=k)
                records.append({"method": m.name, "param": float(p), "variant": vname,
                                "precision": out["precision"], "recall": out["recall"],
                                "kept_real_frac": out["kept_real_frac"], "kept_gen_frac": out["kept_gen_frac"]})
    return pd.DataFrame.from_records(records)

df = eval_methods(methods, Y_train, Y_real, K)

os.makedirs("./pr_outputs", exist_ok=True)
csv_path = "./pr_outputs/pr_sweep_results_cifar10.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

COLORS = {
    "Exact x_t": "#1f77b4",
    "Euler-1": "#2ca02c",
    "D-ODE (rk2)": "#d62728",
    "Train": "#7f7f7f",
}
def plot_variant(df: pd.DataFrame, variant: str):
    sub = df[df["variant"] == variant].copy()
    plt.figure(figsize=(5.5, 5.5))
    plt.title(f"PR (CIFAR-10) -- {variant}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.xlim(-0.02, 1.02); plt.ylim(-0.02, 1.02)
    for m, g in sub.groupby("method"):
        color = COLORS.get(m, None)
        gg = g.sort_values("param")
        if m != "Train" and len(gg) > 1:
            plt.plot(gg["recall"], gg["precision"], "-", lw=1.5, color=color, alpha=0.8)
        plt.scatter(gg["recall"], gg["precision"], s=45, color=color, label=m, alpha=0.9)
        for _, row in gg.iterrows():
            p = row["param"]
            if not pd.isna(p):
                plt.annotate(f"{p:g}", (row["recall"], row["precision"]),
                             textcoords="offset points", xytext=(5,3), fontsize=8, color=color)
    plt.legend(loc="upper left", frameon=False); plt.grid(alpha=0.25)
    out = f"./pr_outputs/pr_curve_cifar10_{variant}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")

for vname, _, _ in VARIANTS:
    plot_variant(df, vname)
plt.show()