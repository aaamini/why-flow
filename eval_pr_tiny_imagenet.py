from __future__ import annotations
from typing import Any, List, Dict, Tuple, Optional
import math, os, tarfile, shutil
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

from new_metrics import pr_knn_conditioned
from methods import MethodSpec, make_evf_generators
from torch import Tensor
from feature_extractor import InceptionPool3, Whitener

from torchvision import transforms
from torchvision.datasets import ImageFolder

import requests
from tqdm import tqdm

# =========================
# Config
# =========================
DATASET = "tiny-imagenet-200"
DATA_ROOT = "./data"
TIN_DIR = os.path.join(DATA_ROOT, DATASET)
TIN_TAR = os.path.join(DATA_ROOT, "tiny-imagenet-200.tar.gz")
TIN_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"  # canonical URL (zip)
TIN_ZIP = os.path.join(DATA_ROOT, "tiny-imagenet-200.zip")
AUTO_DOWNLOAD = True  # set False to avoid network fetch

N_TRAIN = 10000       # subset size for EVF "train" set
N_REAL  = 20000       # subset size for "real" eval set
K       = 3

P_GEN  = 0.95
P_REAL = 0.5

T_VALUES     = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
STEP_VALUES  = [2, 4, 6, 8, 10, 12, 16]
N_EVAL       = N_REAL

MEASURE_IN_FEATURES = True
WHITEN_FEATURES     = False
IMAGE_SIZE          = 64     # Tiny ImageNet native resolution
ENC_BATCH_SIZE      = 128

RNG_SEED = 123

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Tiny ImageNet presence check and (optional) download/extract
# =========================
def tiny_imagenet_exists(root: str) -> bool:
    train_dir = os.path.join(root, DATASET, "train")
    val_dir   = os.path.join(root, DATASET, "val")
    return os.path.isdir(train_dir) and os.path.isdir(val_dir)

def download_with_progress(url: str, dst: str):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {os.path.basename(dst)}")
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        pbar.close()

def extract_zip(path_zip: str, dst_dir: str):
    import zipfile
    with zipfile.ZipFile(path_zip, "r") as zf:
        zf.extractall(dst_dir)

os.makedirs(DATA_ROOT, exist_ok=True)

if not tiny_imagenet_exists(DATA_ROOT):
    if not AUTO_DOWNLOAD:
        raise FileNotFoundError(
            f"{DATASET} not found at {TIN_DIR}. Please place the dataset under {TIN_DIR} "
            "with subfolders train/ and val/ (official layout), or enable AUTO_DOWNLOAD."
        )
    print(f"{DATASET} not found locally. Attempting to download...")
    try:
        if not os.path.isfile(TIN_ZIP):
            download_with_progress(TIN_URL, TIN_ZIP)
        print("Extracting zip...")
        extract_zip(TIN_ZIP, DATA_ROOT)
        # After extraction, path should be ./data/tiny-imagenet-200
        if not tiny_imagenet_exists(DATA_ROOT):
            raise RuntimeError("Extraction completed, but dataset structure not found. Please verify the archive.")
    except Exception as e:
        raise RuntimeError(
            f"Automatic download/extract failed: {e}\n"
            f"Manually download from {TIN_URL} and extract to {DATA_ROOT} so that {TIN_DIR}/train exists."
        )
else:
    print(f"Found {DATASET} at {TIN_DIR}. Using local copy.")

# =========================
# Dataset and transform
# =========================
# Tiny ImageNet images can have small variations; we ensure 64x64,
# convert to tensor in [0,1]. No augmentation for evaluation.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),    # [0,1], shape [3, H, W]
])

# Use the training split for both Y_train (reference set for EVF) and Y_real (evaluation target set),
# sampled as disjoint subsets.
train_folder = os.path.join(TIN_DIR, "train")
dataset = ImageFolder(root=train_folder, transform=transform)

# Load all into memory once (if memory is a concern, we can stream and sample)
loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
imgs, labels = next(iter(loader))  # imgs: [N,3,64,64]
N_total = imgs.size(0)
print(f"Tiny ImageNet train loaded: {N_total} images")

# Flatten to [N, 3*H*W] and move to device
X_flat_all = imgs.view(N_total, -1).to(device)

# Sample disjoint subsets
rng = np.random.RandomState(RNG_SEED)
need = N_TRAIN + N_REAL
if need > N_total:
    raise ValueError(f"Requested N_TRAIN+N_REAL={need} exceeds Tiny ImageNet train size {N_total}.")

perm = rng.permutation(N_total)
idx_train = perm[:N_TRAIN]
idx_real  = perm[N_TRAIN:N_TRAIN+N_REAL]

Y_train = X_flat_all[idx_train].contiguous()
Y_real  = X_flat_all[idx_real].contiguous()

print(f"Prepared: Y_train {tuple(Y_train.shape)}, Y_real {tuple(Y_real.shape)}, device={Y_train.device}")

# =========================
# Feature mapping (Inception pool3) + optional whitening
# =========================
if MEASURE_IN_FEATURES:
    enc = InceptionPool3(device=device)

    def flat_to_imgs(flat: Tensor) -> Tensor:
        return flat.view(flat.size(0), 3, IMAGE_SIZE, IMAGE_SIZE).clamp(0,1)

    @torch.no_grad()
    def to_features(flat: Tensor) -> Tensor:
        imgs_ = flat_to_imgs(flat)
        return enc(imgs_, batch_size=ENC_BATCH_SIZE)   # [N, 2048] on CPU

    Phi_train = to_features(Y_train)
    Phi_real  = to_features(Y_real)

    if WHITEN_FEATURES:
        wh = Whitener().fit(Phi_real)
        Phi_train = wh.transform(Phi_train)
        Phi_real  = wh.transform(Phi_real)

    @torch.no_grad()
    def gen_to_features(flat: Tensor) -> Tensor:
        Phi = to_features(flat)
        return wh.transform(Phi) if WHITEN_FEATURES else Phi
else:
    Phi_train, Phi_real = Y_train, Y_real
    gen_to_features = lambda x: x

# =========================
# EVF generators and method configs
# =========================
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

# =========================
# Evaluation
# =========================
def eval_methods(methods: List[MethodSpec], Y_train: torch.Tensor, Y_real: torch.Tensor, k: int) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    Train = Phi_train
    Real  = Phi_real

    # Baseline
    for vname, pg, prc in VARIANTS:
        out = pr_knn_conditioned(Real, Train, Train, p_gen=pg, p_real=prc, k=k)
        records.append({"method": "Train", "param": math.nan, "variant": vname,
                        "precision": out["precision"], "recall": out["recall"],
                        "kept_real_frac": out["kept_real_frac"], "kept_gen_frac": out["kept_gen_frac"]})

    # Methods
    for m in methods:
        for p in m.params:
            X_flat = m.generator(p, m.n_samples)     # [N, 3*H*W]
            X_feat = gen_to_features(X_flat)
            for vname, pg, prc in VARIANTS:
                out = pr_knn_conditioned(Real, X_feat, Train, p_gen=pg, p_real=prc, k=k)
                records.append({"method": m.name, "param": float(p), "variant": vname,
                                "precision": out["precision"], "recall": out["recall"],
                                "kept_real_frac": out["kept_real_frac"], "kept_gen_frac": out["kept_gen_frac"]})
    return pd.DataFrame.from_records(records)

df = eval_methods(methods, Y_train, Y_real, K)

# =========================
# Save and plot
# =========================
os.makedirs("./pr_outputs", exist_ok=True)
csv_path = "./pr_outputs/pr_sweep_results_tiny_imagenet.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

COLORS = {
    "Exact x_t": "#1f77b4",
    "Euler-1": "#2ca02c",
    "D-ODE (rk2)": "#d62728",
    "Train": "#7f7f7f",
}

import matplotlib
def plot_variant(df: pd.DataFrame, variant: str):
    sub = df[df["variant"] == variant].copy()
    plt.figure(figsize=(5.5, 5.5))
    plt.title(f"PR (Tiny ImageNet) -- {variant}")
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
    out = f"./pr_outputs/pr_curve_tiny_imagenet_{variant}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")

for vname, _, _ in VARIANTS:
    plot_variant(df, vname)
plt.show()