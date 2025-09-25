import os, torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import torch.nn.functional as F

# --- Config ---
DATA_ROOT = "./data"
IMAGE_SIZE = 32   # if 32, we pad 28->32; set to 28 to keep native size
N_TRAIN = 1024
N_REAL  = 2000
SEED = 123
N_SAMPLES = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Check local MNIST and load (no download if files exist) ---
def exists_mnist(root: str) -> bool:
    raw = os.path.join(root, "MNIST", "raw")
    req = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    return all(os.path.isfile(os.path.join(raw, f)) for f in req)

download_flag = not exists_mnist(DATA_ROOT)
mnist = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    download=download_flag,
    transform=transforms.ToTensor(),  # [0,1], [1,28,28]
)

# Load all train images
loader = torch.utils.data.DataLoader(mnist, batch_size=len(mnist), shuffle=False, num_workers=0)
imgs, labels = next(iter(loader))  # imgs: [60000,1,28,28]

# Resize/pad to IMAGE_SIZE if needed
MNIST_SIZE = 28
if IMAGE_SIZE == MNIST_SIZE:
    imgs_use = imgs
elif IMAGE_SIZE == 32:
    pad_h = (IMAGE_SIZE - MNIST_SIZE) // 2  # 2
    pad_w = (IMAGE_SIZE - MNIST_SIZE) // 2  # 2
    imgs_use = F.pad(imgs, (pad_w, pad_w, pad_h, pad_h))  # (L, R, T, B)
else:
    raise ValueError(f"Unsupported IMAGE_SIZE={IMAGE_SIZE}. Use 28 or 32.")

# Flatten and move to device
X_flat = imgs_use.view(imgs_use.size(0), -1).to(device)

# Sample disjoint Y_train/Y_real
rng = np.random.RandomState(SEED)
perm = rng.permutation(X_flat.size(0))
idx_train = perm[:N_TRAIN]
idx_real  = perm[N_TRAIN:N_TRAIN+N_REAL]
Y_train = X_flat[idx_train].contiguous()
Y_real  = X_flat[idx_real].contiguous()

# --- EVF generators ---
from methods import make_evf_generators
gens = make_evf_generators(Y_train)

# --- Helpers to visualize/save ---
def flat_to_imgs(flat: torch.Tensor) -> torch.Tensor:
    # [N, 1*H*W] -> [N,1,H,W], clamp to [0,1]
    return flat.view(flat.size(0), 1, IMAGE_SIZE, IMAGE_SIZE).clamp(0,1)

def save_grid(tensor_imgs: torch.Tensor, path: str, nrow: int = 8, cmap="gray"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(tensor_imgs.cpu(), nrow=nrow, padding=2)
    plt.figure(figsize=(min(nrow, tensor_imgs.size(0)) * 1.2, 1.2 * (tensor_imgs.size(0) // nrow + 1)))
    plt.axis("off")
    # make_grid returns [3,H,W] even for single-channel; take first channel for grayscale
    img = grid.permute(1, 2, 0).numpy()
    if img.shape[2] == 1 or cmap == "gray":
        plt.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(img)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# --- Generate samples ---
# Euler-1 at a given t
t_euler = 0.8
X_euler = gens.euler_one_step(t_euler, N_SAMPLES)  # [N, H*W]
imgs_euler = flat_to_imgs(X_euler)
save_grid(imgs_euler, f"./pr_outputs/mnist_samples_euler1_t{t_euler}.png", nrow=8, cmap="gray")

# D-ODE (rk2) with chosen steps
steps = 8
X_dode = gens.dode(int(steps), N_SAMPLES, t1=1.0, method="rk2")  # [N, H*W]
imgs_dode = flat_to_imgs(X_dode)
save_grid(imgs_dode, f"./pr_outputs/mnist_samples_dode_rk2_steps{steps}.png", nrow=8, cmap="gray")

# Optional: Exact x_t for comparison
t_exact = 0.8
X_exact = gens.exact_xt(t_exact, N_SAMPLES)
imgs_exact = flat_to_imgs(X_exact)
save_grid(imgs_exact, f"./pr_outputs/mnist_samples_exact_xt_t{t_exact}.png", nrow=8, cmap="gray")