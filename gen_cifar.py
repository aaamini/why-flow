import os, torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import datasets, transforms

# --- Config ---
DATA_ROOT = "./data"
IMAGE_SIZE = 32
N_TRAIN = 2048
N_REAL  = 5000
SEED = 123
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Check local CIFAR-10 and load ---
def exists_cifar10(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "cifar-10-batches-py"))

download_flag = not exists_cifar10(DATA_ROOT)
cifar = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=download_flag,
    transform=transforms.ToTensor(),  # [0,1], [3,32,32]
)

# Load all into memory
loader = torch.utils.data.DataLoader(cifar, batch_size=len(cifar), shuffle=False, num_workers=0)
imgs, labels = next(iter(loader))  # imgs: [50000,3,32,32] in [0,1]

# Flatten and move to device
X_flat = imgs.view(imgs.size(0), -1).to(device)

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
    # [N, 3*H*W] -> [N,3,H,W]
    return flat.view(flat.size(0), 3, IMAGE_SIZE, IMAGE_SIZE).clamp(0,1)

def save_grid(tensor_imgs: torch.Tensor, path: str, nrow: int = 8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(tensor_imgs.cpu(), nrow=nrow, padding=2)
    plt.figure(figsize=(min(nrow, tensor_imgs.size(0)) * 1.2, 1.2 * (tensor_imgs.size(0) // nrow + 1)))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# --- Generate samples ---
N_SAMPLES = 64  # number of images to visualize
# Euler-1 at a given t
t_euler = 0.8
X_euler = gens.euler_one_step(t_euler, N_SAMPLES)  # [N, 3*32*32]
imgs_euler = flat_to_imgs(X_euler)
save_grid(imgs_euler, "./pr_outputs/cifar10_samples_euler1_t0.8.png", nrow=8)

# D-ODE (rk2) with a chosen number of steps
steps = 8
X_dode = gens.dode(int(steps), N_SAMPLES, t1=1.0, method="rk2")  # [N, 3*32*32]
imgs_dode = flat_to_imgs(X_dode)
save_grid(imgs_dode, f"./pr_outputs/cifar10_samples_dode_rk2_steps{steps}.png", nrow=8)

# Optionally also visualize the "Exact x_t" for comparison
t_exact = 0.8
X_exact = gens.exact_xt(t_exact, N_SAMPLES)
imgs_exact = flat_to_imgs(X_exact)
save_grid(imgs_exact, "./pr_outputs/cifar10_samples_exact_xt_t0.8.png", nrow=8)