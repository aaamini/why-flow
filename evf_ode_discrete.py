#%%
import torch, math
from torch import nn, Tensor
import matplotlib.pyplot as plt
import numpy as np
from metrics import nn_rmse_to_targets

from evf import EmpiricalVectorField, sample_rho_t_empirical, Integrator, uniform_grid
from nnvf import NNVectorField, train_nn_field, TrainCfg

from sklearn.datasets import make_moons, make_circles

def make_line(n: int, slope: float, intercept: float):
    x = torch.randn(n, 1)
    y = slope * x + intercept
    return torch.cat([x, y], dim=1)

def make_sine(n: int, frequency: float = 1.0, phase: float = 0.0):
    x = torch.rand(n, 1)
    y = torch.sin(2 * math.pi * frequency * x + phase)
    return torch.cat([x, y], dim=1)



def dode(Y: Tensor, n_steps: int, t: float, n_samp: int, method: str) -> Tensor:
    field = EmpiricalVectorField(Y.double())
    integ = Integrator(field)
    t_grid = uniform_grid(n_steps, t1=t)    
    x0 = torch.randn(n_samp, Y.size(1), device=Y.device, dtype=Y.dtype)    
    xT, mids, _ = integ.integrate(x0, t_grid, method=method, return_traj=True)
    t_star = float(mids[-1])
    return xT.float(), t_star
#%%
# Build noiseless target (same Y you already use)


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 0
torch.manual_seed(seed)

# Choose dataset: "2moons" or "2circles"
dataset_name = "2moons"
if dataset_name == "2moons":
    make_data = lambda n: torch.tensor(make_moons(n, noise=0.0)[0], device=device, dtype=torch.float32)
elif dataset_name == "2circles":
    make_data = lambda n: torch.tensor(make_circles(n, noise=0.0, factor=0.5)[0], device=device, dtype=torch.float32)
else:
    raise ValueError(f"Unknown dataset_name: {dataset_name}")
#make_data = lambda n: torch.tensor(make_line(n, .25, 0.5), device=device, dtype=torch.float32)
#make_data = lambda n: torch.tensor(make_sine(n, .5, 0.0), device=device, dtype=torch.float32)
#Y_np = make_sine(n_orig, .5, 0.0)
n_orig = 1024
Y = make_data(n_orig)      # try small n first

# Build a suffix for saved filenames that encodes dataset, seed, and n
suffix = f"{dataset_name}_seed{seed}_n{n_orig}"

n_samp = 4096
n_steps = 10
t = 1.0
method = "rk2"
xT, t_star = dode(Y, n_steps, t, n_samp, method)

# Neural (train on the same fixed Y)
nnf = NNVectorField(dim=2, h=64).to(device)
hist = train_nn_field(nnf, Y, TrainCfg(iters=5000, batch=256, lr=1e-2, log_every=100))
nint = Integrator(nnf)
x0 = torch.randn(n_samp, Y.size(1), device=Y.device, dtype=Y.dtype)
x_nn = nint.integrate(x0, uniform_grid(n_steps, t1=t), method=method, return_traj=False)

# Yd = Y.double() if True else Y
# method = "euler"
# field = EmpiricalVectorField(Yd)
# integ = Integrator(field)
# #t_grid = uniform_grid(n_steps) if grid=="uniform" else near1_grid(n_steps, gamma=4.0)
# t_grid = uniform_grid(n_steps, t1=1.0)
# # ODE samples
# x0 = torch.randn(n_samp, Yd.size(1), device=device, dtype=Yd.dtype)
# xT, mids, _ = integ.integrate(x0, t_grid, method=method, return_traj=True)
# t_star = float(mids[-1])


idx = torch.randint(0, Y.size(0), (n_samp,), device=Y.device)
x0 = Y[idx]
#xT = integ.step_rk2(x0, .9, -(1.0-t_star))

print(f"dataset: {dataset_name}, method: {method}, n_steps: {n_steps}, t_star: {t_star}")

ref = sample_rho_t_empirical(Y, t_star, n_samp)

# Exact ODE (EVF)
fig1, ax1 = plt.subplots(figsize=(4, 4), constrained_layout=True)
ax1.scatter(ref[:,0].cpu(), ref[:,1].cpu(), s=2, alpha=0.5)
ax1.axis("equal")
fig1.savefig(f"exact_ode_evf_{suffix}.png", dpi=200)
plt.close(fig1)

# Discretized ODE (EVF)
fig2, ax2 = plt.subplots(figsize=(4, 4), constrained_layout=True)
ax2.scatter(xT[:,0].cpu(), xT[:,1].cpu(), s=2, alpha=0.5)
ax2.axis("equal")
fig2.savefig(f"discretized_ode_evf_{suffix}.png", dpi=200)
plt.close(fig2)

# Discretized ODE (NN Field)
fig3, ax3 = plt.subplots(figsize=(4, 4), constrained_layout=True)
ax3.scatter(x_nn[:,0].cpu(), x_nn[:,1].cpu(), s=2, alpha=0.5)
ax3.axis("equal")
fig3.savefig(f"discretized_ode_nn_field_{suffix}.png", dpi=200)
plt.close(fig3)

# Training Data
fig3, ax3 = plt.subplots(figsize=(4, 4), constrained_layout=True)
ax3.scatter(Y[:,0].cpu(), Y[:,1].cpu(), s=2, alpha=1)
ax3.axis("equal")
fig3.savefig(f"training_data_{suffix}.png", dpi=200)
plt.close(fig3)

Y_fine_grid = make_data(4096)
print(f"ODE RMSE: {nn_rmse_to_targets(xT.float(), Y_fine_grid):.4g}")
print(f"Ref RMSE: {nn_rmse_to_targets(ref, Y_fine_grid):.4g}")


    # %%
