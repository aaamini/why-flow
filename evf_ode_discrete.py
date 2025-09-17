#%%
import torch, math
from torch import nn, Tensor
import matplotlib.pyplot as plt
import numpy as np
from metrics import nn_rmse_to_targets

# --- Empirical vector field (clean: no blend, no top-k) ---
class EmpiricalVectorField:
    def __init__(self, Y: Tensor, eps: float = 1e-6):
        self.Y = Y
        self.eps = eps
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        B, d = x.shape
        Y = self.Y.to(device=x.device, dtype=x.dtype)              # [n,d]
        one_minus_t = (1.0 - t).clamp_min(self.eps)                # [B,1]
        sigma2 = one_minus_t**2
        x2  = (x**2).sum(-1, keepdim=True)                         # [B,1]
        yy2 = (Y**2).sum(-1, keepdim=True).T                       # [1,n]
        xy  = x @ Y.T                                              # [B,n]
        num = x2 - 2.0 * t * xy + (t**2) * yy2                     # [B,n]
        logw = - num / (2.0 * sigma2)                              # [B,n]
        w = torch.softmax(logw, dim=1)
        mu = w @ Y                                                 # [B,d]
        return (mu - x) / one_minus_t                              # [B,d]
    __call__ = forward

# --- Integrator with RK2 / RK4 and custom time grids ---
class Integrator:
    def __init__(self, field): self.field = field
    @staticmethod
    def _time_like(x: Tensor, s: float) -> Tensor:
        return x.new_full((x.size(0),1), float(s))
    @torch.no_grad()
    def step_euler(self, x: Tensor, t0: float, dt: float) -> Tensor:
        t0 = self._time_like(x, t0); dt = self._time_like(x, dt)
        return x + dt*self.field(t0, x)
    @torch.no_grad()
    def step_rk2(self, x: Tensor, t0: float, dt: float) -> Tensor:
        t0 = self._time_like(x, t0); dt = self._time_like(x, dt)
        k1  = self.field(t0, x) # v(t0, x)
        k2  = self.field(t0 + 0.5*dt, x + 0.5*dt*k1) # v(t0 + dt/2, x + dt/2*k1)
        return x + dt*k2
    @torch.no_grad()
    def step_rk4(self, x: Tensor, t0: float, dt: float) -> Tensor:
        t0 = self._time_like(x, t0); dt = self._time_like(x, dt)
        k1  = self.field(t0,            x)
        k2  = self.field(t0+0.5*dt,    x + 0.5*dt*k1)
        k3  = self.field(t0+0.5*dt,    x + 0.5*dt*k2)
        k4  = self.field(t0+dt,        x + dt*k3)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    @torch.no_grad()
    def integrate(self, x0: Tensor, t_grid, method="rk2", return_traj=False):
        x = x0
        step = self.step_euler if method=="euler" else self.step_rk2 if method=="rk2" else self.step_rk4
        mids = []
        if return_traj: traj=[x0.detach().clone()]
        for a,b in zip(t_grid[:-1], t_grid[1:]):
            dt = b - a
            mids.append(a + 0.5*dt)   # midpoint time actually used by RK2/RK4’s first mid eval
            x = step(x, a, dt)
            if return_traj: traj.append(x.detach().clone())
        return (x, torch.tensor(mids, device=x0.device), torch.stack(traj,0)) if return_traj else x

# --- Reference sampler for rho_t (mixture at time t) ---
@torch.no_grad()
def sample_rho_t_empirical(Y: Tensor, t: float, m: int) -> Tensor:
    z = torch.randn(m, Y.size(1), device=Y.device, dtype=Y.dtype)
    idx = torch.randint(0, Y.size(0), (m,), device=Y.device)
    y = Y[idx]
    return (1.0 - t) * z + t * y

# --- Time grids ---
def uniform_grid(n_steps: int, t0=0.0, t1=1.0):
    return [t0 + i*(t1-t0)/n_steps for i in range(n_steps+1)]

def near1_grid(n_steps: int, gamma: float = 4.0, t0=0.0, t1=1.0):
    # concentrate steps near 1: t = 1 - (1-s)^gamma, s=0..1
    s = torch.linspace(0,1,n_steps+1)
    t = 1.0 - (1.0 - s)**gamma
    return (t0 + (t1-t0)*t).tolist()

# --- Runner: one configuration ---
# @torch.no_grad()
# def run_once(Y, n_steps=12, method="rk2", grid="uniform", n_samp=1024, use_double=True):
#     device = Y.device
#     Yd = Y.double() if use_double else Y
#     field = EmpiricalVectorField(Yd)
#     integ = Integrator(field)
#     t_grid = uniform_grid(n_steps) if grid=="uniform" else near1_grid(n_steps, gamma=4.0)
#     # ODE samples
#     x0 = torch.randn(n_samp, Yd.size(1), device=device, dtype=Yd.dtype)
#     xT, mids, _ = integ.integrate(x0, t_grid, method=method, return_traj=True)
#     t_star = float(mids[-1])          # last midpoint actually used
#     # Reference samples at same time
#     ref = sample_rho_t_empirical(Yd, t_star, n_samp)
#     # Metrics
#     mmd = mmd_rbf(xT.float(), ref.float())
#     ed  = energy_distance(xT, ref)
#     rmse_to_Y_ode  = nn_rmse_to_targets(xT, Yd)
#     rmse_to_Y_ref  = nn_rmse_to_targets(ref, Yd)
#     return {
#         "t_star": t_star,
#         "xT": xT.float(), "ref": ref.float(),
#         "mmd": mmd, "energy": ed,
#         "rmse_to_Y_ode": rmse_to_Y_ode, "rmse_to_Y_ref": rmse_to_Y_ref,
#     }

# --- Plot: ODE vs Reference vs Target ---
# def plot_three(x_ode: Tensor, x_ref: Tensor, Y: Tensor, title: str, xlim=(-3,3), ylim=(-3,3)):
#     fig, axes = plt.subplots(1,3, figsize=(12,4), sharex=True, sharey=True)
#     for ax, data, name in zip(axes, [x_ode, x_ref, Y], ["ODE @ t*", "Reference ρ_t* (mixture)", "Target (Y)"]):
#         D = data.detach().cpu()
#         ax.scatter(D[:,0], D[:,1], s=8)
#         ax.set_title(name)
#         ax.set_xlim(*xlim); ax.set_ylim(*ylim)
#     fig.suptitle(title)
#     plt.tight_layout(); plt.show()

def make_line(n: int, slope: float, intercept: float):
    x = torch.randn(n, 1)
    y = slope * x + intercept
    return torch.cat([x, y], dim=1)

def make_sine(n: int, frequency: float = 1.0, phase: float = 0.0):
    x = torch.rand(n, 1)
    y = torch.sin(2 * math.pi * frequency * x + phase)
    return torch.cat([x, y], dim=1)
#%%
# Build noiseless target (same Y you already use)
from sklearn.datasets import make_moons, make_circles

device = "cuda" if torch.cuda.is_available() else "cpu"

make_data = lambda n: torch.tensor(make_moons(n, noise=0.0)[0], device=device, dtype=torch.float32)
#make_data = lambda n: torch.tensor(make_circles(n, noise=0.0, factor=0.5)[0], device=device, dtype=torch.float32)     
#make_data = lambda n: torch.tensor(make_line(n, .25, 0.5), device=device, dtype=torch.float32)
#make_data = lambda n: torch.tensor(make_sine(n, .5, 0.0), device=device, dtype=torch.float32)
#Y_np = make_sine(n_orig, .5, 0.0)
n_orig = 25
Y = make_data(n_orig)      # try small n first

n_samp = 4096
n_steps = 8
grid = "uniform"
Yd = Y.double() if True else Y
method = "euler"
field = EmpiricalVectorField(Yd)
integ = Integrator(field)
t1 = 1
#t_grid = uniform_grid(n_steps) if grid=="uniform" else near1_grid(n_steps, gamma=4.0)
t_grid = uniform_grid(n_steps, t1=t1) if grid=="uniform" else near1_grid(n_steps, gamma=4.0, t1=t1)
# ODE samples
x0 = torch.randn(n_samp, Yd.size(1), device=device, dtype=Yd.dtype)
xT, mids, _ = integ.integrate(x0, t_grid, method=method, return_traj=True)
t_star = float(mids[-1])


idx = torch.randint(0, Y.size(0), (n_samp,), device=Y.device)
x0 = Y[idx]
#xT = integ.step_rk2(x0, .9, -(1.0-t_star))

t_grid_str =  "".join(str(t)+"\n" for t in t_grid)
print(f"grid: \n{t_grid_str}method: {method}, n_steps: {n_steps}, t_star: {t_star}")


ref = sample_rho_t_empirical(Y, t_star, n_samp)

#plt.scatter(Y_fine_grid[:,0], Y_fine_grid[:,1], s=1, alpha=0.5)
#plt.scatter(ref[:,0], ref[:,1], s=2, alpha=0.5)
plt.scatter(xT[:,0].cpu(), xT[:,1].cpu(), s=2, alpha=0.5)#
plt.scatter(Y[:,0].cpu(), Y[:,1].cpu(), s=2, alpha=0.5)#
plt.legend(["Reference", "ODE"])
#plt.savefig("fig_evf_ode_discretee_two_circles.png", dpi=150)

Y_fine_grid = make_data(4096)
print(f"ODE RMSE: {nn_rmse_to_targets(xT.float(), Y_fine_grid):.4g}")
print(f"Ref RMSE: {nn_rmse_to_targets(ref, Y_fine_grid):.4g}")


# %%
