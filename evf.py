import torch, math
from torch import nn, Tensor
import matplotlib.pyplot as plt
import numpy as np

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
