
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import torch
from torch import Tensor
#import evf_sweep as EVF
from evf import sample_rho_t_empirical, EmpiricalVectorField, Integrator, uniform_grid

GeneratorFn = Callable[[Any, int], Tensor]

@dataclass
class MethodSpec:
    name: str
    params: Sequence[Any]
    generator: GeneratorFn
    n_samples: int

class EVFGenerators:
    def __init__(self, Y_train: Tensor):
        assert Y_train.dim() == 2, "Y_train must be [N,D]"
        self.Y_train = Y_train
        self.device = Y_train.device
        self.dtype = Y_train.dtype

    @torch.no_grad()
    def exact_xt(self, t: float, n_samp: int) -> Tensor:
        return sample_rho_t_empirical(self.Y_train, float(t), int(n_samp))

    @torch.no_grad()
    def euler_one_step(self, t: float, n_samp: int) -> Tensor:
        x_t = sample_rho_t_empirical(self.Y_train, float(t), int(n_samp))
        field = EmpiricalVectorField(self.Y_train.double())
        t_tensor = x_t.new_full((x_t.size(0), 1), float(t))
        v = field(t_tensor, x_t)
        return x_t + (1.0 - float(t)) * v
        # return EVF.euler_one_step(self.Y_train, x_t, float(t))

    @torch.no_grad()
    def dode(self, steps: int, n_samp: int, *, t1: float = 1.0, method: str = "rk2") -> Tensor:
        field = EmpiricalVectorField(self.Y_train.double())
        integ = Integrator(field)
        t_grid = uniform_grid(steps, t1=float(t1))
        x0 = torch.randn(n_samp, self.Y_train.size(1), device=self.device, dtype=self.dtype)
        xT = integ.integrate(x0, t_grid, method=method, return_traj=False)
        return xT.float()
        # return EVF.dode(self.Y_train, int(steps), float(t1), int(n_samp), method=str(method))

def make_evf_generators(Y_train: Tensor) -> EVFGenerators:
    return EVFGenerators(Y_train)
