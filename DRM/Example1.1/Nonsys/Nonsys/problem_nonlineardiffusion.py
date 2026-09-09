from typing import Callable, Dict
from functools import partial
import os
import math
import torch
import torch.nn as nn                     # neural networks
from torch import autograd, Tensor


def _gradient(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    grad = autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(
        outputs), create_graph=True, only_inputs=True)
    return grad[0]

def grad(func: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    x_clone = x.clone().detach().requires_grad_(True)
    fx = func(x_clone)
    return _gradient(fx, x_clone)

def laplace(func: Callable[[torch.Tensor], torch.Tensor], x):
    xclone = x.clone().detach().requires_grad_(True)
    uforward = func(xclone)

    grad = autograd.grad(uforward, xclone, grad_outputs=torch.ones_like(uforward),\
        only_inputs=True, create_graph=True, retain_graph=True)[0]

    result = autograd.grad(grad[:, 0:1], xclone, grad_outputs=torch.ones_like(grad[:, 0:1]),\
        only_inputs=True, create_graph=True, retain_graph=True)[0][:, 0:1]
    for k in range(1, x.size(1)):
        result = torch.cat([result, autograd.grad(grad[:, k:(k+1)], xclone, grad_outputs=torch.ones_like(grad[:, k:(k+1)]),\
        only_inputs=True, create_graph=True, retain_graph=True)[0][:, k:(k+1)]], dim = 1)

    return torch.sum(result, dim = 1, keepdim = True)

def _square(x: torch.Tensor) -> torch.Tensor:
    return torch.square(x)

def _dot(xvector: torch.Tensor, yvector: torch.Tensor) -> torch.Tensor:
    return torch.sum(xvector * yvector, dim = 1)

def _integral(fx: torch.Tensor) -> torch.Tensor:
    return torch.mean(fx)

def ut(u: Callable[[torch.Tensor], torch.Tensor], t:float, x:torch.Tensor):
    return t * u(x)


class NonsysElliptical(object):
    def __init__(self, ndim: int, config: Dict[str, dict]) -> None:
        self.ndim = ndim
        self.kappa = 0.0
        self.alpha = 1.0
        self.domainSize = 1.0
        self.boundarySize = 2.0 * self.ndim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p = 1.1
        torch.manual_seed(20230713)
        self.loss_function = nn.MSELoss(reduction='mean')
        
# Solution: SUM sin(pi/2*xi)
# Equation: -Delta u + epsilon_1 * nablau + pi^2/4*u = f
# where:
# f = pi^2/2*SUM sin(pi/2*xi) + pi/2*cos(pi/2*x1)
# Boundary: \nabla u*n = h
        
    def l2Norm(
        self,
        u1: Callable[[torch.Tensor], torch.Tensor],
        u2: Callable[[torch.Tensor], torch.Tensor],
        x_domain: torch.Tensor
    ) -> torch.Tensor:
        return _integral(_square(u1(x_domain)-u2(x_domain)))

    def solution(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sum(x ** 2, dim=1, keepdim=True) ** (1/2)
        result = 1 - r ** (self.p/ (self.p - 1))
        return result

    def grad_solution(self, x: torch.Tensor) -> torch.Tensor:
        xclone = x.clone().detach().requires_grad_(True)
        ux = self.solution(xclone)
        grad = autograd.grad(ux, xclone, grad_outputs=torch.ones_like(ux), \
                             only_inputs=True, create_graph=True, retain_graph=True)[0]
        result = -torch.sum(grad**2, dim = 1, keepdim = True) ** ((self.p - 2)/2) * grad
        return result

    def f(self, x: torch.Tensor) -> torch.Tensor:
        result = 0*x[:, 0:1] + 3 * (self.p / (self.p - 1)) ** (self.p - 1)
        return result

    def Phi(
        self,
        u: Callable[[torch.Tensor], torch.Tensor],
        x_domain: torch.Tensor,
        x_boundary: torch.Tensor,
        x_interface: torch.Tensor,
        mu
    ) -> torch.Tensor:
        xclone = x_domain.clone().detach().requires_grad_(True)
        ux = u(xclone)
        grad = autograd.grad(ux, xclone, grad_outputs=torch.ones_like(ux), \
                             only_inputs=True, create_graph=True, retain_graph=True)[0]
        result = 1/self.p * torch.sum(grad ** 2, dim=1, keepdim=True) ** (self.p / 2) - self.f(x_domain) * u(x_domain)

        res_PDE = self.domainSize * _integral(result)

        res_boundary = 100 * self.loss_function(u(x_boundary), self.solution(x_boundary))

        return res_PDE + res_boundary