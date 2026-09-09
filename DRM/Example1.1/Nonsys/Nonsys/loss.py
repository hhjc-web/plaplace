from typing import Callable

from torch import nn
from torch import Tensor

from .problem_nonlineardiffusion import NonsysElliptical
from ..utils.nn import clone_freeze


class NonsysEllipticalLoss(nn.Module):
    def __init__(
        self,
        lamb: float,
        mu: float,
        problem: NonsysElliptical
    ) -> None:
        super().__init__()
        self.lamb = lamb
        self.mu = mu
        self.problem = problem

    def loss_step(
        self,
        unew: Callable[[Tensor], Tensor],
        uold: Callable[[Tensor], Tensor],
        first: bool,
        x_domain: Tensor,
        x_boundary: Tensor,
        x_interface: Tensor,
    ) -> Tensor:
        return self.problem.Phi(unew,x_domain,x_boundary,x_interface,self.mu)/self.lamb

    def pre_loss(
        self,
        u: Callable[[Tensor], Tensor],
        x_domain: Tensor
    ) -> Tensor:
        return self.problem.l2Norm(u, self.problem.solution, x_domain)