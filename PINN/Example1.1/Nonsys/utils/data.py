from cmath import sqrt
from math import ceil, floor
import torch
from torch import Tensor
from typing import Union, Dict
import numpy as np


def get_sampler(sampler_type: str, **kwargs):
    if sampler_type == "time_spatial":
        return TimeSpatialSampler(**kwargs)
    elif sampler_type == "spatial":
        return SpatialSampler(**kwargs)
    else:
        raise ValueError("Invalid value for 'sampler_type': %s" % sampler_type)


def _transform(x: Tensor, lb: float, ub: float) -> Tensor:
    return (ub - lb) * x + lb


class TimeSpatialSampler(object):
    def __init__(
        self,
        domain: Dict[str, float],
        batch_size_domain: int,
        batch_size_init: int,
        batch_size_bound: int
    ) -> None:
        self.start = domain["start"]
        self.end = domain["end"]
        assert(self.start < self.end)
        self.left = domain["left"]
        self.right = domain["right"]
        assert(self.left < self.right)
        self.bot = domain["bot"]
        self.top = domain["top"]
        assert(self.bot < self.top)
        self.n_domain = batch_size_domain
        self.n_init = batch_size_init
        self.n_bound = batch_size_bound
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _time_domain_sample(self, n_samples: int) -> Tensor:
        result = torch.rand(n_samples, 3, device = self.device)
        result[:, 0] = _transform(result[:, 0], self.start, self.end)
        result[:, 1] = _transform(result[:, 1], self.left, self.right)
        result[:, 2] = _transform(result[:, 2], self.bot, self.top)
        return result

    def _boundary_sample(self, n_samples: int) -> Dict[str, Tensor]:
        result = self._time_domain_sample(n_samples)
        for k in range(0,n_samples):
            if(k % 4 == 0):
                result[k,1] = self.left
            elif(k % 4 == 1):
                result[k,1] = self.right
            elif(k % 4 == 2):
                result[k,2] = self.top
            else:
                result[k,2] = self.bot
        return result

    def _slice_sample(self, n_samples: int, moment: float) -> Tensor:
        result = self._time_domain_sample(n_samples)
        result[:,0] = moment
        return result
    
    def __call__(
        self,
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        xtdomain = self._time_domain_sample(self.n_domain)
        bound = self._boundary_sample(self.n_bound)
        startdomain = self._slice_sample(self.n_init, self.start)
        enddomain = self._slice_sample(self.n_init, self.end)
        return {"xtdomain": xtdomain, "boundary": bound, "start": startdomain, "end": enddomain}

class SpatialSampler(object):
    def __init__(
        self,
        batch_size_domain: int,
        batch_size_bound: int,
        batch_size_interface: int,
        r0,
        n_dim: int
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_domain = batch_size_domain
        self.n_bound = batch_size_bound
        self.n_interface = batch_size_interface
        self.n_dim = n_dim
        self.r0 = r0

    def _domain_sample(self, n_samples: int) -> Tensor:
        # result = 2*torch.rand(n_samples, self.n_dim, device = self.device)-1

        r = torch.rand([n_samples, 1]).to(self.device)
        mean = np.zeros([self.n_dim])
        cov = np.eye(self.n_dim)
        theta = np.random.multivariate_normal(mean, cov, n_samples, 'raise')
        theta = torch.from_numpy(theta).float().to(self.device)
        theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))
        result = r ** (1 / self.n_dim) * theta

        return result

    def _boundary_sample(self, n_samples: int) -> Dict[str, Tensor]:
        mean = np.zeros([self.n_dim])
        cov = np.eye(self.n_dim)
        theta = np.random.multivariate_normal(mean, cov, n_samples, 'raise')
        theta = torch.from_numpy(theta).float().to(self.device)
        result = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))
        return result

    def _interface_sample(self, n_samples: int) -> Dict[str, Tensor]:
        mean = np.zeros([self.n_dim])
        cov = np.eye(self.n_dim)
        theta = np.random.multivariate_normal(mean, cov, n_samples, "raise")
        theta = torch.from_numpy(theta).float().to(self.device)
        result = theta / torch.sqrt(torch.sum(theta**2, dim=1, keepdim=True))
        result = self.r0 * result
        return result

    def __call__(
        self,
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        domain = self._domain_sample(self.n_domain)
        bound = self._boundary_sample(self.n_bound)
        interface = self._interface_sample(self.n_interface)
        return {"domain": domain, "boundary": bound, "interface": interface}
