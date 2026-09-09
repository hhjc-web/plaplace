import os
from typing import Union, Dict, Optional, Tuple, List

import numpy as np
import scipy.io as sio
import torch
import math
from itertools import chain
from torch import Tensor
from torch import autograd, Tensor

from .problem_nonlineardiffusion import NonsysElliptical
from .loss import NonsysEllipticalLoss
from ..utils.nn import clone_freeze, get_network
from ..utils.data import get_sampler
from ..utils.optimize import get_optim_adam

@torch.no_grad()
def _write_log(
    log_file: str,
    n_epochs: int,
    epoch: int,
    solution_error: Optional[float] = None,
    grad_solution_error: Optional[float] = None
) -> None:
    mode = "a"
    with open(log_file, mode) as f:
        f.write("epoch = {:d}/{:d}, solution_error = {:6.4f}, grad_solution_error = {:6.4f}\n".
                format(epoch + 1, n_epochs, solution_error, grad_solution_error))

@torch.no_grad()
def _generate_uniform_mesh(numEachDim: int, dim1: int, dim2: int, totalDim: int) -> Tensor:
    grid_tmp = torch.linspace(-1.0, 1.0, numEachDim + 1)
    grid_x1, grid_x2 = torch.meshgrid([grid_tmp, grid_tmp], indexing='xy')
    grid_x1, grid_x2 = grid_x1.reshape((-1, 1)), grid_x2.reshape((-1, 1))
    result = torch.ones((numEachDim + 1)*(numEachDim + 1), totalDim) * 0.0
    result[:, dim1:(dim1 + 1)] = grid_x1
    result[:, dim2:(dim2 + 1)] = grid_x2
    return result

class DeepNonsysElliptical(object):
    def __init__(
        self,
        problem: NonsysElliptical,
        config: Dict[str, dict]
    ) -> None:
        # optimal control problem
        self.problem = problem
        # networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.solution_net = get_network(**config["solution_net_params"]).to(self.device)
        self.grad_net = get_network(**config["grad_net_params"]).to(self.device)
        # loss
        self.loss_fn = NonsysEllipticalLoss(problem=self.problem,**config["loss_params"])
        # optimizer
        self.lr = config["adam_optim_params"]["lr"]
        self.weightDecay = config["adam_optim_params"]["weight_decay"]
        # data
        self.sampler = get_sampler(n_dim = self.problem.ndim, **config["sampler_params"])
        self.test_sampler = get_sampler(n_dim=self.problem.ndim, **config["testsampler_params"])
        # train
        self.num_epochs = config["train_params"]["num_epochs"]
        self.perepoch_steps = config["train_params"]["perepoch_steps"]
        self.output_path = config["train_params"]["output_path"]
        self.solution_err_log = np.zeros(self.num_epochs)
        self.grad_solution_err_log = np.zeros(self.num_epochs)
        self.step_log = np.zeros([1,3])
        self.log_path = os.path.join(self.output_path, "log.txt")
        self.loss_path = os.path.join(self.output_path, "loss.mat")
        self.save_int = 0
        self.p = problem.p
        
    def _train_solution_one_step(
        self,
        first,
        samples: Dict[str, Union[Tensor, Dict[str, Tensor]]]
    ) -> float:
        data_domain = samples["domain"]
        data_boundary = samples["boundary"]
        data_interface = samples["interface"]
        loss = self.loss_fn.loss_step(self.solution_net, self.grad_net, first, data_domain, data_boundary,data_interface)
        # if(first):
        #     loss = self.loss_fn.pre_loss(self.solution_net,data_domain)
        self.solution_optimizer.zero_grad()
        self.solution_optimizer2.zero_grad()
        loss.backward()
        # print(list(self.solution_net.parameters()))
        self.solution_optimizer.step()
        self.solution_optimizer2.step()
        return loss.item()

    def _compute_optimal_solution(self, error_samples: Tensor) -> None:
        self.optimal_solution_value = self.problem.solution(error_samples)
        self.optimal_solution_norm = torch.norm(self.optimal_solution_value)
        self.optimal_grad_solution_value = self.problem.grad_solution(error_samples)
        self.optimal_grad_solution_norm = torch.norm(self.optimal_grad_solution_value)

    def _compute_error(self, error_samples: Tensor) -> Tuple[float]:
        pred_solution_value = self.solution_net(error_samples)
        pred_grad_solution_value = self.grad_net(error_samples)

        solution_error = torch.norm(self.optimal_solution_value - pred_solution_value)
        grad_solution_error = torch.norm(self.optimal_grad_solution_value - pred_grad_solution_value)
        return (solution_error / self.optimal_solution_norm), (grad_solution_error / self.optimal_grad_solution_norm)

    def _save_results(self, plot_samples: Tensor) -> None:
        pred_solution_value = self.solution_net(plot_samples)
        pred_solution_value = pred_solution_value.cpu()

        pred_grad_solution_value = self.grad_net(plot_samples)
        pred_grad_solution_value = pred_grad_solution_value.cpu()
        plot_samples_copy = plot_samples.clone()
        plot_copy = plot_samples_copy.cpu()
        
        optimal_solution_value_copy = self.problem.solution(plot_samples).clone().cpu()
        optimal_grad_solution_value_copy = self.problem.grad_solution(plot_samples).clone().cpu()
        
        sio.savemat(self.loss_path, {"solution_error": self.solution_err_log,
                                     "grad_solution_error": self.grad_solution_err_log,
                                     "step_error": self.step_log})
        sio.savemat(os.path.join(self.output_path, "solution" + str(self.save_int) + ".mat"), {"samples": plot_copy.detach().numpy(),
                                         "optimal_solution": optimal_solution_value_copy.detach().numpy(),
                                         "optimal_grad_solution": optimal_grad_solution_value_copy.detach().numpy(),
                                         "pred_solution": pred_solution_value.detach().numpy(),
                                         "pred_grad_solution": pred_grad_solution_value.detach().numpy()})
        self.save_int += 1

    def train(self) -> None:
        with open(self.log_path, "w") as f:
            f.write("Training ... \n")

        error_samples = self.test_sampler()
        error_samples = error_samples["domain"]
        plot_samples = _generate_uniform_mesh(99, 0, 1, self.problem.ndim)
        plot_samples = plot_samples.to(self.device)
        self._compute_optimal_solution(error_samples)
        self._save_results(plot_samples)
        # solution_error = self._compute_error(error_samples)

        check_step = 10
        jump_step = 5
        solution_loss_save = np.zeros(check_step - 2)
        solution_error_save = np.zeros(check_step - 2)
        grad_solution_error_save = np.zeros(check_step - 2)
        self.solution_optimizer = get_optim_adam(self.solution_net.parameters(), self.lr, self.weightDecay)
        self.solution_optimizer2 = get_optim_adam(self.grad_net.parameters(), self.lr, self.weightDecay)

        for epoch in range(0, self.num_epochs):
            
            loss_save = float("inf")
            for local_step in range(self.perepoch_steps):
                samples = self.sampler()
                solution_loss = self._train_solution_one_step(epoch == 0, samples)
                solution_error, grad_solution_error = self._compute_error(error_samples)
                solution_error = solution_error.cpu().detach().numpy()
                grad_solution_error = grad_solution_error.cpu().detach().numpy()
                self.step_log = np.append(self.step_log, [[solution_loss, solution_error, grad_solution_error]], axis = 0)
                solution_loss_save[local_step % (check_step - 2)] = solution_loss
                solution_error_save[local_step % (check_step - 2)] = solution_error
                grad_solution_error_save[local_step % (check_step - 2)] = grad_solution_error
                if(local_step % check_step == (check_step-1)):
                    with open(self.log_path, "a") as f:
                        f.write("solution loss = {:8f}, solution error = {:8f}, grad_solution error = {:8f}\n"\
                         .format(np.mean(solution_loss_save),np.mean(solution_error_save),np.mean(grad_solution_error_save)))

            # record log
            solution_error, grad_solution_error = self._compute_error(error_samples)
            solution_error = solution_error.cpu().detach().numpy()
            grad_solution_error = grad_solution_error.cpu().detach().numpy()
            self.solution_err_log[epoch] = solution_error
            self.grad_solution_err_log[epoch] = grad_solution_error
            _write_log(self.log_path, self.num_epochs, epoch,
                       solution_error, grad_solution_error)
            if(epoch % 1 == 0):
                self._save_results(plot_samples)

        torch.save(self.solution_net, 'solution.pkl')