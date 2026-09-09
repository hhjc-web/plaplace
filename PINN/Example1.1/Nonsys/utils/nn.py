from typing import Callable, List

import torch
from torch import Tensor
import torch.nn as nn

import copy


def ReLU2Grad(x):
    return 0.2 * torch.log(torch.exp(x) + 1.0) * torch.exp(x) / (torch.exp(x)+1)

def ReLU2(x):
    return 0.1 * torch.log(torch.exp(x) + 1.0)**2


def tanhGrad(x):
    return 1.0 - torch.pow(nn.Tanh()(x), 2.0)


def get_network(net_type, activation, net_dim, out_dim, **kwargs):
    # activation function
    if activation == "relu":
        activation = ReLU3
    elif activation == "relu3":
        activation = ReLU3
        gradactivation = ReLU3h
    elif activation == "sigmoid":
        activation = nn.Sigmoid()
    elif activation == "tanh":
        activation = nn.Tanh()
        gradactivation = tanhGrad
    else:
        raise ValueError("Invalid value for 'activation': %s" % activation)
    # net type
    if net_type == "mlp":
        return MLP(
            in_features=net_dim, out_features=out_dim, activation=activation, **kwargs
        )
    elif net_type == "mynet":
        return MyNet(
            in_features=net_dim, out_features=out_dim, activation=activation, **kwargs
        )
    elif net_type == "GradMLP":
        return GradMLP(
            in_features=net_dim,
            out_features=out_dim,
            activation=activation,
            activationGrad=gradactivation,
            **kwargs
        )
    else:
        raise ValueError("Invalid value for 'net_type': %s" % net_type)


@torch.no_grad()
def _freeze_params(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        m.weight.requires_grad_(False)
        m.bias.requires_grad_(False)


@torch.no_grad()
def clone_freeze(net: nn.Module) -> None:
    net_copy = copy.deepcopy(net)
    net_copy.apply(_freeze_params)
    return net_copy


@torch.no_grad()
def _init_weights_bias(m: nn.Module, std=1.0) -> None:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=std, mean=0.0)
        nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        width_list: List[int],
        activation: Callable[[Tensor], Tensor],
        scale=1.0,
    ) -> None:
        super(MLP, self).__init__()
        self.in_layer = nn.Linear(in_features, width_list[0])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in zip(width_list, width_list[1:])
            ]
        )
        self.out_layer = nn.Linear(width_list[-1], out_features)
        self.act = activation
        self.scale = scale
        self.apply(_init_weights_bias)
        self.baseIndex = -1
        self.baseDim = width_list[-1]

    def forward(self, x: Tensor):
        x = self.in_layer(x)
        x = self.act(x)
        for layer in self.hiddens:
            x = layer(x)
            x = self.act(x)
        out = self.out_layer(x)
        out *= self.scale
        return out

    def basefunc(self, x: Tensor):
        if self.baseIndex == -1:
            return self.forward(x)
        elif self.baseIndex == self.baseDim:
            return torch.ones(x.shape[0], 1) + x[:, 0:1] - x[:, 0:1]
        else:
            x = self.in_layer(x)
            x = self.act(x)
            for layer in self.hiddens:
                x = layer(x)
                x = self.act(x)
            return x

    def freezeLayer(self, k: int):
        assert k > -1
        assert k <= len(self.hiddens)
        if k == 0:
            self.in_layer.weight.requires_grad_(False)
            self.in_layer.bias.requires_grad_(False)
        else:
            self.hiddens[k - 1].weight.requires_grad_(False)
            self.hiddens[k - 1].bias.requires_grad_(False)

    def releaseLayer(self, k: int):
        assert k > -1
        assert k <= len(self.hiddens)
        if k == 0:
            self.in_layer.weight.requires_grad_(True)
            self.in_layer.bias.requires_grad_(True)
        else:
            self.hiddens[k - 1].weight.requires_grad_(True)
            self.hiddens[k - 1].bias.requires_grad_(True)


class GradMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        width_list: List[int],
        activation: Callable[[Tensor], Tensor],
        activationGrad: Callable[[Tensor], Tensor],
        dropoutRate=0.0,
        scale=1.0,
    ) -> None:
        super(GradMLP, self).__init__()
        self.inLayerWeight = nn.Parameter(torch.randn([in_features, width_list[0]]))
        self.inLayerBias = nn.Parameter(torch.zeros([width_list[0]]))
        self.hiddensWeight = nn.ParameterList()
        self.hiddensBias = nn.ParameterList()
        for k in range(len(width_list) - 1):
            self.hiddensWeight.append(torch.randn([width_list[k], width_list[k + 1]]))
            self.hiddensBias.append(torch.zeros([width_list[k + 1]]))
        self.outLayerWeight = nn.Parameter(torch.zeros([width_list[-1], out_features]))
        self.outLayerBias = nn.Parameter(torch.zeros([out_features]))
        self.act = activation
        self.actGrad = activationGrad
        self.scale = scale
        self.baseIndex = -1
        self.baseDim = width_list[-1]
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropoutRate) for _ in range(len(width_list))]
        )

    def oneLayerFunc(self, x: Tensor, w: Tensor, b: Tensor):
        out = torch.matmul(x, w) + b
        out = self.act(out)
        return out

    def forward(self, x: Tensor):
        out = self.oneLayerFunc(x, self.inLayerWeight, self.inLayerBias)
        out = self.dropouts[0](out)
        for k in range(len(self.hiddensWeight)):
            out = self.oneLayerFunc(out, self.hiddensWeight[k], self.hiddensBias[k])
            out = self.dropouts[k + 1](out)
        out = torch.matmul(out, self.outLayerWeight) + self.outLayerBias
        return out

    def Grad(self, x: Tensor):
        out = torch.matmul(x, self.inLayerWeight) + self.inLayerBias
        gradx = torch.unsqueeze(self.actGrad(out), dim=-1) * self.inLayerWeight.t()
        out = self.act(out)
        out = self.dropouts[0](out)
        for k in range(len(self.hiddensWeight)):
            out = torch.matmul(out, self.hiddensWeight[k]) + self.hiddensBias[k]
            gradx = torch.matmul(
                torch.unsqueeze(self.actGrad(out), dim=-1) * self.hiddensWeight[k].t(),
                gradx,
            )
            out = self.act(out)
            out = self.dropouts[k + 1](out)
        gradx = torch.matmul(self.outLayerWeight.t(), gradx)
        return gradx

    def Gradbase(self, x: Tensor):
        out = torch.matmul(x, self.inLayerWeight) + self.inLayerBias
        gradx = torch.unsqueeze(self.actGrad(out), dim=-1) * self.inLayerWeight.t()
        out = self.act(out)
        out = self.dropouts[0](out)
        for k in range(len(self.hiddensWeight)):
            out = torch.matmul(out, self.hiddensWeight[k]) + self.hiddensBias[k]
            gradx = torch.matmul(
                torch.unsqueeze(self.actGrad(out), dim=-1) * self.hiddensWeight[k].t(),
                gradx,
            )
        return gradx

    def basefunc(self, x: Tensor):
        out = self.oneLayerFunc(x, self.inLayerWeight, self.inLayerBias)
        out = self.dropouts[0](out)
        for k in range(len(self.hiddensWeight)):
            out = self.oneLayerFunc(out, self.hiddensWeight[k], self.hiddensBias[k])
            out = self.dropouts[k + 1](out)
        return out


class MyNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        width_list: List[int],
        activation: Callable[[Tensor], Tensor],
        r0,
        scale=1.0,
    ) -> None:
        super().__init__()
        width_list_plus = []
        for k in range(len(width_list)):
            width_list_plus.append(width_list[k] + 1)
        self.in_layer = nn.Linear(in_features + 1, width_list[0])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in zip(width_list_plus, width_list[1:])
            ]
        )
        self.r0 = r0
        self.out_layer = nn.Linear(width_list_plus[-1], out_features)
        self.act = activation
        self.scale = scale
        self.apply(_init_weights_bias)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)
        self.baseIndex = -1
        self.baseDim = width_list[-1]

    def forward(self, x: Tensor, place):
        if place == "in":
            index = x[:, 0:1] - x[:, 0:1] - 1
        elif place == "out":
            index = x[:, 0:1] - x[:, 0:1] + 1
        else:
            raise ValueError("Invalid value for 'place': %s" % place)
        identity = torch.cat([x, index], dim=1)
        x = self.in_layer(identity)
        x = self.act(x)
        for layer in self.hiddens:
            x = torch.cat([x, index], dim=1)
            x = layer(x)
            x = self.act(x)
        x = torch.cat([x, index], dim=1)
        out = self.out_layer(x)
        out *= self.scale
        return out

    def basefunc(self, x: Tensor, place):
        if place == "in":
            index = torch.sum(x**2, dim = 1, keepdim = True) - self.r0**2
        elif place == "out":
            index = x[:, 0:1] - x[:, 0:1] + 1
        else:
            raise ValueError("Invalid value for 'place': %s" % place)
        identity = torch.cat([x, index], dim=1)
        x = self.in_layer(identity)
        x = self.act(x)
        for layer in self.hiddens:
            x = torch.cat([x, index], dim=1)
            x = layer(x)
            x = self.act(x)
        x = torch.cat([x, index], dim=1)
        return x

    def freezeLayer(self, k: int):
        assert k > -1
        assert k <= len(self.hiddens)
        if k == 0:
            self.in_layer.weight.requires_grad_(False)
            self.in_layer.bias.requires_grad_(False)
        else:
            self.hiddens[k - 1].weight.requires_grad_(False)
            self.hiddens[k - 1].bias.requires_grad_(False)

    def releaseLayer(self, k: int):
        assert k > -1
        assert k <= len(self.hiddens)
        if k == 0:
            self.in_layer.weight.requires_grad_(True)
            self.in_layer.bias.requires_grad_(True)
        else:
            self.hiddens[k - 1].weight.requires_grad_(True)
            self.hiddens[k - 1].bias.requires_grad_(True)
