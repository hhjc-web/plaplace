from torch import optim


def get_optim_adam(params, lr: float, weight_decay: float):
    return optim.Adam(params = filter(lambda p: p.requires_grad, params), lr=lr, weight_decay=weight_decay)
    # return optim.SGD(params = params, lr = lr, weight_decay = weight_decay)

def get_optim_sgd(params, lr: float, weight_decay: float):
    # return optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    return optim.SGD(params = filter(lambda p: p.requires_grad, params), lr = lr, weight_decay = weight_decay)