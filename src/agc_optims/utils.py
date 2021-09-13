import torch

def unitwise_norm(x):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim >= 4:
        dim = [i for i in range(1, x.ndim)]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5

def agc(param, clipping=1e-2, eps=1e-3):
    """
        Implements the adaptive gradient clipping according to the NFNet paper: https://arxiv.org/pdf/2102.06171.pdf
    """

    param_norm = torch.max(unitwise_norm(param.detach()), torch.tensor(eps).to(param.device))
    grad_norm = unitwise_norm(param.grad.detach())
    max_norm = param_norm * clipping
    trigger = grad_norm > max_norm
    clipped_grad = param.grad * (max_norm / torch.max(grad_norm,torch.tensor(1e-6).to(grad_norm.device)))
    param.grad.detach().copy_(torch.where(trigger, clipped_grad, param.grad))
