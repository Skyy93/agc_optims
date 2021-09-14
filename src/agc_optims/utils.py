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
    param_data = param.detach()
    grad_data = param.grad.detach()
    max_norm = unitwise_norm(param_data).clamp_(min=eps).mul_(clipping) # Clamp works faster than .max()! But iff min= is a float not a Tensor!
    grad_norm = unitwise_norm(grad_data)
    clipped_grad = grad_data.mul_((max_norm.div_(grad_norm.clamp(min=1e-6))))
    new_grads = torch.where(grad_norm < max_norm, grad_data, clipped_grad)
    param.grad.detach().copy_(new_grads)

