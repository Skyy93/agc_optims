import torch
from torch.optim.optimizer import Optimizer
from agc_optims.utils import agc

class AGC(Optimizer):
    r"""Implements adaptive gradient clipping independent from the optimizer.
    This implementation of AGC adapted from https://github.com/vballoli/nfnets-pytorch.
    Args:
        optimizer (Optimizer): Optimizer that inherits the Optimizer class from PyTorch
        clipping (float, optional): clipping value for the AGC (default: 1e-2)
        agc_eps (float, optional): term used in agc to prevent grads clipped to zero (default: 1e-3)
    .. _High-Performance Large-Scale Image Recognition Without Normalization 
        https://arxiv.org/abs/2102.06171
    """
    def __init__(self, optimizer, clipping=1e-2, agc_eps=1e-3):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("No optimizer is given, inserted: {}".format(type(optimizer)))
        if not 0.0 <= clipping < 1.0:
            raise ValueError("Invalid clipping parameter: {}".format(clipping))
        if not 0.0 <= agc_eps:
            raise ValueError("Invalid agc_eps value: {}".format(agc_eps))
        self.optimizer = optimizer
        defaults = dict(clipping=clipping, agc_eps=agc_eps)
        defaults = {**self.optimizer.defaults, **defaults}
        param_groups = self.optimizer.__getstate__()['param_groups']
        super(AGC, self).__init__(param_groups, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            clipping = group['clipping']
            agc_eps = group['agc_eps']
            for p in group['params']:
                agc(param=p, clipping=clipping, eps=agc_eps)
        
        self.optimizer.step(closure)

        return loss
