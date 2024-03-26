from loguru import logger
from termcolor import colored
import torch
import torch.nn as nn
import torch.distributed as torch_dist

from .utils import module2tree


class ModuleMixin:
    def set_requires_grad(self, keys=None, requires_grad=True):
        param_cnt = 0
        if keys is None:
            for p in self.parameters():
                p.requires_grad = requires_grad
                param_cnt += p.numel()
        else:
            for k in keys:
                for n, p in self.named_parameters():
                    if n.startswith(k):
                        p.requires_grad = requires_grad
                        param_cnt += p.numel()
        return param_cnt

    def unfreeze(self, keys=None, verbose=True):
        param_cnt = self.set_requires_grad(keys, requires_grad=True)
        if verbose:
            logger.info(f"Unfreezing {param_cnt:,} parameters form {keys}")

    def freeze(self, keys=None, verbose=True):
        param_cnt = self.set_requires_grad(keys, requires_grad=False)
        if verbose:
            logger.info(f"Freezing {param_cnt:,} parameters from {keys}")

    @property
    def device(self):
        # FIXME: we temporarily assume all parameters are on the same device
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def param_groups(self):
        return {
            "default": {n: p for n, p in self.named_parameters() if p.requires_grad},
        }

    @property
    def num_params_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def half(self):
        for p in self.parameters():
            p.data = p.data.half()
        return self

    def to(self, *args, **kwargs):
        for p in self.parameters():
            p.data = p.data.to(*args, **kwargs)
        return self

    def pretty_format(self, list_limit=1):
        return module2tree(self, list_limit=list_limit)


class BaseModule(nn.Module, ModuleMixin):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError
