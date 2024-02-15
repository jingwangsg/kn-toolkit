from loguru import logger
from termcolor import colored
from .mixin_util import _find_module_by_keys, module2tree
import torch

class ModelMixin:

    def set_requires_grad(self, keys=None, requires_grad=True):
        if keys is None:
            for p in self.parameters():
                p.requires_grad = requires_grad
        else:
            for k in keys:
                for n, p in self.named_parameters():
                    if n.startswith(k):
                        p.requires_grad = requires_grad

    def unfreeze(self, keys=None):
        self.set_requires_grad(keys, requires_grad=True)

    def freeze(self, keys=None):
        self.set_requires_grad(keys, requires_grad=False)

    @property
    def device(self):
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
            if p.dtype == torch.float32:
                p.data = p.data.to(*args, **kwargs)
        return self

    def pretty_format(self, list_limit=1):
        return module2tree(self, list_limit=list_limit)
