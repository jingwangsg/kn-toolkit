import torch
import torch.nn as nn
import copy

def init_module(module, init_cfg=None):
    cfg = dict(initializer_range=0.02)
    if init_cfg: cfg.update(init_cfg)

    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=cfg["initializer_range"])
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def clones(module, N):
    modules = nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    for module in modules:
        module.apply(init_module)
    return modules
