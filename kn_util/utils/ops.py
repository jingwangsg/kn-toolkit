import torch
import torch.nn as nn
import copy
from ..utils.init import init_module

def list_to_dict(cur_dict):
    """ convert a list of dict to a dict of list """
    ret_dict = {}
    for item in cur_dict:
        for k,v in item.items():
            ret_dict[k] = ret_dict.get(k, []) + [v]
    
    return ret_dict

def clones(module, N):
    modules = nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    for module in modules:
        module.apply(init_module)
    return modules


def detach_collections(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach()
    if isinstance(x, dict):
        for k in x:
            x[k] = detach_collections(x[k])
    if isinstance(x, list):
        for k in x:
            k = detach_collections(k)
    return x
