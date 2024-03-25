import copy
import torch.nn as nn
import torch.nn.functional as F


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_params_by_prefix(state_dict, prefix=None, remove_prefix=False):
    if prefix is None:
        return state_dict
    if remove_prefix:
        return {k[len(prefix) :].strip("."): v for k, v in state_dict.items() if k.startswith(prefix)}
    else:
        return {k: v for k, v in state_dict.items() if k.startswith(prefix)}
