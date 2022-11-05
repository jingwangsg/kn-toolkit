from typing import Sequence, Mapping
import torch
import numpy as np


def explore_content(x, name="default", depth=0, print_str=True):
    ret_str = ""

    if isinstance(x, Sequence):
        ret_str += "\t" * depth + f"{name}(List, {len(x)} elements)\n"
        for idx, v in enumerate(x):
            ret_str += explore_content(v, name=str(idx), depth=depth + 1)  # type: ignore

    elif isinstance(x, Mapping):
        ret_str += "\t" * depth + f"{name}(Dict, {len(x)} elements)\n"
        for k, v in x.items():
            ret_str += explore_content(v, name=k, depth=depth + 1)

    elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        ret_str += "\t" * depth + f"{name}({type(x)}, {x.shape})" + "\n"

    else:
        if depth < 2:
            ret_str += "\t" * depth + f"{name}({type(x)}"
            for k, v in vars(x):
                ret_str += explore_content(v, k, depth=depth + 1)

    if depth == 0 and print_str:
        print(ret_str)

    return ret_str
