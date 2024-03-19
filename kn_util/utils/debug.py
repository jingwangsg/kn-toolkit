from IPython import embed
from ipdb import set_trace as ipdb_set_trace
from pdb import set_trace as pdb_set_trace

from typing import Any, Mapping, Sequence

try:
    import torch
    import numpy as np
except:
    pass


def explore_content(
    x,
    name="default",
    depth=0,
    max_depth=3,
    str_len_limit=20,
    list_limit=10,
    print_str=True,
):
    ret_str = ""
    sep = "    "
    if isinstance(x, str):
        if len(x) < str_len_limit:
            ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{x})\n"
        else:
            ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n"
    elif isinstance(x, int) or isinstance(x, float) or isinstance(x, bool):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{x})\n"

    elif isinstance(x, Sequence):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n"
        if not isinstance(x, str):
            for idx, v in enumerate(x):
                ret_str += explore_content(v, name=str(idx), depth=depth + 1, max_depth=max_depth)  # type: ignore
                if idx > list_limit:
                    ret_str += sep * (depth + 1) + "...\n"
                    break

    elif isinstance(x, Mapping):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n"
        for k, v in x.items():
            ret_str += explore_content(v, name=k, depth=depth + 1, max_depth=max_depth)

    elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}[{x.dtype}]{sep}{tuple(x.shape)})" + "\n"
    else:
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__})\n"
        if hasattr(x, "__dict__") and depth < max_depth:  # in case it's not a class
            # ret_str += "\t" * depth + f"{name}({type(x).__name__}"
            for k, v in vars(x).items():
                ret_str += explore_content(v, k, depth=depth + 1, max_depth=max_depth)

    if depth != 0:
        return ret_str
    else:
        if print_str:
            print(ret_str)
        else:
            return ret_str


EC = explore_content