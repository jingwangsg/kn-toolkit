from typing import Sequence, Mapping
import torch
import copy
import numpy as np


def explore_content(x, name="default", depth=0, max_depth=2, print_str=False):
    ret_str = ""

    if isinstance(x, Sequence):
        ret_str += "\t" * depth + f"{name}\t({type(x).__name__}\t{len(x)} elements)\n"
        if not isinstance(x, str):
            for idx, v in enumerate(x):
                ret_str += explore_content(v, name=str(idx), depth=depth + 1, max_depth=max_depth)  # type: ignore

    elif isinstance(x, Mapping):
        ret_str += "\t" * depth + f"{name}\t({type(x).__name__}\t{len(x)} elements)\n"
        for k, v in x.items():
            ret_str += explore_content(v, name=k, depth=depth + 1, max_depth=max_depth)

    elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        ret_str += "\t" * depth + f"{name}\t({type(x).__name__}[{x.dtype}]\t{tuple(x.shape)})" + "\n"
    else:
        ret_str += "\t" * depth + f"{name}\t({type(x).__name__})\n"
        if hasattr(x, "__dict__") and depth < max_depth:  # in case it's not a class
            ret_str += "\t" * depth + f"{name}({type(x).__name__}"
            for k, v in vars(x).items():
                ret_str += explore_content(v, k, depth=depth + 1, max_depth=max_depth)

    if depth == 0 and print_str:
        print(ret_str)

    return ret_str


def dict_diff(before, now):
    add = dict()
    delete = dict()
    all_keys = set(before.keys()) | set(now.keys())

    for k in all_keys:
        if k in now and k in before:
            if id(now[k]) != id(before[k]):
                add[k] = now[k]
                delete[k] = before[k]
        elif k not in before and k in now:
            add[k] = now[k]
        elif k in before and k not in now:
            delete[k] = before[k]
        else:
            raise Exception()

    delete_str = explore_content(delete, "DELETED", max_depth=0)
    add_str = explore_content(add, "ADDED", max_depth=0)

    del add
    del delete
    del all_keys

    return "-" * 60 + "\n" + delete_str + "+" * 60 + "\n" + add_str
