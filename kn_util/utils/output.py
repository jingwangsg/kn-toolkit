from typing import Sequence, Mapping
import numpy as np
import torch

import torch.nn as nn
from termcolor import colored
from omegaconf import OmegaConf
from kn_util.config.lazy import _convert_target_to_string
import yaml


def dict2str(x, delim=": ", sep="\n", fmt=".2f", exclude_keys=[]):
    """
    Convert a dictionary to a string

    Parameters
    ----------
    x : dict
        Dictionary to be converted to a string

    Returns
    -------
    str
        String representation of the dictionary
    """
    kv_list = []
    for k, v in x.items():
        sv = "{:{}}".format(v, fmt)
        if k not in exclude_keys:
            kv_list.append("{k}{delim}{v}".format(k=k, delim=delim, v=sv))
        else:
            kv_list.append("{k}{delim}{v}".format(k=k, delim=delim, v=v))

    return sep.join(kv_list)


def max_memory_allocated():
    """
    Get the maximum memory allocated by pytorch

    Returns
    -------
    int
        Maximum memory allocated by pytorch
    """
    import torch
    return torch.cuda.max_memory_allocated() / 1024.0 / 1024.0


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = ["  |" + (numSpaces - 2) * ' ' + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def replace_target(cfg_dict):
    if "_target_" in cfg_dict:
        try:
            cfg_dict["_target_"] = _convert_target_to_string(cfg_dict["_target_"])
        except:
            cfg_dict["_target_"] = str(cfg_dict["_target_"])

    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            replace_target(v)


def lazyconf2str(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    replace_target(cfg_dict)
    yaml_data = yaml.dump(cfg_dict, indent=4)
    return yaml_data


def module2tree(rt_module: nn.Module, print_limit_list=1):
    # we treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = rt_module.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in rt_module._modules.items():
        mod_str = module2tree(module, print_limit_list=print_limit_list)
        mod_str = _addindent(mod_str, 4)
        if rt_module._get_name() in ("Sequential", "ModuleList") and int(key) == print_limit_list and mod_str:
            child_lines.append(
                colored("|- ... ({} children)".format(len(rt_module._modules) - print_limit_list), "grey"))
            break
        child_lines.append(colored("|-" + '(' + key + '): ', "blue", attrs=["blink", "bold"]) + mod_str)
    lines = extra_lines + child_lines

    main_str = colored(rt_module._get_name(), "green", attrs=["blink", "bold"]) + '('
    if lines:
        # simple one-liner info, which most builtin modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += '  )'
    return main_str


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
            ret_str += (sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n")
    elif isinstance(x, int) or isinstance(x, float) or isinstance(x, bool):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{x})\n"

    elif isinstance(x, Sequence):
        ret_str += (sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n")
        if not isinstance(x, str):
            for idx, v in enumerate(x):
                ret_str += explore_content(v, name=str(idx), depth=depth + 1, max_depth=max_depth)  # type: ignore
                if idx > list_limit:
                    ret_str += sep * (depth + 1) + "...\n"
                    break

    elif isinstance(x, Mapping):
        ret_str += (sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n")
        for k, v in x.items():
            ret_str += explore_content(v, name=k, depth=depth + 1, max_depth=max_depth)

    elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        ret_str += (sep * depth + f"{name}{sep}({type(x).__name__}[{x.dtype}]{sep}{tuple(x.shape)})" + "\n")
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
