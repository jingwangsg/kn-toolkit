import torch.nn as nn
from termcolor import colored
from loguru import logger
from termcolor import colored
import torch
import torch.nn as nn
import torch.distributed as torch_dist


def unwrap_model(model):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _find_module_by_keys(model, keys):
    lora_modules = set()
    for k in keys:
        for n, m in model.named_modules():
            if n.startswith(k):
                lora_modules.add(m)
    return list(lora_modules)


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = ["  |" + (numSpaces - 2) * " " + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def module2tree(rt_module: nn.Module, list_limit=None):
    # we treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = rt_module.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split("\n")
    child_lines = []
    is_list = rt_module._get_name() in ("Sequential", "ModuleList")
    rep_cnt = 0
    last_module = None
    for key, module in rt_module._modules.items():
        mod_str = module2tree(module, list_limit=list_limit)
        mod_str = _addindent(mod_str, 4)
        if not is_list:
            child_lines.append(colored("|-" + "(" + key + "): ", "blue", attrs=["blink", "bold"]) + mod_str)
        else:
            if module.__class__ != last_module:
                if rep_cnt >= list_limit:
                    child_lines.append(colored(f"|- ... ({module.__class__.__name__} repeat for {rep_cnt  - list_limit} time(s))", "grey"))
                rep_cnt = 1
                last_module = module.__class__
                child_lines.append(colored("|-" + "(" + key + "): ", "blue", attrs=["blink", "bold"]) + mod_str)
            else:
                if rep_cnt < list_limit:
                    child_lines.append(colored("|-" + "(" + key + "): ", "blue", attrs=["blink", "bold"]) + mod_str)
                rep_cnt += 1

    if rep_cnt >= list_limit:
        child_lines.append(colored(f"|- ... ({module.__class__.__name__} repeat for {rep_cnt  - list_limit} time(s))", "grey"))

    lines = extra_lines + child_lines

    main_str = colored(rt_module._get_name(), "green", attrs=["blink", "bold"]) + "("
    if lines:
        # simple one-liner info, which most builtin modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

    main_str += "  )"
    return main_str


def set_requires_grad(model, keys=None, requires_grad=True):
    param_cnt = 0
    if keys is None:
        for p in model.parameters():
            p.requires_grad = requires_grad
            param_cnt += p.numel()
    else:
        for k in keys:
            for n, p in model.named_parameters():
                if n.startswith(k):
                    p.requires_grad = requires_grad
                    param_cnt += p.numel()
    return param_cnt


def unfreeze(model, keys=None, verbose=True):
    param_cnt = set_requires_grad(model, keys, requires_grad=True)
    if verbose:
        logger.info(f"Unfreezing {param_cnt:,} parameters from {keys}")


def freeze(model, keys=None, verbose=True):
    param_cnt = set_requires_grad(model, keys, requires_grad=False)
    if verbose:
        logger.info(f"Freezing {param_cnt:,} parameters from {keys}")


def get_device(model):
    return next(model.parameters()).device


def get_dtype(model):
    return next(model.parameters()).dtype


def get_named_parameters(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def get_num_params_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def convert_to_half(model):
    for p in model.parameters():
        p.data = p.data.half()
    return model


def convert_to(model, *args, **kwargs):
    for p in model.parameters():
        p.data = p.data.to(*args, **kwargs)
    return model


def pretty_format(model, list_limit=1):
    return module2tree(model, list_limit=list_limit)
