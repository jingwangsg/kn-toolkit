from loguru import logger
from termcolor import colored
import torch
import torch.nn as nn


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
