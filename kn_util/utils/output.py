
import torch
import yaml
from omegaconf import OmegaConf

from kn_util.config.lazy import _convert_target_to_string


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

