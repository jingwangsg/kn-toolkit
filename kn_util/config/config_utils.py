from .lazy import LazyCall as L

import copy
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Dict


def serializable(_config, depth=0):
    if depth == 0:
        config = copy.deepcopy(_config)
    else:
        config = _config
    if not isinstance(config, DictConfig):
        return
    else:
        for k in config.keys():
            if isinstance(config[k], type):
                config[k] = str(config)
                return

            serializable(config[k], depth + 1)
    if depth == 0:
        return config


def eval_str_impl(s):
    return eval(s)


def eval_str(s):
    return L(eval_str_impl)(s=s)


from hydra.utils import instantiate as _instantiate
import copy


def instantiate(_cfg, _convert_="none", **kwargs):
    cfg = copy.deepcopy(_cfg)
    cfg.update(kwargs)
    return _instantiate_manual(cfg, _convert_=_convert_)
    # return _instantiate(cfg, _convert_=_convert_)


def _instantiate_manual(_cfg, **kwargs):
    kwargs.pop("_convert_", None)
    # cfg = copy.copy(_cfg)
    cfg = _cfg
    is_iterable = hasattr(cfg, "__iter__") and not isinstance(cfg, str)

    if not is_iterable:
        return cfg
    
    if isinstance(cfg, ListConfig):
        return [_instantiate_manual(x) for x in cfg]

    if isinstance(cfg, DictConfig):
        cfg = {k: v for k, v in cfg.items()}
        target = cfg.pop("_target_", None)

        for param in cfg:
            cfg[param] = _instantiate_manual(cfg[param])

        if target:
            return target(**cfg, **kwargs)

    return cfg
