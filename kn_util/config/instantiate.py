from hydra.utils import instantiate as _instantiate
import copy

def instantiate(_cfg, _convert_="none", **kwargs):
    cfg = copy.deepcopy(_cfg)
    cfg.update(kwargs)
    return _instantiate(cfg, _convert_=_convert_)
    