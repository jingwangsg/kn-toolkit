import copy
from omegaconf import DictConfig

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