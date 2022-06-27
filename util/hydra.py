from omegaconf import DictConfig


def replace_reference(conf):
    for k in conf:
        if isinstance(conf[k], str):
            conf[k] = conf[k]
        elif isinstance(conf[k], DictConfig):
            replace_reference(conf[k])
