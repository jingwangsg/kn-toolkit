from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import glob


def replace_reference(conf):
    for k in conf:
        if isinstance(conf[k], str):
            conf[k] = conf[k]
        elif isinstance(conf[k], DictConfig):
            replace_reference(conf[k])


def fix_relative_path(conf):
    for k in conf:
        if isinstance(conf[k], str) and k.endswith("dir"):
            if not conf[k].startswith(conf.root_dir):
                conf[k] = conf.root_dir + "/" + conf[k]
        elif isinstance(conf[k], DictConfig):
            fix_relative_path(conf[k])


def load_checkpoint_from_log(dir):
    checkpoint_paths = glob.glob(dir + "/*.ckpt")
    best_checkpoint_path = None
    best_step = -1
    for checkpoint_path in checkpoint_paths:
        step = int(checkpoint_path.split("-")[-1].split("=")[1].split(".")[0])
        if best_step < step:
            best_checkpoint_path = checkpoint_path
            best_step = step

    conf = OmegaConf.load(dir + "/.hydra/config.yaml")

    return dict(best_checkpoint_path=best_checkpoint_path, conf=conf)

