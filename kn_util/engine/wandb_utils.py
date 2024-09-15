import wandb
from omegaconf import OmegaConf

from ..dist import is_main_process


def setup_wandb(project="", exp="", cfg=None, enable=True):
    mode = "online" if is_main_process() and enable else "disabled"
    wandb.init(
        project=project,
        name=exp,
        config=OmegaConf.to_container(cfg),
        mode=mode,
    )
