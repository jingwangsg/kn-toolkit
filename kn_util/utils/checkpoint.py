import torch
import os.path as osp
import os
import torch
import torch.nn as nn
import subprocess
import numpy as np
import glob


class CheckPointer:

    def __init__(self, monitor, work_dir, mode="min") -> None:
        self.monitor = monitor
        self.best_metric = None
        self.work_dir = work_dir
        self.mode = mode

        self.ckpt_latest = osp.join(self.work_dir, "ckpt-latest.pth")
        self.ckpt_best = osp.join(self.work_dir, "ckpt-best-ep{}-{}.pth")

    def better(self, new, orig):
        if orig is None:
            return True
        if self.mode == "min":
            return new < orig
        elif self.mode == "max":
            return new > orig
        else:
            raise NotImplementedError()

    def save_if_exists(self, obj, name, save_dict):
        if obj is not None:
            save_dict[name] = obj.state_dict()

    def save_checkpoint(self, model, optimizer, num_epochs, metric_vals=None, lr_scheduler=None):
        """save latest checkpoint only for metric_vals=None to resume latest epoch
        For metric_vals not None, update best checkpoint
        """
        save_dict = dict(model=model.state_dict(),
                         optimizer=optimizer.state_dict(),
                         num_epochs=num_epochs,
                         metrics=metric_vals)
        self.save_if_exists(lr_scheduler, "lr_scheduler", save_dict)
        torch.save(save_dict, self.ckpt_latest)

        if metric_vals:
            if self.better(metric_vals[self.monitor], self.best_metric):
                self.best_metric = metric_vals[self.monitor]
                subprocess.run(f"rm -rf {self.ckpt_best}".format('*', osp.basename(self.monitor), '*'), shell=True)
                torch.save(save_dict, self.ckpt_best.format(num_epochs, np.round(self.best_metric, decimals=6)))
                return True
        return False

    def load_checkpoint(self, model, optimizer, lr_scheduler=None, mode="latest"):
        if mode == "latest":
            fn = self.ckpt_latest
        elif mode == "best":
            ckpt_best = glob.glob(osp.join(self.work_dir, self.ckpt_best.format("*", "*")))[0]
            fn = ckpt_best
        else:
            raise NotImplementedError()
        load_dict = torch.load(fn)

        model.load_state_dict(load_dict["model"])
        optimizer.load_state_dict(load_dict["optimizer"])
        if lr_scheduler:
            if "lr_scheduler" not in load_dict:
                raise Exception("lr_scheduler not found")
            lr_scheduler.load_state_dict(load_dict["lr_scheduler"])

        return load_dict