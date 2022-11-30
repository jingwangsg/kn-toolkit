import torch
import os.path as osp
import os
import torch
import torch.nn as nn


class CheckPoint:

    def __init__(self, monitor, work_dir, mode="min") -> None:
        self.monitor = monitor
        self.best_metric = None
        self.work_dir = work_dir
        self.mode = mode

        self.ckpt_latest = osp.join(self.work_dir, "ckpt.latest.pth")
        self.ckpt_best = osp.join(self.work_dir, "ckpt.best.pth")

    def better(self, new, orig):
        if orig is None:
            return True
        if self.mode == "min":
            return new < orig
        elif self.mode == "max":
            return new > orig
        else:
            raise NotImplementedError()

    def save_checkpoint(self,
                        model,
                        optimizer,
                        num_epochs,
                        metric_vals,
                        lr_scheduler=None):
        save_dict = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            num_epochs=num_epochs,
            metrics=metric_vals)
        if lr_scheduler:
            save_dict["lr_scheduler"] = lr_scheduler.save_dict()
        torch.save(save_dict, self.ckpt_latest)
        if self.better(metric_vals[self.monitor], self.best_metric):
            self.best_metric = metric_vals[self.monitor]
            os.symlink(self.ckpt_latest, self.ckpt_best)

    def load_checkpoint(self,
                        model,
                        optimizer,
                        lr_scheduler=None,
                        mode="latest"):
        if mode == "latest":
            fn = self.ckpt_latest
        elif mode == "best":
            fn = self.ckpt_best
        else:
            raise NotImplementedError()
        load_dict = torch.load(fn)
        model.load_state_dict(load_dict["model"])
        load_dict["model"] = model
        optimizer.load_state_dict(load_dict["optimizer"])
        load_dict["optimizer"] = optimizer
        if lr_scheduler:
            if "lr_scheduler" not in load_dict:
                raise Exception("lr_scheduler not found")
            lr_scheduler.load_state_dict(load_dict["lr_scheduler"])
            load_dict["lr_scheduler"] = lr_scheduler

        return load_dict