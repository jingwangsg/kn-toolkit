"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import time
from collections import defaultdict, deque
import sys

from collections import defaultdict
from loguru import logger
import wandb


try:
    import torch
except:
    pass

from ..dist import is_main_process
import torch.distributed as torch_dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        from .dist_utils import is_dist_avail_and_initialized, is_main_process
        import torch
        import torch.distributed as dist

        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):

    def __init__(self, delimiter="\t", logger=None, start_iter=0):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.start_iter = start_iter

        if logger is None:
            from loguru import logger

            self.logger = logger
        else:
            self.logger = logger

    def update(self, **kwargs):
        import torch

        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.global_avg))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, log_freq, header=None, wandb_kwargs=None):
        if wandb_kwargs is not None:
            assert "prefix" in wandb_kwargs, "prefix is required in wandb_kwargs"
            assert "start_iter" in wandb_kwargs, "start_iter is required in wandb_kwargs"

        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_template = [
            header,
            "[{niter" + space_fmt + "}/{total_iter}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_template.append("max mem: {memory:.0f}")
        log_template = self.delimiter.join(log_template)

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % log_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                metric_dict = dict(
                    niter=i,
                    total_iter=len(iterable),
                    eta=eta_string,
                    meters=str(self),
                    time=str(iter_time),
                    data=str(data_time),
                )

                if torch.cuda.is_available():
                    metric_dict["memory"] = torch.cuda.max_memory_reserved() / MB

                if wandb_kwargs is not None:
                    prefix = wandb_kwargs["prefix"]
                    wandb_metric = {f"{prefix}/{k}": v for k, v in metric_dict.items()}
                    niter = i + wandb_kwargs["start_iter"]
                    wandb.log(**wandb_metric, step=niter)

                log_str = log_template.format(**metric_dict)
                self.logger.info(log_str)

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable)))


def setup_logger_logging():
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def setup_logger_loguru(
    logger=logger,
    name=None,
    filename=None,
    stdout=True,
    include_function=False,
    include_filepath=False,
    master_only=True,
):
    # when filename = None and stdout=False, loguru will not log anything
    # this is espeically useful for distributed training

    template = ""
    if name is not None:
        template += "{name}|"

    template += "<green>{time:YY-MM-DD HH:mm:ss}</green>|<blue>{level}</blue>"
    if include_filepath:
        template += "<cyan>> {file.path}({line})</cyan>\n\033[1m=>\033[0m"
    if include_function:
        template += "<cyan>{function}</cyan>"
    template += " \033[1m{message}\033[0m"

    logger.remove(0)
    if master_only:
        if not torch_dist.is_initialized():
            print("[WARNING] torch distributed is not initialized before setting up logger")
            print("master_only will be ignored")
        if not is_main_process():
            return
    if filename is not None:
        logger.add(filename, level="INFO", format=template, enqueue=True)
    if stdout:
        logger.add(sys.stdout, format=template, enqueue=True)
