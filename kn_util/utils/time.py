import time
from contextlib import contextmanager

import torch
from loguru import logger

import kn_util.dist as dist


class Timer:
    def __init__(self):
        self.records = dict()

    def start(self, key):
        st = time.time()
        self.records[key] = dict(accumulated=0, st=st, counts=0)

    def stop(self, key):
        torch.cuda.synchronize()
        self.records[key] = time.time() - self.records[key]
        logger.info(f"{key}: {self.records[key]}")


disable_timer = False

@contextmanager
def timer_ctx(desc=None, master_only=True):
    global disable_timer
    if disable_timer:
        yield
        return
    torch.cuda.synchronize()
    st = time.time()
    yield
    torch.cuda.synchronize()
    if master_only and dist.get_rank() == 0:
        print(f"{desc}: {time.time() - st:.3f} sec")
    else:
        RANK = dist.get_rank()
        print(f"RANK#{RANK} {desc}: {time.time() - st:.3f} sec")
