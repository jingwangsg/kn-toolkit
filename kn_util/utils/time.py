import torch
import time
from loguru import logger
from contextlib import contextmanager


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


@contextmanager
def timer_ctx(desc=None):
    torch.cuda.synchronize()
    st = time.time()
    yield
    torch.cuda.synchronize()
    logger.info(f"{desc}: {time.time() - st:.3f} sec")
