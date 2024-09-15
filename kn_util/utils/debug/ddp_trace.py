import inspect
from contextlib import contextmanager

import torch.distributed as dist

from ...dist import get_rank, synchronize

ignored = False


class SyncContextManager:
    def __enter__(self):
        self.frame = inspect.currentframe().f_back
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def sync(self):
        dist.barrier()


def sync_decorator(func):
    def wrapper(*args, **kwargs):
        source = inspect.getsource(func)
        dist.barrier()

        lines = source.split("\n")

        end_of_func_args = None
        for idx, line in enumerate(lines):
            if line.endswith("):"):
                end_of_func_args = idx
                break

        lineno_start = end_of_func_args + 1
        indent = len(lines[lineno_start]) - len(lines[lineno_start].lstrip())
        code_lines = [line[indent:] for line in lines[lineno_start:]]  # Remove the indentation

        locals_last_frame = inspect.currentframe().f_back.f_locals
        for line in code_lines:
            exec(line, globals(), locals_last_frame)
            dist.barrier()

    return wrapper


def ddp_synchronize_trace(frame, event, arg):
    if event == "line" and not ignored:
        synchronize()

    return ddp_synchronize_trace


@contextmanager
def disable_synchronize_trace():
    global ignored
    ignored = True
    yield
    ignored = False


@contextmanager
def rank_only(rank=0):
    # for code wrapped in this context
    # 1. it will be only executed in specified rank
    # 2. it won't be traced by ddp_synchronize_trace to prevent blocking
    global ignored
    if get_rank() == rank:
        ignored = True
        yield
        ignored = False
    else:
        return
