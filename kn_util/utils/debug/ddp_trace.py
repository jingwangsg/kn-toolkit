from ...dist import synchronize, get_rank
from contextlib import contextmanager

ignored = False


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
