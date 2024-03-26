from IPython import embed
from ipdb import set_trace as ipdb_set_trace
from pdb import set_trace as pdb_set_trace
import sys

from typing import Any, Mapping, Sequence, List
from icecream import argumentToString, ic

try:
    import torch
    import numpy as np
except:
    pass


def install_pdb_handler():
    """Signals to automatically start pdb:
    1. CTRL+\\ breaks into pdb.
    2. pdb gets launched on exception.
    """

    import signal
    import pdb

    def handler(_signum, _frame):
        pdb.set_trace()

    signal.signal(signal.SIGQUIT, handler)

    # Drop into PDB on exception
    # from https://stackoverflow.com/questions/13174412
    def info(type_, value, tb):
        if hasattr(sys, "ps1") or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type_, value, tb)
        else:
            import traceback
            import pdb

            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type_, value, tb)
            print()
            # ...then start the debugger in post-mortem mode.

    sys.excepthook = info


def install_pudb_handler():
    import signal
    import pudb
    import pudb.remote as pudb_remote

    def handler(_signum, _frame):
        pudb.set_trace()

    signal.signal(signal.SIGQUIT, handler)

    def info(_type, value, tb):
        # if hasattr(sys, "ps1") or not sys.stderr.isatty():
        #     sys.__excepthook__(_type, value, tb)
        # else:
        import traceback

        traceback.print_exception(_type, value, tb)
        print()
        pudb.pm()

        sys.excepthook = info


@argumentToString.register(torch.Tensor)
def _torch_tensor_to_string(x: torch.Tensor):
    return f"torch.Tensor, shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"


@argumentToString.register(np.ndarray)
def _numpy_ndarray_to_string(x: np.ndarray):
    return f"numpy.ndarray, shape={tuple(x.shape)}, dtype={x.dtype}"


# @argumentToString.register(List)
# def _list_to_string(x: List):
#     _examples = [str(v) for v in x[:3]]
#     _str = f"List, len={len(x)}, examples={_examples}"
#     return _str


def explore_content(
    x,
    name="default",
    depth=0,
    max_depth=3,
    str_len_limit=20,
    list_limit=10,
    print_str=True,
):
    ret_str = ""
    sep = "    "
    if isinstance(x, str):
        if len(x) < str_len_limit:
            ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{x})\n"
        else:
            ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n"
    elif isinstance(x, int) or isinstance(x, float) or isinstance(x, bool):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{x})\n"

    elif isinstance(x, Sequence):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n"
        if not isinstance(x, str):
            for idx, v in enumerate(x):
                ret_str += explore_content(v, name=str(idx), depth=depth + 1, max_depth=max_depth)  # type: ignore
                if idx > list_limit:
                    ret_str += sep * (depth + 1) + "...\n"
                    break

    elif isinstance(x, Mapping):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n"
        for k, v in x.items():
            ret_str += explore_content(v, name=k, depth=depth + 1, max_depth=max_depth)

    elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}[{x.dtype}]{sep}{tuple(x.shape)})" + "\n"
    else:
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__})\n"
        if hasattr(x, "__dict__") and depth < max_depth:  # in case it's not a class
            # ret_str += "\t" * depth + f"{name}({type(x).__name__}"
            for k, v in vars(x).items():
                ret_str += explore_content(v, k, depth=depth + 1, max_depth=max_depth)

    if depth != 0:
        return ret_str
    else:
        if print_str:
            print(ret_str)
        else:
            return ret_str


EC = explore_content
