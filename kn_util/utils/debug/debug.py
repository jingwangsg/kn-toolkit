try:
    import torch
    import torch.nn as nn
    import numpy as np
    from torch import Tensor, memory_format

except:
    pass

from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
)

from IPython import embed
from ipdb import set_trace as ipdb_set_trace
from pdb import set_trace as pdb_set_trace
import sys

from typing import Any, Mapping, Sequence, List
from icecream import argumentToString, ic
import gc
import inspect
import copy
from termcolor import colored
import warnings
import functools
from typing import Union, Optional, Tuple, Callable, Dict


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
