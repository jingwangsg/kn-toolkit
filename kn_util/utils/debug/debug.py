try:
    import numpy as np
    import torch

except:
    pass

import sys

from icecream import argumentToString


def install_pdb_handler():
    """Signals to automatically start pdb:
    1. CTRL+\\ breaks into pdb.
    2. pdb gets launched on exception.
    """

    import pdb
    import signal

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

            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type_, value, tb)
            print()
            # ...then start the debugger in post-mortem mode.

    sys.excepthook = info


def install_pudb_handler():
    import signal

    import pudb

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
