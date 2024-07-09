from termcolor import colored
import sys
from kn_util.dist import is_main_process, synchronize

try:
    import debugpy

    def setup_debugpy(endpoint="localhost", port=5678):
        if is_main_process():
            debugpy.listen((endpoint, port))
            print(colored(f"Waiting for debugger attach on {endpoint}:{port}", "red"))
            debugpy.wait_for_client()
        synchronize()

except ImportError:
    def setup_debugpy(endpoint="localhost", port=5678):
        raise NotImplementedError("debugpy is not installed. Please install it with `pip install debugpy`")
