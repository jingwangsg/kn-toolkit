import os

import debugpy
from termcolor import colored

from kn_util.dist import get_rank, synchronize
from kn_util.utils.system import run_cmd


def setup_debugpy(endpoint="localhost", port=5678, rank=0, force=False):
    if "DEBUGPY" not in os.environ:
        return
    rank = int(os.getenv("DEBUGPY_RANK", rank))
    port = int(os.getenv("DEBUGPY_PORT", port))
    endpoint = os.getenv("DEBUGPY_ENDPOINT", endpoint)
    if get_rank() != rank:
        synchronize()
        return

    # print(colored(f"rank: {get_rank()}, is_main_process: {is_main_process()}", "red"))
    if force:
        run_cmd("ps aux | grep /debugpy/adapter | awk '{print $2}' | xargs kill -9", fault_tolerance=True)
        print(colored("Force killed debugpy", "red"))
    try:
        debugpy.listen((endpoint, port))
        print(colored(f"Waiting for debugger attach on {endpoint}:{port}", "red"))
        debugpy.wait_for_client()
    except:
        print(colored(f"Failed to setup debugpy, {endpoint}:{port} occupied", "red"))

    synchronize()


# success = False

# def setup_debugpy(endpoint="localhost", port=5678, rank=0):
#     global success
#     try:
#         debugpy.listen((endpoint, port))
#         print(colored(f"Waiting for debugger attach on {endpoint}:{port}", "red"))
#         debugpy.wait_for_client()
#         success = True
#     except:
#         if success:
#             print(colored(f"Already listening on {endpoint}:{port}", "red"))
#         else:
#             print(colored(f"Failed to setup debugpy, {endpoint}:{port} occupied", "red"))
