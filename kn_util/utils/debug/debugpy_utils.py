from termcolor import colored
import sys
from kn_util.dist import is_main_process, synchronize, get_rank

import debugpy

success = False


def setup_debugpy(endpoint="localhost", port=5678, rank=0):
    global success
    # print(colored(f"rank: {get_rank()}, is_main_process: {is_main_process()}", "red"))
    try:
        if get_rank() == rank:
            debugpy.listen((endpoint, port))
            print(colored(f"Waiting for debugger attach on {endpoint}:{port}", "red"))
            debugpy.wait_for_client()
        success = True
    except:
        if success:
            print(colored(f"Already listening on {endpoint}:{port}", "red"))
        else:
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
