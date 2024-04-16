"""
Adjusted from https://github.com/li-js/gpu_memory_profiling/blob/master/gpu_profile.py
A usage example is provided in https://github.com/li-js/gpu_memory_profiling/blob/master/example_mnist.py
The debugging pytorch for cuda memory leaking is also discuessed in https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741
"""

import datetime
import linecache
from termcolor import colored
import os, os.path as osp
import pandas as pd

# Setting an environment variable to make CUDA operations synchronous, helpful for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Importing necessary libraries for GPU management and PyTorch
from py3nvml import py3nvml

import torch
import socket

from ...system import run_cmd
from ....dist import synchronize

_red_print = lambda x: print(colored(x, "red"))


def _get_filenames(dirname="."):

    cmd = f"fd --base-directory {dirname} --glob '*.py' --absolute-path --type file"
    filenames = run_cmd(cmd, verbose=False).stdout.split("\n")
    filenames = [_.strip() for _ in filenames if _.strip() != ""]
    return filenames


class CudaProfiler:
    @classmethod
    def setup(
        cls,
        profiled_filenames=_get_filenames(),
        print_tensor_sizes=False,
        use_incremental=False,
        memory_change_only=False,
        gpu_profile_fn="gpu_profile.prof.txt",
    ):
        cls.print_tensor_sizes = print_tensor_sizes
        cls.use_incremental = use_incremental
        cls.profiled_filenames = [osp.abspath(_) for _ in profiled_filenames]

        cls.last_tensor_sizes = set()
        cls.last_meminfo_used = 0
        cls.lineno = None
        cls.func_name = None
        cls.filename = None
        cls.module_name = None
        cls.memory_change_only = memory_change_only
        cls.tensor_record = dict()
        cls.checkpoints = dict()
        cls.timestamp = 0

        cls.gpu_profile_fn = gpu_profile_fn
        with open(cls.gpu_profile_fn, "w") as f:
            f.write(f"GPU memory profiling for {len(profiled_filenames)} files\n")
        # cls.gpu_profile_fn = f"Host_{socket.gethostname()}_mem_prof-{datetime.datetime.now():%d-%b-%y-%H-%M-%S}.prof.txt"
        _red_print(f"Profiling for {len(profiled_filenames)} files")
        _red_print(f"profiling gpu usage to {cls.gpu_profile_fn}")

    @classmethod
    def setup_profile_fn(cls, profile_fn):
        cls.gpu_profile_fn = profile_fn

    @classmethod
    def setup_checkpoint(cls, name="checkpoint_tensors"):
        """
        by comparing tensor ids between current and last checkpoint
        to test whether there is memory leaking
        """
        tensor_ids = set(id(x) for x in tensor_loader_gc())
        if name not in cls.checkpoints:
            _red_print(f"First checkpoint setup for {name}")
            meminfo = cls.get_meminfo()
            cls.checkpoints[name] = dict(tensor_ids=tensor_ids, mem_used=meminfo.used)
        elif cls.checkpoints[name]["mem_used"] != cls.get_meminfo().used:
            _red_print("Memory leaking detected")
            _red_print(f"Before: {cls.checkpoints[name]['mem_used']} After: {cls.get_meminfo().used}")
            diff_ids = cls.checkpoints[name].symmetric_difference(tensor_ids)
            df = cls.get_tensor_infos(diff_ids)
            import ipdb

            ipdb.set_trace()

    @classmethod
    def get_tensor_infos(cls, _ids):
        record_list = [cls.tensor_record[_id] for _id in _ids]
        df = pd.DataFrame(record_list, columns=["timestamp", "size", "type", "loc"])
        df.sort_values(by="timestamp", ascending=True, inplace=True)
        return df

    @classmethod
    def get_meminfo(cls):
        py3nvml.nvmlInit()
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")

        def _must_int(x):
            try:
                return int(x)
            except:
                raise ValueError("CUDA_VISIBLE_DEVICES must be set to a single device for profiling")

        gpu_idx = _must_int(cuda_visible_devices)

        handle = py3nvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        py3nvml.nvmlShutdown()

        return meminfo

    @classmethod
    def profile(cls, frame, event, arg):
        """
        A callback function for profiling GPU memory usage.
        This is achieved by using sys.settrace() to set this function as the trace function.

        It is triggered before each line of code is executed, allowing it to record the memory used after each line.
        """
        use_incremental = cls.use_incremental
        print_tensor_sizes = cls.print_tensor_sizes

        if event == "line":
            try:
                # Only proceed if the previous line number is set (i.e., not the first line of the script)
                if cls.lineno is not None:

                    line = linecache.getline(cls.filename, cls.lineno)
                    where_str = cls.module_name + " " + cls.func_name + ":" + str(cls.lineno)

                    meminfo = cls.get_meminfo()

                    new_meminfo_used = meminfo.used
                    # Display incremental memory usage if use_incremental is True, else total memory used
                    mem_display = new_meminfo_used - cls.last_meminfo_used if use_incremental else new_meminfo_used

                    if cls.memory_change_only and new_meminfo_used == cls.last_meminfo_used:
                        # don't log if no change in memory usage
                        return cls.profile

                    # Writing the memory usage information to the profiling log
                    with open(cls.gpu_profile_fn, "a+") as f:
                        f.write(f"{where_str:<50}: {(mem_display)/1024**2:<7.1f}Mb {line.rstrip()}\n")

                        cls.last_meminfo_used = new_meminfo_used
                        # Optionally, print sizes of tensors if flag is set
                        if print_tensor_sizes is True:
                            for tensor in tensor_loader_gc():
                                if not hasattr(tensor, "dbg_alloc_where"):
                                    tensor.dbg_alloc_where = where_str
                            new_tensor_sizes = {(type(x), tuple(x.size()), x.dbg_alloc_where, id(x)) for x in tensor_loader_gc()}
                            union = new_tensor_sizes.union(cls.last_tensor_sizes)
                            # new appearing tensors
                            for t, s, loc, _id in union.symmetric_difference(new_tensor_sizes):
                                f.write(f"+ {loc:<50} {str(s):<20} {str(t):<10} {_id}\n")
                                cls.tensor_record[_id] = dict(
                                    loc=loc,
                                    size=tuple(s),
                                    type=str(t),
                                    timestamp=cls.timestamp,
                                )

                            # deleted tensors
                            for t, s, loc, _id in union.symmetric_difference(cls.last_tensor_sizes):
                                f.write(f"- {loc:<50} {str(s):<20} {str(t):<10} {_id}\n")

                            cls.last_tensor_sizes = new_tensor_sizes

                # Preparing for the next line to be executed
                cls.lineno = None

                # Extracting function name, filename, module name, and line number from the frame for the next call
                cls.func_name = frame.f_code.co_name

                cls.filename = frame.f_globals["__file__"]
                if cls.filename.endswith(".pyc") or cls.filename.endswith(".pyo"):
                    cls.filename = cls.filename[:-1]
                cls.module_name = frame.f_globals["__name__"]
                cls.lineno = frame.f_lineno

                # Filtering out unnecessary lines from other scripts or modules for focused profiling
                if osp.abspath(cls.filename) not in cls.profiled_filenames:
                    cls.lineno = None

                cls.timestamp += 1

                return cls.profile

            except (KeyError, AttributeError):
                # In case of an exception, do nothing and just pass
                pass

        return cls.profile


def tensor_loader_gc(cuda_only=True):
    """
    Generator function to iterate over all tensors currently loaded into memory.

    If gpu_only is True, it yields only those tensors that are allocated on the GPU.
    """
    import gc

    gc.collect()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            # Yield the tensor if it's on the GPU (if gpu_only is True)
            if not cuda_only or tensor.is_cuda:
                yield tensor
        except Exception as e:
            # In case of an exception, do nothing and just pass
            pass
