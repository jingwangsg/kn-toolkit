from kn_util.data import fix_tensor_to_float32, merge_list_to_tensor
import numpy as np
import torch.distributed as dist
import warnings
import os
_warn_flag = False

def default_collate_fn(x):
    try:
        return fix_tensor_to_float32(merge_list_to_tensor(x))
    except:
        global _warn_flag
        if not _warn_flag:
            warnings.warn("part of outputs are not tensor")
            _warn_flag = True
        return x

def pad_to_multiple_of(datapipe, divisor):
    if not hasattr(datapipe, "__len__"):
        datapipe.set_length(len(list(datapipe)))
    to_len = int(np.ceil(len(datapipe) / divisor)) * divisor
    datapipe.cycle(2).header(to_len)
    return datapipe
    

def prepare_for_ddp(datapipe, shuffle=True):
    if not dist.is_initialized() or not dist.is_available():
        return datapipe
    
    world_size = os.environ["WORLD_SIZE"]
    if shuffle:
        datapipe = datapipe.shuffle()
    datapipe = pad_to_multiple_of(datapipe, world_size)
    datapipe = datapipe.sharding_filter()

    return datapipe
