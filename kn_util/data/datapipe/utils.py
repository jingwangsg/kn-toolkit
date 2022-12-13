from kn_util.data import fix_tensor_to_float32, merge_list_to_tensor
import numpy as np
import torch.distributed as dist
import torch
import warnings
import os
from kn_util.basic import global_get
from kn_util.basic import Signal


def default_collate_fn(x):
    non_tensor_dict = dict()
    ks = list(x.keys())
    for k in ks:
        if not isinstance(x[k][0], np.ndarray):
            non_tensor_dict[k] = x.pop(k)

    tensor_dict = fix_tensor_to_float32(merge_list_to_tensor(x))
    tensor_dict.update(non_tensor_dict)
    return tensor_dict


def pad_to_multiple_of(datapipe, divisor):
    if not hasattr(datapipe, "__len__"):
        datapipe = datapipe.set_length(len(list(datapipe)))
    to_len = int(np.ceil(len(datapipe) / divisor)) * divisor
    datapipe = datapipe.cycle(2).header(to_len)
    return datapipe


def prepare_for_dataloader(datapipe, shuffle=True):
    # if dist.is_initialized() and dist.is_available():
    #     world_size = int(os.environ["WORLD_SIZE"])
    # else:
    #     world_size = 1
    if shuffle and not global_get(Signal.train_no_shuffle, False):
        datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    if dist.is_initialized():
        datapipe = datapipe.fullsync()

    return datapipe
