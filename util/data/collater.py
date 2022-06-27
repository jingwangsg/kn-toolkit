import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def collate_map(data_dict_list, mapper, keys):
    """
    data_dict_list: list of data dict
    mapper: transform that is applied on feature list
    keys: key that is applied

    * generally, mapper is to transform numpy arrays into batch tensors
    """
    ret_dict = {}

    for k in keys:
        feat_list = [x[k] for x in data_dict_list]
        ret_dict[k] = mapper(feat_list)

    return ret_dict


def pad_collate_mapper(arr_list, max_len=None, prefix_pad=False, value=0):
    tensor_list = []

    if max_len is None:
        max_len = 0
        for arr in arr_list:
            if len(arr) > max_len:
                max_len = len(arr)
    for arr in arr_list:
        cur_tensor = torch.tensor(arr)
        pad_len = 0 if max_len - len(arr) < 0 else max_len - len(arr)
        pad_args = (0, pad_len) if not prefix_pad else (pad_len, 0)
        cur_tensor = F.pad(cur_tensor, pad_args, mode="constant", value=0)

        tensor_list += [cur_tensor[None, :]]

    return torch.cat(tensor_list)


def cat_collate_mapper(arr_list):
    return torch.cat([torch.tensor(arr)[None, :] for arr in arr_list])
