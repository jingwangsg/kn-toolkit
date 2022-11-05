import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import numpy as np


def general_pad(
    arr_list: List[np.ndarray], fill_value=None, axis=None, to_length=None, return_mask=True
):
    assert axis is not None
    assert fill_value is not None
    to_length = 0
    if to_length is not None:
        to_length = to_length
    else:
        for arr in arr_list:
            to_length = np.maximum(to_length, arr.shape[axis])

    to_shape = list(arr_list[0].shape)
    to_shape[axis] = to_length

    ret_arr = []
    ret_mask = []

    shape_dim = len(arr_list[0].shape)

    for arr in arr_list:
        full_arr = np.full(to_shape, fill_value=fill_value, dtype=arr[0].dtype)
        sub_slices = tuple([slice(0, arr.shape[_]) for _ in range(shape_dim)])
        full_arr[sub_slices] = arr
        ret_arr += [full_arr]
        if return_mask:
            full_arr = np.zeros(to_shape, dtype=bool)
            full_arr[sub_slices] = True
            flatten_slices = tuple(
                [slice(0, arr.shape[_]) if _ == axis else 0 for _ in range(shape_dim)]
            )
            ret_mask += [full_arr[flatten_slices]]

    if return_mask:
        return ret_arr, ret_mask
    else:
        return ret_mask


def fix_tensor_to_float32(feature_dict):
    for k, v in feature_dict.items():
        if v.dtype == torch.float64:
            feature_dict[k] = v.float()
    return feature_dict


def stack_list_to_tensor(feature_dict, include_keys=None, exclude_keys=None):
    if include_keys is None:
        include_keys = list(feature_dict.keys())
    if exclude_keys is None:
        exclude_keys = []

    for k in include_keys:
        if k in exclude_keys:
            continue
        feature_dict[k] = torch.from_numpy(np.stack(feature_dict[k]))

    return feature_dict


def collect_features_from_sample_list(sample_list, keys=None):
    assert keys is not None
    has_single_key = isinstance(keys, str)
    if has_single_key:
        keys = [keys]

    ret_list = []
    for k in keys:
        ret_list += [[s[k] for s in sample_list]]

    if has_single_key:
        return ret_list[0]
    else:
        return ret_list
