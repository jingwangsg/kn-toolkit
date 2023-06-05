import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import numpy as np
from .tensor_ops import general_pad_arr
from beartype import beartype

ListOfArrayLike = List[Union[np.ndarray, torch.Tensor]]
Scaler = Union[int, float]

@beartype
def general_pad(
    arr_list: ListOfArrayLike,
    fill_value: Scaler,
    axis: int,
    to_length: Optional[int] = None,
    to_multiple: Optional[int] = None,
    return_mask: Optional[bool] = False
):
    """ Pad a list of array-like objects to the same length along a given axis.
    Args:
        arr_list: A list of array-like objects.
        fill_value: The value to fill the padded elements with.
        axis: The axis to pad along.
        to_length: The length to pad to. If None, the maximum length of the arrays in arr_list is used.
        to_multiple: If not None, the length is padded to the smallest multiple of to_multiple that is greater than or equal to the length.
        return_mask: If True, a mask is returned indicating which elements were padded.
    Returns:
        A tuple of (padded_arr_list, mask) if return_mask is True, otherwise just padded_arr_list.
        Here padded_arr_list is a list of the padded array-like objects.
    """
    assert axis is not None
    assert fill_value is not None

    # backend = None

    # if isinstance(arr_list[0], torch.Tensor):
    #     backend = "pt"
    # elif isinstance(arr_list[0], np.ndarray):
    #     backend = "np"
    # else:
    #     raise ValueError("arr_list must be a list of torch.Tensor or np.ndarray")


    if not isinstance(arr_list, list):
        arr_list = [arr_list]
    
    if to_length is None:
        to_length = 0
        for arr in arr_list:
            to_length = np.maximum(to_length, arr.shape[axis])

    if to_multiple:
        to_length = int(np.ceil(to_length / to_multiple)) * to_multiple


    ret_arr = []
    ret_mask = []

    shape_dim = len(arr_list[0].shape)
    for arr in arr_list:
        cur_arr, cur_mask = general_pad_arr(arr, axis, to_length, fill_value, return_mask=True)
        ret_arr.append(cur_arr)
        if return_mask:
            ret_mask.append(cur_mask)
    
    # if backend == "np":
    #     ret_arr = np.stack(ret_arr, axis=0)
    # elif backend == "pt":
    #     ret_arr = torch.stack(ret_arr, dim=0)
    
    return ret_arr, ret_mask if return_mask else ret_arr

def fix_tensor_to_float32(feature_dict):
    for k, v in feature_dict.items():
        if v.dtype == torch.float64:
            feature_dict[k] = v.float()
    return feature_dict


def merge_list_to_tensor(feature_dict, include_keys=None, exclude_keys=None, mode="stack"):
    if include_keys is None:
        include_keys = list(feature_dict.keys())
    if exclude_keys is None:
        exclude_keys = []

    for k in include_keys:
        if k in exclude_keys:
            continue
        if mode == "stack":
            feature_dict[k] = torch.from_numpy(np.stack(feature_dict[k]))
        else:
            feature_dict[k] = torch.from_numpy(np.concatenate(feature_dict[k], axis=0))

    return feature_dict


def collect_features_from_sample_list(sample_list, keys=None):
    if keys is None:
        keys = list(sample_list[0].keys())
    has_single_key = isinstance(keys, str)
    if has_single_key:
        keys = [keys]

    ret_list = []
    for k in keys:
        ret_list += [[s[k] for s in sample_list]]

    if has_single_key:
        return {keys: ret_list[0]}
    else:
        return dict(zip(keys, ret_list))
