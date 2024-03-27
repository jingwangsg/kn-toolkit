import torch
from torch.utils.data import default_collate
from functools import partial

from .collection_ops import collection_extend_multikeys


def _default_collate_wrapped(batch, default_keys=[], list_keys=[]):
    ret = {}
    batch = [{k: v for k, v in item.items() if k in default_keys or k in list_keys} for item in batch]
    for key in default_keys:
        ret[key] = default_collate([item[key] for item in batch])
    for key in list_keys:
        ret[key] = [item[key] for item in batch]
    return ret


def default_collate_builder(default_keys=[], list_keys=[]):
    return partial(_default_collate_wrapped, default_keys=default_keys, list_keys=list_keys)


def _collate_fn_wrapped(batch, default_collate_by_keys, flatten_keys):
    flatten_batch = collection_extend_multikeys(batch, keys=flatten_keys)
    # flatten_batch = {k: default_collate(v) for k, v in flatten_batch.items()}
    for k, v in flatten_batch.items():
        flatten_batch[k] = default_collate(v)

    ret = {}
    ret.update(flatten_batch)
    ret.update(default_collate_by_keys(batch))
    return ret


def collate_fn_builder(default_keys=[], list_keys=[], flatten_keys=[]):
    """Collate function builder that supports default_collate, list_collate and flatten_collate.
    NOTE: make sure all elements can be recognized by default_collate and correctly concatenated.
    For example, List[List[T]] cannot be concatenated by default_collate into a single tensor. It will be List[Tensor[T]] instead.

    Args:
        default_keys (List[str]): keys to apply default_collate
        list_keys (List[str]): keys to apply list_collate
        flatten_keys (List[str]): keys to apply flatten_collate
            Here flatten_collate will flatten List[List[T]] to List[T] for keys in flatten_keys.
            This is especially useful for handling variable length annotations like timestamps, sentences, etc.

    Return:
        collate_fn (Callable): collate function that applies default_collate, list_collate and flatten_collate

    """
    default_collate_by_keys = default_collate_builder(default_keys, list_keys)
    return partial(_collate_fn_wrapped, default_collate_by_keys=default_collate_by_keys, flatten_keys=flatten_keys)
