from typing import Mapping, Sequence
import torch
from torch.utils.data import default_collate


def nested_apply_tensor(sample, f):
    ## add check for datasets that return none samples for missing items
    if sample == None or len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def nested_to(batch, device, dtype=None, non_blocking=False):
    def _to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device, dtype=dtype, non_blocking=non_blocking)

    return nested_apply_tensor(batch, _to_device)


def collection_get(batch, key, default=None):
    # get the value of key in batch while maintaining the structure of batch
    if isinstance(batch, list):
        return [b.get(key, default) for b in batch]
    if isinstance(batch, dict):
        return {k: v.get(key, default) for k, v in batch.items()}
