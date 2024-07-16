from typing import Mapping, Sequence, List

try:
    import torch
    from torch.utils.data import default_collate
except:
    pass


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


def groupby(data, key=None, agg="unique"):
    assert isinstance(data, List)
    ret_dict = {}
    for d in data:
        k = d.pop(key)
        if agg == "unique":
            ret_dict[k] = d
        elif agg == "append":
            if k not in ret_dict:
                ret_dict[k] = []
            ret_dict[k].append(d)
        else:
            raise NotImplementedError

    return ret_dict


def collection_get(batch, key, default=None):
    """
    Return:
        list: [b[key] for b in batch]
    """
    # get the value of key in batch while maintaining the structure of batch
    if isinstance(batch, list):
        return [b.get(key, default) for b in batch]
    else:
        raise NotImplementedError


def collection_get_multikeys(batch, keys, default=None):
    """
    Return:
        dict: {k: [v[k] for v in batch]}
    """
    if isinstance(batch, list):
        ret = {}
        for k in keys:
            seq = collection_get(batch, k, default=default)
            ret[k] = seq
        return ret
    else:
        raise NotImplementedError


def collection_extend(batch, key, return_batch_indices=False):
    """an extend version of collection_get
    Return:
        list: join(b[key] for b in batch)
    """
    if isinstance(batch, list):
        ret = []
        batch_idxs = []
        for batch_idx, b in enumerate(batch):
            ret.extend(b[key])
            batch_idxs.extend([batch_idx] * len(b[key]))
        if return_batch_indices:
            return ret, batch_idxs
        else:
            return ret
    else:
        raise NotImplementedError


def collection_extend_multikeys(batch, keys):
    """collection_extend that support multiple keys at once
    Return:
        dict: {k: join(v[k] for v in batch)}
    """

    if isinstance(batch, list):
        if len(keys) == 0:
            return {}

        ret = {}
        batch_idxs = None
        for idx, key in enumerate(keys):
            if idx == 0:
                seq, batch_idxs = collection_extend(batch, key, return_batch_indices=True)
            else:
                seq = collection_extend(batch, key, return_batch_indices=False)
            ret[key] = seq
        ret["batch_idxs"] = batch_idxs
        return ret
    else:
        raise NotImplementedError
