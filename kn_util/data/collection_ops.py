from typing import Mapping, Sequence
import torch 
def collection_to_device(batch, device):
    def _to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
    
    collection_apply(batch, _to_device)

def collection_apply(batch, fn=lambda x: x):
    if isinstance(batch, list):
        for i in range(len(batch)):
            batch[i] = collection_apply(batch[i], fn)
    if isinstance(batch, dict):
        for k in batch.keys():
            batch[k] = collection_apply(batch[k], fn)
    try:
        return fn(batch)
    except:
        return batch