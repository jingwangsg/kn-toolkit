from typing import Mapping, Sequence
import torch 
def collection_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, str):
        return batch
    elif isinstance(batch, Mapping):
        return {k: collection_to_device(v, device) for k,v in batch.items()}
    elif isinstance(batch, Sequence):
        return [collection_to_device(x, device) for x in batch]
    else:
        return batch