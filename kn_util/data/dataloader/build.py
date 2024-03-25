import torch
from torch.utils.data import DataLoader, DistributedSampler, default_collate

from ...utils.misc import default


def build_dataloader(
    dataset,
    batch_size=None,
    is_distributed=False,
    num_workers=0,
    pin_memory=True,
    shuffle=False,
):
    kwargs = {}

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        kwargs["shuffle"] = shuffle

    prefetch_factor = None if num_workers == 0 else 5

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        sampler=sampler,
        collate_fn=getattr(dataset, "collate_fn", default_collate),
        **kwargs,
    )
