import torch
from torch.utils.data import DataLoader, default_collate

from ...utils.misc import default
from ..wids import DistributedChunkedSampler, ShardListDataset
from ...dist import DistributedSampler


def build_dataloader(
    dataset,
    # sampler
    is_distributed=False,
    drop_last=False,
    batch_size=None,
    dataset_size=None,
    # dataloader
    num_workers=0,
    pin_memory=True,
    shuffle=False,
    generator=None,
):
    """
    Args:
        dataset (torch.utils.data.Dataset | kn_util.data.wids.ShardListDataset): dataset to load
        batch_size (int): batch size
        total_size (int): total size of the dataset (could be smaller than len(dataset) to load a subset from head)
        is_distributed (bool): whether to use distributed training
        num_workers (int): number of workers for dataloader
        pin_memory (bool): whether to pin memory
        shuffle (bool): whether to shuffle
        generator (torch.Generator): random number generator

    """

    kwargs = {}

    # setup sampler for distributed training
    sampler = None
    if is_distributed:
        if isinstance(dataset, ShardListDataset):
            sampler = DistributedChunkedSampler(
                dataset,
                shuffle=shuffle,
                shufflefirst=shuffle,
                dataset_size=dataset_size,
                drop_last=drop_last,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                dataset_size=dataset_size,
                drop_last=drop_last,
            )
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
        generator=generator,
        **kwargs,
    )
