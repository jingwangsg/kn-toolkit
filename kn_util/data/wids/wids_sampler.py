import random
import numpy as np
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
from typing import Optional
import warnings

from ...dist import get_world_size


def lengths_to_ranges(lengths, start_offset=0):
    """Convert a list of lengths to a list of ranges."""
    ranges = []
    start = start_offset
    for length in lengths:
        ranges.append((start, start + length))
        start += length
    return ranges


def intersect_range(a, b):
    """Return the intersection of the two half-open integer intervals."""
    result = max(a[0], b[0]), min(a[1], b[1])
    if result[0] >= result[1]:
        return None
    return result


def intersect_ranges(rangelist, r):
    """Return the intersection of the half-open integer interval r with the list of half-open integer intervals."""
    result = []
    for a in rangelist:
        x = intersect_range(a, r)
        if x is not None:
            result.append(x)
    return result


def iterate_lengths(lengths, rng, start_offset=0, indexshuffle=True, shardshuffle=True, total_size=None):
    """Iterate over the ranges in a random order."""
    ranges = lengths_to_ranges(lengths, start_offset=start_offset)

    if shardshuffle:
        ranges = rng.sample(ranges, len(ranges))

    shard_indexes = list(range(len(ranges)))
    if shardshuffle:
        rng.shuffle(shard_indexes)
    for i in shard_indexes:
        lo, hi = ranges[i]
        sample_indexes = list(range(lo, hi))
        if total_size is not None:
            # to support drop_last=True
            sample_indexes = [_ if _ < total_size else (_ % total_size) for _ in sample_indexes]
        if indexshuffle:
            rng.shuffle(sample_indexes)
        yield from sample_indexes


class ShardListSampler(Sampler):
    """A sampler that samples consistent with a ShardListDataset.

    This sampler is used to sample from a ShardListDataset in a way that
    preserves locality.

    This returns a permutation of the indexes by shard, then a permutation of
    indexes within each shard. This ensures that the data is accessed in a
    way that preserves locality.

    Note that how this ends up splitting data between multiple workers ends up
    on the details of the DataLoader. Generally, it will likely load samples from the
    same shard in each worker.

    Other more sophisticated shard-aware samplers are possible and will likely
    be added.
    """

    def __init__(self, dataset, *, lengths=None, seed=0, shufflefirst=False):
        if lengths is None:
            lengths = list(dataset.lengths)
        # self.ranges = lengths_to_ranges(lengths)
        self.lengths = lengths
        self.seed = seed
        self.shufflefirst = shufflefirst
        self.epoch = 0

    def __iter__(self):
        self.rng = random.Random(self.seed + 1289738273 * self.epoch)
        shardshuffle = self.shufflefirst or self.epoch > 0
        yield from iterate_lengths(self.lengths, self.rng, shardshuffle=shardshuffle)
        self.epoch += 1


ShardedSampler = ShardListSampler


class ChunkedSampler(Sampler):
    """A sampler that samples in chunks and then shuffles the samples within each chunk.

    This preserves locality of reference while still shuffling the data.
    """

    def __init__(
        self,
        dataset,
        *,
        num_samples=None,
        chunksize=2000,
        seed=0,
        shuffle=False,
        shufflefirst=False,
    ):

        if isinstance(num_samples, int):
            lo, hi = 0, num_samples
        elif num_samples is None:
            lo, hi = 0, len(dataset)
        else:
            lo, hi = num_samples
        self.span = (lo, hi)
        # self.ranges = [(i, min(i + chunksize, hi)) for i in range(lo, hi, chunksize)]
        self.lengths = [min(chunksize, hi - i) for i in range(lo, hi, chunksize)]
        self._len = hi - lo
        self.dataset_size = len(dataset)
        self.seed = seed
        self.shuffle = shuffle
        self.shufflefirst = shufflefirst
        self.epoch = 0
        self.gen_pnt = -1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def load_state(self, state_dict):
        def _safe_overwrite(variable_name, ignore_diff=False):
            if variable_name in state_dict:
                if not ignore_diff:
                    assert getattr(self, variable_name) == state_dict[variable_name]
                setattr(self, variable_name, state_dict[variable_name])

        _safe_overwrite("epoch", ignore_diff=True)
        _safe_overwrite("gen_pnt", ignore_diff=True)
        _safe_overwrite("seed", ignore_diff=True)
        for variable_name in state_dict.keys():
            _safe_overwrite(variable_name, ignore_diff=False)

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "gen_pnt": self.gen_pnt,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "shufflefirst": self.shufflefirst,
            "span": self.span,
            "lengths": self.lengths,
        }

    def __iter__(self):
        self.rng = random.Random(self.seed + 1289738273 * self.epoch)
        shardshuffle = self.shufflefirst or self.epoch > 0
        indices = list(
            iterate_lengths(
                self.lengths,
                self.rng,
                indexshuffle=self.shuffle,
                shardshuffle=(self.shuffle and shardshuffle),
                total_size=self.dataset_size,
                start_offset=self.span[0],
            )
        )

        for i in range(self.gen_pnt + 1, len(indices)):
            self.gen_pnt = i
            yield indices[i]

        self.epoch += 1
        self.gen_pnt = -1

    def __len__(self):
        return self._len


class ChunkedSamplerV2(Sampler):
    def __init__(self, dataset, *, shuffle=False, seed=0, num_shard_in_chunk="max"):
        """
        shuffling is slightly different from ChunkedSampler, here each chunk is a tar file,
        instead of a fixed number of samples.
        """
        self.dataset = dataset
        self.seed = seed
        self.shuffle = shuffle
        self.dataset_size = len(dataset)
        self.epoch = 0

        capacity = dataset.cache.lru.capacity if num_shard_in_chunk == "max" else num_shard_in_chunk
        self.lengths = [np.sum(dataset.lengths[i : i + capacity]) for i in range(0, len(dataset.lengths), capacity)]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        self.rng = random.Random(self.seed + 1289738273 * self.epoch)
        yield from iterate_lengths(
            self.lengths,
            self.rng,
            indexshuffle=self.shuffle,
            shardshuffle=self.shuffle,
            total_size=self.dataset_size,
        )
        self.epoch += 1

    def __len__(self):
        return len(self.dataset)


def estimate_chunksize(dataset):
    shard_capacity = dataset.cache.lru.capacity
    min_shard_size = np.mean(dataset.lengths)
    estimated_chunksize = int(shard_capacity * min_shard_size)

    return estimated_chunksize


def DistributedChunkedSampler(
    dataset: Dataset,
    *,
    num_replicas: Optional[int] = None,
    dataset_size: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    # shufflefirst: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    chunksize: int = "max",
) -> ChunkedSampler:
    """
    Return a ChunkedSampler for the current worker in distributed training.

    Reverts to a simple ChunkedSampler if not running in distributed mode.

    Since the split among workers takes place before the chunk shuffle,
    workers end up with a fixed set of shards they need to download. The
    more workers, the fewer shards are used by each worker.

    Args:
        dataset: The dataset to sample from
        num_replicas: The number of workers
        total_size: The number of samples to use (len(dataset) by default, maybe smaller if given)
        rank: The rank of the current worker
        shuffle: Whether to shuffle the samples within each chunk
        seed: The seed for the random number generator
        drop_last: Whether to drop the last incomplete chunk
        chunksize: The size of each chunk

    """

    estimated_chunksize = estimate_chunksize(dataset)
    chunksize = estimated_chunksize if chunksize == "max" else chunksize

    if num_replicas is None:
        num_replicas = get_world_size()

    if not dist.is_initialized():
        warnings.warn("DistributedChunkedSampler is called without distributed initialized; assuming single process")
        num_replicas = 1
        rank = 0
    else:
        num_replicas = num_replicas or dist.get_world_size()
        rank = rank or dist.get_rank()
    assert rank >= 0 and rank < num_replicas

    dataset_size = dataset_size or len(dataset)

    _offset = 0 if drop_last else (num_replicas - 1)
    worker_chunk = (dataset_size + _offset) // num_replicas

    worker_start = rank * worker_chunk
    worker_end = worker_start + worker_chunk
    sampler = ChunkedSampler(
        dataset,
        num_samples=(worker_start, worker_end),
        chunksize=chunksize,
        seed=seed,
        shuffle=shuffle,
        shufflefirst=shuffle,
    )

    return sampler


import torch, math
from torch.utils.data.distributed import DistributedSampler


class DistributedLocalSampler(DistributedSampler):
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        chunk_size = self.total_size // self.num_replicas
        begin_idx = chunk_size * self.rank
        stop_idx = chunk_size * (self.rank + 1)
        indices = indices[begin_idx:stop_idx]

        # logger.info("[SamplerIndices: ]", indices)
        assert len(indices) == self.num_samples
        return iter(indices)
