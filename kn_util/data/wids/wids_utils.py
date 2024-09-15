import os
import os.path as osp
import tarfile

import numpy as np

from ...utils.io import load_pickle, save_pickle
from ...utils.multiproc import map_async_with_thread
from ...dist import get_rank, get_world_size, all_gather_object
from ...utils.system import get_strhash, is_valid_file


def get_tarfile_keys(files, cache_dir=None):

    keys_by_file = {}

    def _get_keys(file):
        filehash = get_strhash(file)
        file_index_cache = (
            None if cache_dir is None else osp.join(cache_dir, f"{filehash}.pkl")
        )

        if is_valid_file(file_index_cache):
            return load_pickle(file_index_cache)

        with tarfile.open(file, "r") as tar:
            keys = [_.name.split(".")[0] for _ in tar.getmembers()]

        repeated = set()
        unique_keys = []
        for key in keys:
            if key not in repeated:
                repeated.add(key)
                unique_keys.append(key)

        if file_index_cache is not None:
            os.makedirs(osp.dirname(file_index_cache), exist_ok=True)
            save_pickle(unique_keys, file_index_cache)

        return unique_keys

    num_parititons = min(len(files), get_world_size())
    partition_idx = get_rank()

    files_at_rank = (
        np.array_split(files, num_parititons)[partition_idx]
        if partition_idx < num_parititons
        else []
    )

    keys_by_file_at_rank = map_async_with_thread(
        iterable=files_at_rank,
        func=_get_keys,
        verbose=True,
        desc="Gathering keys from tar files",
        num_thread=64,
    )
    keys_by_file = all_gather_object(keys_by_file_at_rank)
    keys_by_file = [keys for sublist in keys_by_file for keys in sublist]

    keys_by_file = {file: keys for file, keys in zip(files, keys_by_file)}

    return keys_by_file


def get_shard_meta(shards, keys_by_shard):
    """
    Filtering keys by filter_ids, build mapping from index to index in shard
    The i-th element now corresponds to the "key_mapping_by_shard[shard_name][i]"-th element in shard

    Args:
        shards: a list of tar file names
        keys_by_shard: a dict with key as the file name and value as a list of keys
        include_keys: similar to key_mapping_by_shard, but it may contain keys that are not in the shard,
            returned key_mapping_by_shard should be (include_keys & keys_in_shard)

    Return:
        inneridx_by_shard: a dict with key as the file name and value as a list of inner indices
        shards: a list of tuples, each tuple contains the file name and the number of keys in the file

    """
    assert len(shards) == len(
        keys_by_shard
    ), "files and keys_by_file should have the same length"

    inneridx_by_shard = dict()

    for file, keys in keys_by_shard.items():
        inneridx_by_shard[file] = []

        # i-th element in shard -> "key_mapping_by_shard[shard_name][i]"-th element in shard
        for idx_in_shard, key in enumerate(keys):
            if include_keys is not None and key not in include_keys[file]:
                continue

            inneridx_by_shard[file] += [idx_in_shard]

    shards = [(shard, len(inneridx_by_shard[file])) for shard in shards]

    return inneridx_by_shard, shards
