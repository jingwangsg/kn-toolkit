import os, os.path as osp
import webdataset as wds
import tarfile
import numpy as np
from hashlib import sha256

from ...utils.io import load_pickle, load_jsonl, save_pickle
from ...utils.multiproc import map_async_with_thread, map_async

from ...utils.system import get_strhash, is_valid_file


def get_file_keys(files, cache_dir=None):
    """
    get the number of keys in each tar file, each key corresponds to a sample in WebDataset
    """

    keys_by_file = {}

    def _get_keys(file):
        filehash = get_strhash(file)
        file_index_cache = None if cache_dir is None else osp.join(cache_dir, f"{filehash}.pkl")

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

    keys_by_file = map_async_with_thread(
        iterable=files,
        func=_get_keys,
        verbose=True,
        desc="Gathering keys from tar files",
        num_thread=32,
    )

    keys_by_file = {file: keys for file, keys in zip(files, keys_by_file)}

    return keys_by_file


def get_file_meta(files, keys_by_file, filter_keys=None):
    """
    Filtering keys by filter_ids, build mapping from index to index in shard
    The i-th element now corresponds to the "key_mapping_by_shard[shard_name][i]"-th element in shard
    """
    key_mapping_by_shard = dict()

    for file, keys in keys_by_file.items():
        if filter_keys is not None:
            cur_filter_ids = filter_keys[file]
        else:
            cur_filter_ids = set(keys)

        key_mapping_by_shard[file] = []

        # i-th element in shard -> "key_mapping_by_shard[shard_name][i]"-th element in shard
        for idx_in_shard, key in enumerate(keys):
            if key not in cur_filter_ids:
                continue

            key_mapping_by_shard[file] += [idx_in_shard]

    shards = [(file, len(key_mapping_by_shard[file])) for file in files]

    return key_mapping_by_shard, shards
