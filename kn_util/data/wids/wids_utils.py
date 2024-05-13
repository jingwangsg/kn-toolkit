import os, os.path as osp
import webdataset as wds
import tarfile
import numpy as np
from ...utils.io import load_pickle

from ...utils.multiproc import map_async_with_thread


def file_indexing(files, cache_file=None):
    """
    get the number of keys in each tar file, each key corresponds to a sample in WebDataset
    """

    keys_by_file = {}

    def _get_keys(file):
        with tarfile.open(file, "r") as tar:
            # import ipdb; ipdb.set_trace()
            keys = [_.name.split(".")[0] for _ in tar.getmembers()]

        repeated = set()
        unique_keys = []
        for key in keys:
            if key not in repeated:
                repeated.add(key)
                unique_keys.append(key)

        return unique_keys

    keys_by_file = map_async_with_thread(
        iterable=files,
        func=_get_keys,
        verbose=True,
        desc="Gathering keys from tar files",
    )

    lengths = [len(keys) for keys in keys_by_file]

    cnt = 0
    key2idx = {}
    for sublist in keys_by_file:
        for key in sublist:
            key2idx[key] = cnt
            cnt += 1

    shards = [(file, length) for file, length in zip(files, lengths)]

    return shards, key2idx
