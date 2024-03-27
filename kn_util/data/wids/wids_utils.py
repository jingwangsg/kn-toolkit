import os, os.path as osp
from ...utils.multiproc import map_async
import webdataset as wds


def get_file_lengths(files):
    def _get_file_length(file):
        cnt = 0
        loader = wds.WebLoader(file, num_workers=4)
        for _ in loader:
            cnt += 1

        return file, cnt

    return map_async(iterable=files, func=_get_file_length, num_process=16, verbose=False)
