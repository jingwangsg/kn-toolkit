import os, os.path as osp
from ...utils.multiproc import map_async_with_thread
import webdataset as wds
import tarfile


def get_file_lengths(files):
    """
    get the number of keys in each tar file, each key corresponds to a sample in WebDataset
    """
    def _get_key_number(file):
        with tarfile.open(file, 'r') as tar:
            members = set([_.name.split(".")[0] for _ in tar.getmembers()])
        
        return file, len(members)

    return map_async_with_thread(iterable=files, func=_get_key_number, num_thread=None, verbose=True)
