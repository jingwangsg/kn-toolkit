import os, os.path as osp
from braceexpand import braceexpand
from glob import glob
from tqdm import tqdm
import tarfile

from kn_util.data.wids import ShardListDataset


def test_on_idx(dataset, idx):
    key = [k for k, v in dataset.key2idx.items() if v == idx]
    return dataset.get_by_key(key[0])

def test_on_idxs(dataset):
    for idx in tqdm(range(len(dataset))):
        sample = test_on_idx(dataset, idx)
        if sample["__index__"] != idx:
            import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    shards = glob(osp.expanduser("~/DATASET/Open-Sora-Plan-1.0.0-Mini/*.tar"))
    dataset = ShardListDataset(shards)

    test_on_idxs(dataset)

    import ipdb

    ipdb.set_trace()
