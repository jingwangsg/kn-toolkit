import os, os.path as osp
from braceexpand import braceexpand
from glob import glob
from kn_util.data.wids import ShardListDataset

import tarfile

if __name__ == "__main__":
    shards = glob(osp.expanduser("~/DATASET/Open-Sora-Plan/Open-Sora-Plan-1.0.0-Mini/*.tar"))
    dataset = ShardListDataset(shards)

    import ipdb

    ipdb.set_trace()
