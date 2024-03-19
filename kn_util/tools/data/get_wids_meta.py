from fire import Fire
from ...utils.io import save_json
from ...utils import default
import glob
import os
import os.path as osp
from webdataset import WebLoader

"""
Example of json file
{
  "name": "ExampleDatasetName",
  "base_path": "/optional/base/path/or/url",
  "shardlist": [
    {
      "url": "path/to/shard1",
      "nsamples": 1000,
      "filesize": 123456,
      "dataset": "optional dataset identifier or description for shard 1"
    },
    {
      "url": "path/to/shard2",
      "nsamples": 2000,
      "filesize": 234567,
      "dataset": "optional dataset identifier or description for shard 2"
    },
    {
      "url": "path/to/shard3",
      "nsamples": 1500,
      "filesize": 345678,
      "dataset": "optional dataset identifier or description for shard 3"
    }
  ]
}
"""


def count_samples(tarfile, num_workers=0):
    with WebLoader(tarfile, num_workers=num_workers) as stream:
        cnt = 0
        for _ in stream:
            cnt += 1
    return cnt


def main(input_dir, output_file, dataset, name=None, num_workers=8):
    tarfiles = glob.glob(f"{input_dir}/*.tar")
    meta = {
        "name": default(dataset, name),
        "base_path": osp.abspath(input_dir),
    }
    shardlist = []
    for tarfile in tarfiles:
        shardlist.append(
            {
                "url": tarfile,
                "nsamples": count_samples(tarfile, num_workers=num_workers),
                "filesize": osp.getsize(tarfile),
                "dataset": dataset,
            }
        )

    meta["shardlist"] = shardlist
    save_json(output_file, meta)
