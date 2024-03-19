from fire import Fire
from ...utils.io import save_json
from ...utils import default
import glob
import os
import os.path as osp
from webdataset import WebLoader
from tqdm import tqdm

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
    loader = WebLoader(tarfile, num_workers=num_workers)
    cnt = 0
    for _ in loader:
        cnt += 1
    return cnt


def main(input_dir, dataset, name=None, num_workers=8, output_file=None):
    tarfiles = glob.glob(f"{input_dir}/*.tar")
    input_dir = osp.abspath(input_dir)

    if output_file is None:
        output_file = osp.join(input_dir, f"wids_meta.json")
    elif not output_file.startswith("/"):
        output_file = osp.join(input_dir, output_file)
    else:
        assert output_file.startswith("/")

    meta = {
        "name": default(dataset, name),
        "base_path": input_dir,
        "wids_version": 1,
    }
    shardlist = []
    for tarfile in tqdm(tarfiles, desc="Counting samples"):
        shardlist.append(
            {
                "url": osp.basename(tarfile),
                "nsamples": count_samples(tarfile, num_workers=num_workers),
                "filesize": osp.getsize(tarfile),
                "dataset": dataset,
            }
        )

    shardlist = sorted(shardlist, key=lambda x: x["url"])
    meta["shardlist"] = shardlist
    save_json(meta, output_file)
    print(f"=> Saved to {output_file}")


Fire(main)
