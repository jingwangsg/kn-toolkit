import argparse
import os.path as osp
import subprocess
import torch
import numpy as np
import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpus", type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    print("launch through python")
    gpus = []
    if args.gpus == "-1":
        num_devices = torch.cuda.device_count()
        gpus = [_ for _ in range(num_devices)]
    else:
        gpus = eval(f"[{args.gpus}]")

    fake_datasets = ["coco2017", "coco_seg2017", "Charades-STA", "ActivityNetCaption"]
    fake_models = ["GroundingFormer", "PSGFormer", "GDBFormer", "QueryMoment", "GTR"]

    for id in gpus:
        dataset = np.random.choice(fake_datasets, size=1)
        model = np.random.choice(fake_models, size=1)
        cmd = f"$HOME/server_utils/python main.py --model {model[0]} --dataset {dataset[0]} --use_bbox_refine --num_epoch 100 --host 127.0.0.1 --ddp --local-rank {id}"
        subprocess.Popen(cmd, shell=True)
        # time.sleep(1)


if __name__ == "__main__":
    main()