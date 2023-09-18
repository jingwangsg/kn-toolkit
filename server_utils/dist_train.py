import argparse
import os.path as osp
import subprocess
import numpy as np
import time
import random


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpus", type=str)
    parser.add_argument("-m", "--mode", choices=["occupy", "attack", "peace"], default="peace")
    parser.add_argument("-t", "--time", type=int, default=-1)
    parser.add_argument("--mem", type=int, default=-1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("-s", "--sleep", action="store_true", default=False)
    #  mode = 0: fight and occupy
    #  mode = 1: fight and exit
    #  mode = 2: peace

    return parser.parse_args()


def get_vailable_memory():
    cmd = "nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader"
    outs = subprocess.run(cmd, shell=True, text=True, capture_output=True).stdout.strip().split("\n")
    memories = [outs[i].split(",")[-2:] for i in range(len(outs))]

    def _parse_value(mem):
        return int(mem.strip().split(" ")[0])

    memory_values = [(_parse_value(_[0]), _parse_value(_[1])) for _ in memories]
    return memory_values


def main():
    args = parse_args()
    print(f"Running in mode[{args.mode}] | delay = {args.delay:.1f} (h)")
    time.sleep((args.delay * 3600))
    gpus = []
    if args.gpus == "-1":
        num_devices = len(get_vailable_memory())
        gpus = [_ for _ in range(num_devices)]
    else:
        gpus = eval(f"[{args.gpus}]")

    fake_datasets = ["coco2017", "coco_seg2017", "Charades-STA", "ActivityNetCaption"]
    fake_models = ["GroundingFormer", "PSGFormer", "GDBFormer", "QueryMoment", "GTR"]

    mode_dict = dict(occupy=0, attack=1, peace=2)

    def launch(id):
        if args.mode != "attack":
            rand_left = random.uniform(2.5, 3.5)
        else:
            rand_left = 1.0
        dataset = np.random.choice(fake_datasets, size=1)
        model = np.random.choice(fake_models, size=1)
        # CUDA_VERSION = subprocess.check_output("cat $HOME/WORKSPACE/kn-toolkit/dotfiles/variable.sh | grep CUDA_VERSION", shell=True).decode("utf-8").strip().split("=")[1]
        CUDA_VERSION = "10"
        cmd = f"$HOME/miniconda3/envs/cuda{CUDA_VERSION}/bin/python main.py --model {model[0]} --dataset {dataset[0]} --use_bbox_refine --num_epoch 100 --host 127.0.0.1 --ddp --local-rank {id}"
        cmd = cmd + f" {mode_dict[args.mode]} {args.time} {args.mem} {rand_left}"
        subprocess.Popen(cmd, shell=True)

    launched = [False for _ in range(len(gpus))]
    if args.mode == "peace":
        memories = get_vailable_memory()
        while (not all(launched)):
            for id in gpus:
                if not launched[id] and memories[id][1] / memories[id][0] > 0.5:
                    launch(id)
                    launched[id] = True
            from time import sleep
            sleep(1)
    else:
        for id in gpus:
            launch(id)

    if args.sleep:
        while True:
            pass


if __name__ == "__main__":
    main()
