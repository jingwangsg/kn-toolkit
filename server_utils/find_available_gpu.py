import subprocess
import argparse
import pandas as pd
from functools import partial
from pathos.multiprocessing import Pool
import time
from tqdm import tqdm
import ipdb


def map_async(iterable, func, num_process=30, desc: object = ""):
    p = Pool(num_process)
    # ret = []
    # for it in tqdm(iterable, desc=desc):
    #     ret.append(p.apply_async(func, args=(it,)))
    ret = p.map_async(func=func, iterable=iterable)
    total = ret._number_left
    pbar = tqdm(total=total, desc=desc)
    while ret._number_left > 0:
        pbar.n = total - ret._number_left
        pbar.refresh()
        time.sleep(0.1)
    p.close()

    return ret.get()


def read_args():
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--mem_thre", default="0", type=str)
    args.add_argument("-n", "--n_gpu", default=30, type=int)
    return args.parse_args()


def query_node(node_idx, partition_name, info, mem_thre, delay):
    ret = []

    # print(f"node_idx:{node_idx} | partition: {partition_name}")
    cmd = f"""
    module load cuda90/toolkit/9.0.176 && \
    timeout {delay} srun -p {partition_name} -w node{node_idx} \
    nvidia-smi --query-gpu=memory.free,memory.total\
    --format=csv,nounits,noheader
    """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout != "":
        lines = result.stdout.split("\n")
        for idx, line in enumerate(lines):
            if line == "":
                continue
            try:
                mem_rest, mem_total = line.split(",")
                mem_rest, mem_total = int(mem_rest.strip()), int(mem_total.strip())
                if mem_rest > mem_thre:
                    ret.append(
                        (f"node_{node_idx}_gpu_{idx}", [mem_rest, mem_total, info])
                    )
            except:
                pass
    if ret == []:
        print(f"node{node_idx} not available!")

    return ret


def query_node_wrap(args, mem_thre, delay):
    return query_node(*args, mem_thre=mem_thre, delay=delay)


def find_gpu(mem_thre, delay=5):

    gpu_dict = {}

    with open("/export/home/kningtg/server_utils/server_list.csv", "r") as f:
        server_infos = f.readlines()

    func_args_list = []

    for server_info in server_infos:
        node_idx, partition_name, info = server_info.split(",")
        info = info.strip()
        if partition_name == "NA":
            continue
        func_args_list += [(node_idx, partition_name, info)]

    query_node_func = partial(query_node_wrap, mem_thre=mem_thre, delay=delay)
    # ipdb.set_trace()

    rets = map_async(func_args_list, query_node_func, num_process=len(func_args_list))
    # rets = [query_node_func(x) for x in func_args_list]
    flat_ret = []
    for ret in rets:
        flat_ret += ret
    gpu_dict = dict(flat_ret)

    df = pd.DataFrame.from_dict(
        gpu_dict, orient="index", columns=["mem_rest", "mem_total", "info"]
    )
    return df


if __name__ == "__main__":
    args = read_args()
    mem_thre = eval(args.mem_thre)
    df = find_gpu(mem_thre)
    # print(gpu_dict)
    df.sort_values(by=["mem_rest"], ascending=False, inplace=True)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        if args.n_gpu == -1:
            print(df)
        else:
            print(df.iloc[: args.n_gpu, :])
