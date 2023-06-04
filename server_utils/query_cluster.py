import subprocess
import argparse
import pandas as pd
from functools import partial
from pathos.multiprocessing import Pool
from tqdm.contrib.concurrent import thread_map
import time
from tqdm import tqdm
import io
import os
import os.path as osp

from collections import OrderedDict


def map_async(iterable, func, num_process=30, desc: object = "", test_flag=False):
    """while test_flag=True, run sequentially"""
    if test_flag:
        ret = [func(x) for x in tqdm(iterable)]
        return ret
    else:
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


class GPUCluster:
    def __init__(self, server_info_fn, timeout):
        server_info = pd.read_csv(server_info_fn, names=["node_idx", "partition", "gpu_type"])
        server_info["partition"] = server_info["partition"].astype(str)
        server_info["gpu_type"] = server_info["gpu_type"].astype(str)
        self.server_info = server_info
        self.timeout = timeout

    def _check_node(self, node_idx):
        cmd = f"sinfo --nodes=node{node_idx:02d} -N --noheader"
        result = subprocess.run(cmd, text=True, capture_output=True, shell=True)
        return result

    def _query_single_node(self, inputs):
        node_idx, partition, cmd, timeout = inputs

        result = self._check_node(node_idx)
        if (
            "alloc" not in result.stdout
            and "idle" not in result.stdout
            and "mix" not in result.stdout
        ):
            if result.stdout:
                print(f"node{node_idx:02d} {result.stdout.split()[-1]}")
            else:
                print(f"node{node_idx:02d} N/A")
            return node_idx, None

        prefix = f"timeout {timeout} srun -N 1 -n 1 -c 1 -p {partition} -w node{node_idx:02d} --export ALL "

        cmd_with_timeout = prefix + cmd

        result = subprocess.run(cmd_with_timeout, shell=True, capture_output=True, text=True)
        # print(cmd_with_timeout)

        if not result.stdout and ("revoke" in result.stderr or "error" in result.stderr):
            print(f"[FAIL] node{node_idx:02d} | {result.stderr}")
            # pass

        return node_idx, result.stdout

    def query_all_node(self, cmd):
        inputs_list = [
            (row["node_idx"], row["partition"], cmd, self.timeout)
            for _, row in self.server_info.iterrows()
        ]

        st = time.time()
        node_stdout = map_async(iterable=inputs_list, func=self._query_single_node)
        print(f"query costs {time.time()-st}(s)")
        return node_stdout

    def find_gpu_available(self):
        gpu_query_cmd = (
            "nvidia-smi --query-gpu=gpu_name,memory.free,memory.total --format=csv,nounits"
        )
        node_stdouts = self.query_all_node(gpu_query_cmd)

        df_list = []

        for node_idx, node_out in node_stdouts:
            if not node_out:
                continue

            cur_df = pd.read_csv(io.StringIO(node_out), dtype=str).reset_index()
            cur_df.rename(columns={"index": "gpu.id"}, inplace=True, errors="raise")
            cur_df["gpu.id"] = cur_df["gpu.id"].astype(str)
            cur_df["gpu.id"] = f"node{node_idx:02d}_gpu#" + cur_df["gpu.id"]

            cur_df[" memory.free [MiB]"] = cur_df[" memory.free [MiB]"].astype(int)
            cur_df[" memory.total [MiB]"] = cur_df[" memory.total [MiB]"].astype(int)

            df_list += [cur_df]

        df = pd.concat(df_list, ignore_index=True)
        df.sort_values(by=[" memory.free [MiB]"], ascending=False, inplace=True)

        return df

    def find_gpu_usage(self, username="kningtg", cmd_include=""):
        gpu_query_cmd = (
            f"py3smi -f --left -w $(($(tput cols)-30)) | grep {username}"
            if not cmd_include
            else f"py3smi -f --left -w $(($(tput cols)-30)) | grep {username} | grep {cmd_include}"
        )
        
        node_stdouts = self.query_all_node(gpu_query_cmd)

        item_list = []

        for node_idx, node_out in node_stdouts:
            if not node_out:
                continue

            for line in node_out.split("\n"):
                if not line:
                    continue

                line = line.replace(" days", "-days")

                _id, usr, pid, time, _ = line.strip("|").strip().split(maxsplit=4)
                cmd, size = _.rsplit(maxsplit=1)

                if cmd_include not in cmd:
                    continue
                # import ipdb; ipdb.set_trace() #FIXME ipdb

                item = {
                    "partition": self.server_info[self.server_info["node_idx"] == node_idx][
                        "partition"
                    ].item(),
                    "gpu.id": f"node{node_idx:02d}_#" + _id,
                    "gpu.occupied": size,
                    "PID": pid,
                    "user": usr,
                    "time": time,
                    "cmd": cmd
                }

                item_list += [item]

        df = pd.DataFrame(item_list)

        return df


def read_args():
    args = argparse.ArgumentParser()
    # args.add_argument("-m", "--mem_thre", default="0", type=str)
    args.add_argument("-t", "--task", type=str)
    args.add_argument("-u", "--user", default="kningtg", type=str)
    args.add_argument("-c", "--command_include", default="", type=str)
    args.add_argument("-n", "--n_gpu", default=20, type=int)
    args.add_argument("-l" "--long", action="store_true")
    args.add_argument("--output_all", action="store_true")
    args.add_argument("--timeout", default=15, type=int)
    args.add_argument("--update", action="store_true")
    return args.parse_args()


'''
def query_node(node_idx, partition_name, info, mem_thre, delay):
    ret = []

    # print(f"node_idx:{node_idx} | partition: {partition_name}")
    cmd = f"""
    module load cuda90/toolkit/9.0.176 && \
    timeout {delay} srun -p {partition_name} -w node{node_idx} \
    nvidia-smi --query-gpu=memory.free,memory.total\
    --format=csv,nounits,noheader
    """
    result = subprocess.Popen(cmd, shell=True, capture_output=True, text=True)
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


def main():
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
'''


def update_server_list(server_info_fn):
    item_list = []

    out = subprocess.run(
        'scontrol show nodes | grep -E "Partitions|NodeName"',
        shell=True,
        text=True,
        capture_output=True,
    ).stdout
    out = out.replace("\n   ", " ")
    for line_index, line in enumerate(out.split("\n")):
        if not line:
            continue
        kv_list = line.split()
        item = {}
        for kv in kv_list:
            if not kv:
                continue
            k, v = kv.split("=")
            item[k] = v

        # print(item)
        if "Partitions" not in item:
            continue

        if item["NodeName"] == "fileserver":
            continue

        ordered_item = OrderedDict(
            [("node_idx", item["NodeName"][-2:]), ("partition", item["Partitions"])]
        )

        item_list += [ordered_item]

    df = pd.DataFrame(item_list)
    df.to_csv(server_info_fn, index=False, header=False)
    print(df)
    print("==================updated===================")


if __name__ == "__main__":
    args = read_args()
    server_info_fn = "~/server_utils/server_list.csv"
    if args.update or not osp.exists(osp.expanduser(server_info_fn)):
        update_server_list(server_info_fn)
    gpu_cluster = GPUCluster(server_info_fn=server_info_fn, timeout=args.timeout)

    if args.task == "usage":
        df = gpu_cluster.find_gpu_usage(username=args.user, cmd_include=args.command_include)
        print(df.to_markdown(index=False))
    elif args.task == "available":
        df = gpu_cluster.find_gpu_available()
        if args.output_all:
            print(df.to_markdown(index=False))
        else:
            print(df.iloc[: args.n_gpu, :].to_markdown(index=False))
