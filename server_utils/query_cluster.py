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
        if ("alloc" not in result.stdout and "idle" not in result.stdout and "mix" not in result.stdout):
            if result.stdout:
                print(f"node{node_idx:02d} {result.stdout.split()[-1]}")
            else:
                print(f"node{node_idx:02d} N/A")
            return node_idx, None

        prefix = f"timeout {timeout} srun -N 1 -n 1 -c 1 -p {partition} -w node{node_idx:02d} --export ALL "

        cmd_with_timeout = prefix + cmd

        result = subprocess.run(cmd_with_timeout, shell=True, capture_output=True, text=True)

        output = result.stdout

        if result.stderr.strip() != "":
            print(f"[FAIL] node{node_idx:02d} | {result.stderr} | {result.stdout}")
            output = ""

        return node_idx, output

    def query_all_node(self, cmd):
        inputs_list = [(row["node_idx"], row["partition"], cmd, self.timeout) for _, row in self.server_info.iterrows()]

        st = time.time()
        node_stdout = map_async(iterable=inputs_list, func=self._query_single_node)
        print(f"query costs {time.time()-st}(s)")
        return node_stdout

    def get_memory_dataframe(self):
        gpu_query_cmd = "nvidia-smi --query-gpu=gpu_name,memory.free,memory.total --format=csv,nounits"
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
        import ipdb
        ipdb.set_trace()

        return df

    def find_gpu_available(self):
        df = self.get_memory_dataframe()
        return df

    def get_usage_dataframe_by_py3smi(self):
        gpu_query_cmd = f"py3smi -f --left -w $(($(tput cols)-30))"

        node_stdouts = self.query_all_node(gpu_query_cmd)

        item_list = []

        for node_idx, node_out in node_stdouts:
            if not node_out:
                continue

            lines = [_.strip() for _ in node_out.split("\n")]
            lines = lines[lines.index("") + 5:-2]  # include process info only

            for line in lines:
                line = line.replace(" days", "-days")
                _id, usr, pid, time, _ = line.strip("|").strip().split(maxsplit=4)
                cmd, size = _.rsplit(maxsplit=1)

                item = {
                    "partition": self.server_info[self.server_info["node_idx"] == node_idx]["partition"].item(),
                    "gpu.id": f"node{node_idx:02d}_#" + _id,
                    "gpu.occupied": size,
                    "PID": pid,
                    "user": usr,
                    "time": time,
                    "cmd": cmd,
                }

                item_list += [item]

        df = pd.DataFrame(item_list)

        return df

    def get_usage_dataframe(self):
        pycmd = "from gpustat.core import GPUStatCollection; gpustat = GPUStatCollection.new_query().jsonify(); print(gpustat)"
        cmd = f"python -c '{pycmd}'"

        node_stdouts = self.query_all_node(cmd)
        item_list = []

        for node_idx, node_stdout in node_stdouts:
            if not node_stdout:
                continue
            import datetime

            node_stdout = eval(node_stdout)

            for gpu in node_stdout["gpus"]:
                for process in gpu["processes"]:
                    item = {
                        "partition": self.server_info[self.server_info["node_idx"] == node_idx]["partition"].item(),
                        "gpu.id": f"{node_stdout['hostname']}_#{gpu['index']}",
                        "gpu.occupied": process["gpu_memory_usage"],
                        "PID": process["pid"],
                        "user": process["username"],
                        "cmd": " ".join(process["full_command"]),
                    }
                    item_list += [item]

        return pd.DataFrame(item_list)

    def find_gpu_usage(self, username="", cmd_include=""):
        df = self.get_usage_dataframe()

        if username:
            df = df[df["user"].str.contains(username)]

        if cmd_include:
            df = df[df["cmd"].str.contains(cmd_include)]

        return df


def read_args():
    args = argparse.ArgumentParser()
    # args.add_argument("-m", "--mem_thre", default="0", type=str)
    args.add_argument("-t", "--task", type=str)
    args.add_argument("-u", "--user", default="", type=str)
    args.add_argument("-c", "--command", default="", type=str)
    args.add_argument("-n", "--n_gpu", default=20, type=int)
    args.add_argument("-l"
                      "--long", action="store_true")
    args.add_argument("--timeout", default=15, type=int)
    args.add_argument("--update", action="store_true")
    return args.parse_args()


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

        not_gpu_node = item["NodeName"] == "fileserver" or "Partitions" not in item
        if not_gpu_node:
            continue

        ordered_item = OrderedDict([("node_idx", item["NodeName"][-2:]), ("partition", item["Partitions"])])

        item_list += [ordered_item]

    df = pd.DataFrame(item_list)
    df.to_csv(server_info_fn, index=False, header=False)
    print(df)
    print("==================updated===================")


if __name__ == "__main__":
    args = read_args()
    server_info_fn = osp.expanduser("~/server_utils/server_list.csv")

    if args.update or not osp.exists(server_info_fn):
        update_server_list(server_info_fn)
    gpu_cluster = GPUCluster(server_info_fn=server_info_fn, timeout=args.timeout)

    if args.task == "usage":
        df = gpu_cluster.find_gpu_usage(username=args.user, cmd_include=args.command)
        pd.set_option("display.max_colwidth", 50)
        print(df)
        # print(df.to_markdown(index=False))
    elif args.task == "available":
        df = gpu_cluster.find_gpu_available()
        if args.n_gpu == -1:
            print(df.to_markdown(index=False))
        else:
            print(df.iloc[:args.n_gpu, :].to_markdown(index=False))
    elif args.task == "stat":
        df = gpu_cluster.get_usage_dataframe()
        df["gpu.occupied.value"] = (df["gpu.occupied"] if "int" in str(df["gpu.occupied"].dtype) else
                                    df["gpu.occupied"].str.replace("MiB", "").astype(int))
        result = df.groupby("user").agg({"gpu.id": ["nunique", "count"], "gpu.occupied.value": "sum"})
        result.columns = ["ngpu", "nproc", "mem"]
        result.sort_values(by=["ngpu", "nproc"], ascending=False, inplace=True)
        print(result.to_markdown(index=True))
