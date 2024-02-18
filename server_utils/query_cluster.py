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
import json
import numpy as np

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext


def map_async_with_thread(
    iterable,
    func,
    num_thread=30,
    desc="",
    verbose=True,
):

    with ThreadPoolExecutor(num_thread) as executor:
        pbar = tqdm(total=len(iterable), desc=desc) if verbose else None
        context = pbar if pbar else nullcontext()

        results = []

        with context:
            futures = {executor.submit(func, x): x for x in iterable}

            for future in as_completed(futures):
                if pbar:
                    pbar.update(1)
                try:
                    result = future.result()  # Get the result from the future
                    results.append(result)  # Append the result to the results list
                except Exception as e:
                    # If there is an exception, you can handle it here
                    # For now, we'll just print it
                    print(f"Exception in thread: {e}")

        return results


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
        server_info = pd.read_csv(
            server_info_fn, names=["node_idx", "partition", "gpu_type"]
        )
        server_info["partition"] = server_info["partition"].astype(str)
        server_info["gpu_type"] = server_info["gpu_type"].astype(str)
        self.server_info = server_info
        self.timeout = timeout

    def _check_node(self, node_idx):
        cmd = f"sinfo --nodes=node{node_idx:02d} -N --noheader"
        result = subprocess.run(cmd, text=True, capture_output=True, shell=True)
        status_str = "N/A" if not result.stdout else result.stdout.split()[-1]
        return status_str

    def _query_single_node(self, inputs):
        node_idx, partition, cmd, timeout = inputs

        status_str = self._check_node(node_idx)

        invalid_status = ["drain", "fail", "drain", "drng", "down"]
        if any([status in status_str for status in invalid_status]):

            self.failed += [{"node": f"node{node_idx:02d}", "err": status_str}]
            return node_idx, None

        prefix = f"timeout {timeout} srun -p {partition} -w node{node_idx:02d} --export ALL --mem=0 "

        cmd_with_timeout = prefix + cmd

        result = subprocess.run(
            cmd_with_timeout, shell=True, capture_output=True, text=True
        )

        output = result.stdout.strip()

        if (
            result.stderr.strip() != ""
            and "has been allocated resources" not in result.stderr
        ):
            self.failed += [
                {"node": f"node{node_idx:02d}", "err": result.stderr.split("\n")[0]}
            ]
            return node_idx, None

        return node_idx, output

    def query_all_node(self, cmd):
        inputs_list = [
            (row["node_idx"], row["partition"], cmd, self.timeout)
            for _, row in self.server_info.iterrows()
        ]
        self.failed = []

        st = time.time()
        node_stdout = map_async_with_thread(
            iterable=inputs_list, func=self._query_single_node
        )

        print(f"query costs {time.time()-st}(s)")

        # failed_df = pd.DataFrame(self.failed)
        # print(failed_df.to_markdown(index=False))
        if len(self.failed) > 0:
            print("Failed nodes:")
            print(", ".join([f"{x['node']}({x['err']})" for x in self.failed]))
        print("\n")

        return node_stdout

    def get_memory_dataframe(self):
        gpu_query_cmd = "gpustat --json"
        node_stdouts = self.query_all_node(gpu_query_cmd)
        node_stdouts = [x for x in node_stdouts if x[1]]

        df_list = []

        for node_idx, node_out in node_stdouts:
            gpu_infos = json.loads(node_out)["gpus"]
            mem_usage = np.sum(
                [p["cpu_memory_usage"] for gpu in gpu_infos for p in gpu["processes"]]
            )
            mem_usage_gb = mem_usage / 1024 / 1024 / 1024
            mem_usage_gb_str = f"{int(np.round(mem_usage_gb)):03d} Gb"
            cpu_usage = (
                np.sum(
                    [
                        np.sum([p["cpu_percent"] for p in gpu["processes"]])
                        for gpu in gpu_infos
                    ]
                )
                / 100
            )
            cpu_usage_str = f"{int(np.round(cpu_usage))}"

            for gpu in gpu_infos:
                users = ", ".join(
                    list(set(f'{p["username"]}' for p in gpu["processes"]))
                )
                item = {
                    "gpu.id": f"node{node_idx:02d}_gpu#{gpu['index']}",
                    "name": gpu["name"],
                    "gpu\n.util": gpu["utilization.gpu"],
                    "memory\n.free": gpu["memory.total"] - gpu["memory.used"],
                    "memory\n.total": gpu["memory.total"],
                    "proc\n.num": len(gpu["processes"]),
                    "proc\n.users": users,
                    # "processes.cpu_usage": ", ".join([f'{p["cpu_percent"]:.1f}%' for p in gpu["processes"]]),
                    "node\n.cpu": cpu_usage_str,
                    "node\n.mem": mem_usage_gb_str,
                }
                df_list += [item]
        df = pd.DataFrame(df_list)

        return df

    def find_gpu_available(self, full=True, sorted=True):
        df = self.get_memory_dataframe()
        if not full:
            columns = [
                "gpu.id",
                "name",
                "gpu\n.util",
                "memory\n.free",
                "memory\n.total",
                "node\n.cpu",
            ]
            df = df[columns]

        if sorted:
            df["weight"] = df["memory\n.free"] * (1 - df["gpu\n.util"] / 100)
            df = df.sort_values(by=["weight"], ascending=False)
            df = df.drop(columns=["weight"])
        else:
            df = df.sort_values(by=["gpu.id"])
        return df

    def get_usage_dataframe(self):
        # pycmd = "from gpustat.core import GPUStatCollection; gpustat = GPUStatCollection.new_query().jsonify(); print(gpustat)"
        # cmd = f"python -c '{pycmd}'"
        cmd = "gpustat -f --json"

        node_stdouts = self.query_all_node(cmd)
        node_stdouts = [x for x in node_stdouts if x[1]]
        item_list = []

        for node_idx, node_stdout in node_stdouts:
            import datetime

            node_stdout = json.loads(node_stdout)

            for gpu in node_stdout["gpus"]:
                for process in gpu["processes"]:
                    item = {
                        "gpu\n.name": self.server_info[
                            self.server_info["node_idx"] == node_idx
                        ]["partition"].item(),
                        "gpu\n.id": f"{node_stdout['hostname']}_#{gpu['index']}",
                        "gpu\n.used": process["gpu_memory_usage"],
                        "gpu\n.util": gpu["utilization.gpu"],
                        "PID": process["pid"],
                        "user": process["username"],
                        "cmd": " ".join(process["full_command"])[:30],
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


def add_args_usage(parser):
    parser.add_argument("-u", "--user", default="", type=str)
    parser.add_argument("-c", "--command", default="", type=str)


def add_args_avail(parser):
    parser.add_argument("-f", "--full", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("-n", "--n_gpu", default=30, type=int)
    parser.add_argument("--update", action="store_true")


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

        ordered_item = OrderedDict(
            [("node_idx", item["NodeName"][-2:]), ("partition", item["Partitions"])]
        )

        item_list += [ordered_item]

    df = pd.DataFrame(item_list)
    df.to_csv(server_info_fn, index=False, header=False)
    print(df)
    print("==================updated===================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_known_args()[0]
    server_info_fn = osp.join(osp.dirname(__file__), "server_list.csv")

    gpu_cluster = GPUCluster(
        server_info_fn=server_info_fn,
        timeout=args.timeout,
    )

    if args.task == "usage":
        add_args_usage(parser)
        args = parser.parse_args()

        df = gpu_cluster.find_gpu_usage(
            username=args.user,
            cmd_include=args.command,
        )
        # print(df)
        print(df.to_markdown(index=False))
    elif args.task == "available":
        add_args_avail(parser)
        args = parser.parse_args()

        if args.update or not osp.exists(server_info_fn):
            update_server_list(server_info_fn)
        df = gpu_cluster.find_gpu_available(full=args.full, sorted=not args.all)
        if args.all:
            print(df.to_markdown(index=False))
        else:
            print(df.iloc[: args.n_gpu, :].to_markdown(index=False))
    elif args.task == "stat":
        df = gpu_cluster.get_usage_dataframe()
        result = df.groupby("user").agg(
            {"gpu\n.id": ["nunique", "count"], "gpu\n.used": ["sum"]}
        )
        result.columns = ["ngpu", "nproc", "mem"]
        result.sort_values(by=["ngpu", "nproc"], ascending=False, inplace=True)
        print(result.to_markdown(index=True))
