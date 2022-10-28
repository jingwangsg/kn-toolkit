import argparse
import pandas as pd
import subprocess


def read_args():
    args = argparse.ArgumentParser()
    args.add_argument("--server_nos", type=str)
    return args.parse_args()


if __name__ == "__main__":
    args = read_args()
    server_list = pd.read_csv(
        "/export/home2/kningtg/server_utils/server_list.csv", header=None
    )
    server_list.set_axis(["no", "partition", "gpu"], axis=1, inplace=True)

    server_nos = [int(no) for no in args.server_nos.split(",")]

    server_list = server_list[server_list["no"].isin(server_nos)]
    partition = server_list["partition"].to_list()
    assert len(set(partition)) == 1
    partition = partition[0]

    # print(server_list)
    node_args = ",".join([f"node{idx:02d}" for idx in server_nos])
    # subprocess.run(
    #     f"srun -p {partition} -w {node_args} --export ALL --mem=0 nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv",
    #     shell=True,
    # )
    print(
        f"srun -p {partition} -w {node_args} --export ALL --mem=0 --pty bash"
    )
