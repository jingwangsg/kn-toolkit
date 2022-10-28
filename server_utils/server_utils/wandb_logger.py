import wandb
from server_utils.query_cluster import find_gpu
import pandas as pd
import time
import matplotlib.pyplot as plt


def log_gpu():
    df = find_gpu(mem_thre=0, delay=20)
    df["node"] = pd.Series(df.index, index=df.index).apply(
        lambda x: x.split("_gpu_")[0]
    )
    df["gpu_id"] = pd.Series(df.index, index=df.index).apply(
        lambda x: x.split("_gpu_")[1]
    )

    df = df[["node", "gpu_id", "mem_rest", "mem_total", "info"]]
    new_cols = {
        "node": "nd",
        "gpu_id": "gid",
        "mem_rest": "R",
        "mem_total": "T",
        "info": "info",
    }
    df = df.rename(columns=new_cols)

    wandb_df = wandb.Table(dataframe=df)

    # wandb_df = wandb.Table(columns=["node", "gpu-status"])

    # node_list = df["node"].unique().tolist()
    # for node in node_list:
    #     cur_df = df[df["node"] == node]

    #     gpu_status_df = cur_df[["gpu_id", "mem_rest"]].copy()
    #     gpu_status_df["mem_used"] = cur_df["mem_total"] - cur_df["mem_rest"]
    #     gpu_status_df = gpu_status_df.set_index(["gpu_id"])
    #     gpu_status_df.sort_values(by=["mem_rest"], ascending=False, inplace=True)
    #     gpu_status_fig = plt.figure()
    #     ax = gpu_status_fig.add_subplot(111)
    #     gpu_status_df.plot.bar(stacked=True, ax=ax)

    #     wandb_df.add_data(node, gpu_status_fig)

    # wandb.log({"gpu-monitor": wandb_df})
    # import ipdb

    # ipdb.set_trace()  # FIXME
    wandb.log({"gpu-monitor": wandb_df})


if __name__ == "__main__":
    wandb.init(project="gpu-cluster")
    while True:
        log_gpu()
        time.sleep(600)
