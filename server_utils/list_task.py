from re import sub
import subprocess
import argparse
from tqdm import tqdm

def read_args():
    args = argparse.ArgumentParser()
    args.add_argument("-u", "--user", type=str, default="kningtg")
    args.add_argument("-k", "--kill", action="store_true")

    return args.parse_args()

def run_cmd(cmd):
    # print(cmd)
    ret_str = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return ret_str

if __name__ == "__main__":
    args = read_args()
    ret_str = run_cmd(f"squeue | grep {args.user}").stdout
    print(ret_str)
    if ret_str != "":
        sp_ret_str = ret_str.split("\n")
        if isinstance(sp_ret_str, str):
            sp_ret_str = [sp_ret_str]
        pids = [x.split()[0].strip() for x in sp_ret_str if x != ""]
        if args.kill:
            for pid in tqdm(pids, desc="killing..."):
                run_cmd(f"scancel {pid}")