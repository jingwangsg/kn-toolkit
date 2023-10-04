import os
import argparse
import os.path as osp
import subprocess
import glob


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--homebrew_bin", type=str, default=osp.expanduser("~/homebrew/bin"))
    return args.parse_args()


args = parse_args()
all_executable = glob.glob(args.homebrew_bin + "/*")
all_executable = [
    subprocess.run(f"readlink -f {executable}", shell=True, capture_output=True, text=True).stdout
    for executable in all_executable
]


def maybe_patch(executable):
    print(executable)
    ret = subprocess.run(f"timeout 1 {executable} --version", shell=True, text=True, capture_output=True)
    if "/lib/x86_64-linux-gnu/" in ret.stderr:
        print(f"patching f{executable}")
        subprocess.run(f"bash ~/patch.sh {executable} $HOME/homebrew/opt/glibc/lib/", shell=True)


from kn_util.basic import map_async

map_async(iterable=all_executable, func=maybe_patch)
