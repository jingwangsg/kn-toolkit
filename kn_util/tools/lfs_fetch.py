import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Fetches all files in a git lfs batch')
    parser.add_argument("--chunk", type=int, help="The chunk number to fetch", default=100)

    return parser.parse_args()


def run_cmd(cmd, return_output=False):
    # print('Running: {}'.format(cmd))
    if return_output:
        return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout
    else:
        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    args = parse_args()
    paths = run_cmd("git lfs ls-files", return_output=True).splitlines()
    paths = [_.split(" - ")[-1] for _ in paths]
    print(f"Found {len(paths)} files to fetch")

    for idx in range(0, len(paths), args.chunk):
        print(f"Fetching chunk {idx//args.chunk} of {len(paths)//args.chunk}")
        cmd = "git lfs fetch --include=\"{}\"".format(",".join(paths[idx:idx + 100]))
        run_cmd(cmd)

    run_cmd("git lfs checkout")
