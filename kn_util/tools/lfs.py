import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='tools for lfs')
    parser.add_argument("command", type=str, help="The command to run")
    return parser


def add_pull_args(parser):
    parser.add_argument("--chunk", type=int, help="The chunk number to fetch", default=100)
    parser.add_argument("--include", type=str, help="The partial path to fetch, split by ,", default=None)
    return parser


def run_cmd(cmd, return_output=False):
    # print('Running: {}'.format(cmd))
    if return_output:
        return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout
    else:
        subprocess.run(cmd, shell=True, check=True)


def pull(parser):
    parser = add_pull_args(parser)
    args = parser.parse_args()
    cmd = "git lfs ls-files"
    if args.include:
        cmd += " --include=\"{}\"".format(args.include)
    paths = run_cmd(cmd, return_output=True).splitlines()
    paths = [_.split(" - ")[-1] for _ in paths]

    print(f"Found {len(paths)} files to fetch")

    for idx in range(0, len(paths), args.chunk):
        print(f"Fetching chunk {idx//args.chunk} of {len(paths)//args.chunk}")
        cmd = "git lfs fetch --include=\"{}\"".format(",".join(paths[idx:idx + 100]))
        run_cmd(cmd)

    run_cmd("git lfs checkout")


def track(parser):
    # args = parse_args()
    cmd = "find ./ -name '*' -type f -not -path './.git*'"

    paths = run_cmd(cmd, return_output=True)
    print(paths)
    cont = input("Continue? (y/n)")
    if cont != "y":
        exit(0)

    paths = paths.splitlines()

    for path in paths:
        cmd = "git lfs track \"{}\"".format(path)
        run_cmd(cmd)


if __name__ == "__main__":
    parser = parse_args()
    command = parser.parse_known_args()[0].command

    if command == "pull":
        pull(parser)
    elif command == "track":
        track(parser)
