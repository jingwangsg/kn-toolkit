import sys
import os

sys.path.insert(0, os.getcwd())
import subprocess
import argparse
import os.path as osp
from ..utils.git_utils import get_origin_url
from ..utils.download import MultiThreadDownloader, get_hf_headers

HF_DOWNLOAD_TEMPLATE = "https://huggingface.co/{org}/{repo}/resolve/main/{path}"


def parse_args():
    parser = argparse.ArgumentParser(description="tools for lfs")
    parser.add_argument("command", type=str, help="The command to run")
    return parser


def run_cmd(cmd, return_output=False):
    # print('Running: {}'.format(cmd))
    if return_output:
        return subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout
    else:
        subprocess.run(cmd, shell=True, check=True)


def lfs_list_files(include=None):
    cmd = "git lfs ls-files"
    if include:
        cmd += ' --include="{}"'.format(include)
    paths = run_cmd(cmd, return_output=True).splitlines()
    paths = [_.split(" ")[-1].strip() for _ in paths]
    return paths


def pull(args):
    paths = lfs_list_files(include=args.include)

    print(f"=> Found {len(paths)} files to fetch")

    for idx in range(0, len(paths), args.chunk):
        print(f"=> Fetching chunk {idx//args.chunk} of {len(paths)//args.chunk}")
        cmd = 'git lfs fetch --include="{}"'.format(",".join(paths[idx : idx + 100]))
        run_cmd(cmd)

    run_cmd("git lfs checkout")


def track(args):
    # args = parse_args()
    cmd = "find ./ -name '*' -type f -not -path './.git*'"

    paths = run_cmd(cmd, return_output=True)
    print(paths)
    cont = input("Continue? (y/n)")
    if cont != "y":
        exit(0)

    paths = paths.splitlines()

    for path in paths:
        cmd = 'git lfs track "{}"'.format(path)
        run_cmd(cmd)


def _parse_repo_url(url):
    """parse org, repo from url
    url like https://huggingface.co/TheBloke/stable-vicuna-13B-GGUF
    """
    items = [_ for _ in url.split("/") if _ != ""]
    org, repo = items[-2:]
    if items[-3] == "datasets":
        org = "datasets/" + org

    return org, repo


def download(
    url_template,
    include=None,
    **downloader_kwargs,
):
    # clone the repo
    paths = lfs_list_files(include=include)
    print(f"=> Found {len(paths)} files to download")

    url = get_origin_url()
    org, repo = _parse_repo_url(url)
    if not osp.exists(".downloaded"):
        run_cmd("touch .downloaded")

    meta_handler = open(".downloaded", "r+")
    downloaded = set([_.strip() for _ in meta_handler.readlines()])
    print(f"=> Found {len(downloaded)} files already downloaded")

    headers = get_hf_headers()
    downloader = MultiThreadDownloader(
        headers=headers,
        **downloader_kwargs,
    )

    for path in paths:
        if path in downloaded:
            print(f"=> {path} already downloaded")
            continue

        url = url_template.format(org=org, repo=repo, path=path)

        print(f"{url} \n=> {path}")
        downloader.download(url=url, path=path)

        meta_handler.write(path + "\n")
        meta_handler.flush()


def download_recursive(**download__kwargs):
    cwd = os.getcwd()
    repos = run_cmd("find ./ -name '.git' -type d", return_output=True).splitlines()
    repos = [osp.join(cwd, osp.dirname(_)) for _ in repos]
    print(f"=> Found {len(repos)} repos")
    for repo in repos:
        print(f"=> Downloading {repo}")
        os.chdir(repo)
        download(
            url_template=HF_DOWNLOAD_TEMPLATE,
            **download__kwargs,
        )


def main():
    parser = parse_args()
    command = parser.parse_known_args()[0].command

    if command == "pull":
        parser.add_argument(
            "--chunk", type=int, help="The chunk number to fetch", default=100
        )
        parser.add_argument(
            "--include",
            type=str,
            help="The partial path to fetch, split by ,",
            default=None,
        )
        args = parser.parse_args()
        pull(args)
    elif command == "track":
        args = parser.parse_args()
        track(args)
    elif command == "download":
        parser.add_argument(
            "--include",
            type=str,
            help="The partial path to fetch, split by ,",
            default=None,
        )
        parser.add_argument(
            "--template",
            type=str,
            help="The chunk number to fetch",
            default=HF_DOWNLOAD_TEMPLATE,
        )
        parser.add_argument("--proxy", type=str, help="The proxy to use", default=None)
        parser.add_argument(
            "--recursive",
            action="store_true",
            help="Whether to download recursively",
            default=False,
        )
        parser.add_argument(
            "--num-shards",
            type=int,
            help="The number of shards to use",
            default=1,
        )
        parser.add_argument(
            "--max-retries",
            type=int,
            help="The number of retries to use",
            default=10,
        )
        parser.add_argument(
            "-n",
            "--num-threads",
            type=int,
            help="The number of threads to use",
            default=8,
        )
        parser.add_argument(
            "--verbose",
            type=int,
            default=1,
            help="Whether to print verbose information",
        )
        args = parser.parse_args()

        if not args.recursive:
            download(
                url_template=args.template,
                include=args.include,
                num_threads=args.num_threads,
                max_retries=args.max_retries,
                proxy=args.proxy,
                verbose=args.verbose,
            )
        else:
            print(
                "=> Downloading recursively! only supports huggingface git repos for now"
            )
            download_recursive(
                num_threads=args.num_threads,
                max_retries=args.max_retries,
                proxy=args.proxy,
                verbose=args.verbose,
            )
