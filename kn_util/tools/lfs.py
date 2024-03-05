import sys
import os

sys.path.insert(0, os.getcwd())
from kn_util.utils.logger import setup_logger_loguru
import subprocess
import argparse
import os.path as osp
from ..utils.git_utils import get_origin_url
from ..utils.download import MultiThreadDownloader, get_hf_headers, Downloader
from ..utils.rich import get_rich_progress_download
from rich.console import Group
from rich.live import Live
from rich.progress import Progress
import copy
import time
import multiprocessing as mp

# from concurrent.futures import ProcessPoolExecutor, wait
from pathos.multiprocessing import ProcessPool

HF_DOWNLOAD_TEMPLATE = "https://huggingface.co/{org}/{repo}/resolve/main/{path}"


def parse_args():
    parser = argparse.ArgumentParser(description="tools for lfs")
    parser.add_argument("command", type=str, help="The command to run")
    return parser


def run_cmd(cmd, return_output=False):
    # print('Running: {}'.format(cmd))
    if return_output:
        return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout
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


def download_file(downloader, url, path):
    downloader.download(url, path)


def wait(not_done, timeout=0.5):
    done = set()
    time.sleep(timeout)
    _not_done = copy.copy(not_done)
    for future in _not_done:
        if future.ready():
            not_done.remove(future)
            done.add(future)
            continue
    return done, not_done


def download_repo(
    url_template,
    include=None,
    num_processes=1,
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

    paths = [path for path in paths if path not in downloaded]
    urls = [url_template.format(org=org, repo=repo, path=path) for path in paths]

    # executor = ProcessPoolExecutor(max_workers=num_processes)
    process_pool = ProcessPool(num_processes)

    progress = get_rich_progress_download()
    for _ in range(num_processes):
        progress.add_task("", visible=False)

    progress.start()

    url_path = iter(zip(urls, paths))

    downloader_kwargs["verbose"] = 0

    # TODO: why do we need a manager here? mp.Queue fails
    manager = mp.Manager()
    downloaders = [
        MultiThreadDownloader(
            headers=headers,
            **downloader_kwargs,
            queue=manager.Queue(),
        )
        for _ in range(num_processes)
    ]

    not_done = set()
    for process_id in range(num_processes):
        url, path = next(url_path, (None, None))
        if (url is None) or (path is None):
            continue

        future = process_pool.apipe(
            downloaders[process_id].download,
            url=url,
            path=path,
        )
        downloaders[process_id]._path = path
        future._process_id = process_id
        future._path = path

        not_done.add(future)

    # start polling
    while not_done:
        done, not_done = wait(not_done, timeout=0.5)

        for process_id in range(num_processes):
            message_queue = downloaders[process_id].message_queue

            path = getattr(downloaders[process_id], "_path", None)
            if path is None:
                # this means downloader is not downloading anything
                continue
            while True:
                try:
                    message = message_queue.get_nowait()
                    if message[0] == "filesize":
                        progress.update(
                            process_id,
                            total=message[1],
                            completed=0,
                            description=f"{path} [{process_id:02d}]",
                            visible=True,
                            refresh=True,
                        )
                    elif message[0] == "advance":
                        progress.update(process_id, advance=message[1])
                except:
                    break

        for future in done:
            process_id = future._process_id
            path = future._path
            downloaders[process_id].clear_message()

            meta_handler.write(path + "\n")
            meta_handler.flush()

            url, path = next(url_path, (None, None))
            if url is not None:
                # apipe equals to submit in ProcessPoolExecutor
                future = process_pool.apipe(
                    downloaders[process_id].download,
                    url=url,
                    path=path,
                )
                downloaders[process_id]._path = path
                future._process_id = process_id
                future._path = path
                not_done.add(future)

    progress.stop()


def download_recursive(**download_kwargs):
    cwd = os.getcwd()
    repos = run_cmd("find ./ -name '.git' -type d", return_output=True).splitlines()
    repos = [osp.join(cwd, osp.dirname(_)) for _ in repos]
    print(f"=> Found {len(repos)} repos")
    for repo in repos:
        print(f"=> Downloading {repo}")
        os.chdir(repo)
        download_repo(
            url_template=HF_DOWNLOAD_TEMPLATE,
            **download_kwargs,
        )


def main():
    parser = parse_args()
    command = parser.parse_known_args()[0].command

    if command == "pull":
        parser.add_argument("--chunk", type=int, help="The chunk number to fetch", default=100)
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
        parser.add_argument("--proxy-port", type=int, help="The proxy port to use", default=None)
        parser.add_argument(
            "--recursive",
            action="store_true",
            help="Whether to download recursively",
            default=False,
        )
        parser.add_argument("--num-processes", type=int, help="The number of process to use", default=1)
        parser.add_argument("--max-retries", type=int, help="The number of retries to use", default=None)
        parser.add_argument(
            "-n",
            "--num-threads",
            type=int,
            help="The number of threads to use",
            default=4,
        )
        parser.add_argument("--timeout", type=int, help="The timeout", default=10)
        parser.add_argument(
            "--verbose",
            type=int,
            default=1,
            help="Whether to print verbose information",
        )
        parser.add_argument("--log-stdout", action="store_true", help="Whether to log to stdout")
        parser.add_argument("--log-file", type=str, help="The log file to use", default=None)
        parser.add_argument("--token", type=str, help="The token to use", default=None)
        args = parser.parse_args()

        if args.token is not None:
            os.environ["HF_TOKEN"] = args.token

        setup_logger_loguru(
            filename=args.log_file,
            stdout=args.log_stdout,
            include_filepath=False,
        )

        proxy = None
        if args.proxy or args.proxy_port:
            proxy = f"http://127.0.0.1:{args.proxy_port}" if args.proxy_port else args.proxy

        if not args.recursive:
            download_repo(
                url_template=args.template,
                include=args.include,
                num_threads=args.num_threads,
                num_processes=args.num_processes,
                max_retries=args.max_retries,
                timeout=args.timeout,
                proxy=proxy,
                # verbose=args.verbose,
            )
        else:
            print("=> Downloading recursively! only supports huggingface git repos for now")
            download_recursive(
                num_processes=args.num_processes,
                num_threads=args.num_threads,
                max_retries=args.max_retries,
                timeout=args.timeout,
                proxy=proxy,
                # verbose=args.verbose,
            )
