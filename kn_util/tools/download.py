from ..utils.download import CommandDownloader
import argparse
from ..utils.download import get_hf_headers, get_random_headers, MultiThreadDownloader
from fire import Fire
import os.path as osp
import os


def add_basic_parser(parser):
    parser.add_argument("url", type=str, help="The url to download")
    parser.add_argument(
        "-o", "--output", type=str, help="The output path", default=None
    )
    parser.add_argument("--max-retries", type=int, help="The max retries", default=None)
    parser.add_argument("--proxy", type=str, help="The proxy to use", default=None)
    parser.add_argument("--mode", type=str, help="The mode to use", default="thread")
    parser.add_argument("--token", type=str, help="The token to use", default=None)


def add_thread_parser(parser):
    parser.add_argument(
        "-n", "--num-threads", type=int, help="The number of threads", default=8
    )
    parser.add_argument(
        "--chunk-size", type=int, help="The chunk size", default=1024 * 100
    )
    parser.add_argument("--timeout", type=int, help="The timeout", default=10)
    parser.add_argument(
        "-v", "--verbose", type=int, help="The verbosity level", default=1
    )


def main():
    parser = argparse.ArgumentParser()
    add_basic_parser(parser)

    if args.token is not None:
        os.environ["HF_TOKEN"] = args.token

    args = parser.parse_known_args()[0]
    url = args.url
    output = osp.basename(url) if args.output is None else args.output

    headers = get_random_headers()
    if "huggingface" in url:
        print("=> Detected huggingface url, using huggingface headers")
        headers = get_hf_headers()

    if args.mode == "thread":
        add_thread_parser(parser)
        args = parser.parse_args()

        downloader = MultiThreadDownloader(
            headers=headers,
            num_threads=args.num_threads,
            chunk_size_download=args.chunk_size,
            max_retries=args.max_retries,
            timeout=args.timeout,
            verbose=args.verbose,
        )
        downloader.download(url=url, path=output)

    elif args.mode in ["axel", "wget"]:

        if args.mode == "axel":
            CommandDownloader.download_axel(
                url=args.url,
                out=args.output,
                proxy=args.proxy,
                headers=headers,
            )
        elif args.mode == "wget":
            CommandDownloader.download_wget(
                url=args.url,
                out=args.output,
                proxy=args.proxy,
                headers=headers,
            )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")
