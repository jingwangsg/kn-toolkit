from ..utils.download import Downloader
import argparse
from kn_util.utils.download import get_headers


def get_args():
    parser = argparse.ArgumentParser(description='tools for lfs')

    parser.add_argument("url", type=str, help="The url to parse")
    parser.add_argument("--output", type=str, default="auto", help="The output path")
    parser.add_argument("--num-shards", type=int, default=10, help="The number of shards to download")
    parser.add_argument("--chunk-size", type=int, default=1024, help="The chunk size to download")
    parser.add_argument("--proxy", type=str, default=None, help="The proxy to use")
    parser.add_argument("--direct", action="store_true", default=False, help="Direct download")
    parser.add_argument("--hf", action="store_true", default=False, help="Download from huggingface")
    parser.add_argument("--low-memory", action="store_true", default=False, help="Low memory usage")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    headers = get_headers(from_hf=args.hf) if args.hf else None
    if args.direct:
        Downloader.download(url=args.url, out=args.output, proxy=args.proxy, headers=headers)
    else:
        Downloader.async_sharded_download(url=args.url,
                                          out=args.output,
                                          num_shards=args.num_shards,
                                          chunk_size=args.chunk_size,
                                          headers=headers,
                                          proxy=args.proxy,
                                          low_memory=args.low_memory)
