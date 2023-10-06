from ..utils.download import Downloader
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='tools for lfs')

    parser.add_argument("url", type=str, help="The url to parse")
    parser.add_argument("--output", type=str, default=None, help="The output path")
    parser.add_argument("--num_shards", type=int, default=10, help="The number of shards to download")
    parser.add_argument("--chunk_size", type=int, default=1024, help="The chunk size to download")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    Downloader.async_sharded_download_to_file(url=args.url,
                                              save_name=args.output,
                                              num_shards=args.num_shards,
                                              chunk_size=args.chunk_size)
