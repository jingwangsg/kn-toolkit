from concurrent.futures import ThreadPoolExecutor
import httpx
from tqdm import tqdm
from huggingface_hub.utils._headers import build_hf_headers
from huggingface_hub.utils._headers import _http_user_agent as http_user_agent
from contextlib import nullcontext
import io
import os
import os.path as osp
import tempfile
import random
from ..utils.system import run_cmd
from ..utils.io import save_json, load_json


# https://www.iamhippo.com/2021-08/1546.html
USER_AGENT_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
]


def get_pbar(total, desc=None, disable=False, **kwargs):
    return tqdm(
        total=total,
        desc=desc,
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        smoothing=0.1,
        miniters=1,
        ascii=True,
        disable=disable,
        **kwargs,
    )


def get_hf_headers():
    user_agent_header = http_user_agent()
    headers = build_hf_headers(
        user_agent=user_agent_header, token="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"
    )
    return headers


def get_random_headers():
    # only for huggingface
    user_agent_header = USER_AGENT_LIST[random.randint(0, len(USER_AGENT_LIST) - 1)]
    headers = {"User-Agent": user_agent_header}
    return headers


class Downloader:
    def __init__(
        self,
        chunk_size_download=1024,
        headers=None,
        verbose=True,
    ):
        self.chunk_size_download = chunk_size_download
        self.headers = headers
        self.verbose = verbose

    def download(url):
        pass


def retry_wrapper(max_retries=10):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"=> Retry {i+1}/{max_retries} failed: {e}")
            raise Exception(f"Retry {max_retries} times, still failed")

        return wrapper

    return decorator


class MultiThreadDownloader(Downloader):
    def __init__(
        self,
        headers=None,
        verbose=1,
        num_threads=4,
        max_retries=10,
        chunk_size_download=1024 * 100,
        chunk_size_merge=1024**3,
        # proxy=None,
    ):
        super().__init__(
            chunk_size_download=chunk_size_download,
            headers=headers,
            verbose=verbose,
        )
        self.num_threads = num_threads
        self.max_retries = max_retries
        self.client = httpx.Client(
            headers=headers,
            follow_redirects=True,
            timeout=None,
        )
        self.chunk_size_merge = chunk_size_merge

    def is_support_range(self, url):
        headers = self.client.head(url).headers
        return "Accept-Ranges" in headers and headers["Accept-Ranges"] == "bytes"

    def get_filesize(self, url):
        return int(self.client.head(url).headers["Content-Length"])

    def range_merge(self, shard_path, output_file, s_pos, e_pos):
        with open(shard_path, "rb") as f:
            f.seek(s_pos)
            output_file.seek(s_pos)
            for chunk in iter(lambda: f.read(self.chunk_size_merge), b""):
                output_file.write(chunk)
            assert f.tell() == e_pos + 1

    def range_download(self, url, s_pos, e_pos, shard_path, thread_idx, pbar):

        thread_pbar = get_pbar(
            e_pos - s_pos + 1,
            desc=f"Thread {thread_idx}",
            position=thread_idx + 1,
            disable=self.verbose < 2,
        )

        if osp.exists(shard_path):
            buffer = open(shard_path, "rb+")
        else:
            buffer = open(shard_path, "wb+")

        buffer.seek(0, os.SEEK_END)
        skip_bytes = buffer.tell()
        s_pos += skip_bytes
        pbar.update(skip_bytes)
        thread_pbar.update(skip_bytes)

        with self.client.stream(
            "GET",
            url,
            headers={
                **self.headers,
                "Range": f"bytes={s_pos}-{e_pos}",
            },
        ) as r:
            for chunk in r.iter_bytes(chunk_size=self.chunk_size_download):
                buffer.write(chunk)
                if pbar:
                    pbar.update(len(chunk))
                    thread_pbar.update(len(chunk))

    def clear_cache(self, path):
        dirname, filename = osp.dirname(path), osp.basename(path)
        cache_pattern = osp.join(dirname, f".{filename}.*")
        run_cmd(f"rm -rf {cache_pattern}")
        print(f"=> Cache of {filename} cleared")

    def resolve_download_meta(self, url, path, filesize):
        file_dir, filename = osp.dirname(path), osp.basename(path)
        download_meta_path = osp.join(file_dir, f".{filename}.meta.json")

        download_meta = {
            "url": url,
            "filesize": filesize,
            "num_threads": self.num_threads,
        }
        if osp.exists(download_meta_path):
            download_meta_load = load_json(download_meta_path)
            if download_meta_load != download_meta:
                print("=> Download meta file exists but not match, re-download")
                self.clear_cache(path)
        else:
            save_json(download_meta, download_meta_path)

    def download(self, url, path):
        file_dir, filename = osp.dirname(path), osp.basename(path)

        filesize = self.get_filesize(url)
        file_chunk_size = (filesize + self.num_threads - 1) // self.num_threads
        ranges = [
            (i * file_chunk_size, min((i + 1) * file_chunk_size - 1, filesize - 1))
            for i in range(self.num_threads)
        ]
        self.resolve_download_meta(url, path, filesize)

        if not self.is_support_range(url):
            print(f"{url} does not support range download")
            return

        # self.range_download(url, 0, filesize, path, 0)
        pbar = None
        pbar = get_pbar(
            filesize,
            desc=f"Downloading {filename}",
            disable=not self.verbose,
            position=0,
        )

        thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
        for i, (s_pos, e_pos) in enumerate(ranges):
            shard_path = osp.join(file_dir, f".{filename}.{i:02d}")
            thread_pool.submit(
                self.range_download,
                url=url,
                s_pos=s_pos,
                e_pos=e_pos,
                shard_path=shard_path,
                thread_idx=i,
                pbar=pbar,
            )

        output_file = open(path, "wb+")
        for i, (s_pos, e_pos) in enumerate(ranges):
            shard_path = osp.join(file_dir, f".{filename}.{i:02d}")
            thread_pool.submit(
                self.range_merge,
                shard_path=shard_path,
                output_file=output_file,
                s_pos=s_pos,
                e_pos=e_pos,
            )

        thread_pool.shutdown(wait=True)
        self.clear_cache(path)


class CommandDownloader(Downloader):

    @classmethod
    def download_axel(
        cls,
        url,
        out=None,
        headers=None,
        proxy=None,
        num_shards=None,
        timeout=5,
        retries=3,
        verbose=True,
    ):
        if out == "auto":
            out = cls.get_output_path(url)

        if proxy == "auto":
            proxy = "127.0.0.1:8091"

        axel_args = ""
        if proxy:
            axel_args += f" --proxy {proxy}"

        if headers:
            for k, v in headers.items():
                axel_args += f' --header "{k}:{v}"'

        if num_shards:
            axel_args += f" --num-connections {num_shards}"

        axel_args += f" --max-redirect {retries} --timeout {timeout}"

        cmd = f"axel {axel_args} '{url}' -o '{out}'"
        if out is not None:
            run_cmd(cmd, verbose=verbose)
        else:
            with tempfile.NamedTemporaryFile() as f, io.BytesIO() as buffer:
                run_cmd(cmd, verbose=verbose, out=f.name)
                buffer.write(f.read())
            return buffer

    @classmethod
    def download_wget(
        cls, url, out=None, headers=None, proxy=None, timeout=5, retries=3, verbose=True
    ):
        if out == "auto":
            out = cls.get_output_path(url)

        if proxy == "auto":
            proxy = "127.0.0.1:8091"

        wget_args = ""
        if proxy:
            wget_args += f" --proxy=on --proxy http://{proxy}"

        wget_args += (
            f" --tries {retries} --timeout {timeout} --no-check-certificate --continue"
        )

        if headers:
            for k, v in headers.items():
                wget_args += f' --header "{k}:{v}"'

        if out is not None:
            cmd = f"wget {wget_args} '{url}' -O '{out}'"
            run_cmd(cmd, verbose=verbose)
        else:
            with tempfile.NamedTemporaryFile() as f, io.BytesIO() as buffer:
                cmd = f"wget {wget_args} '{url}' -O '{f.name}'"
                run_cmd(cmd, verbose=verbose)
                buffer.write(f.read())
            buffer.seek(0)
            return buffer
