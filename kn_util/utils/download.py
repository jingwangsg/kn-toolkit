from concurrent.futures import ThreadPoolExecutor, wait
import httpx
from tqdm import tqdm
import json
from huggingface_hub.utils._headers import build_hf_headers
from huggingface_hub.utils._headers import _http_user_agent as http_user_agent

# from transformers.utils.hub import http_user_agent
from contextlib import nullcontext
import io
import os
import os.path as osp
import tempfile
import random
from functools import lru_cache
from ..utils.system import run_cmd, force_delete, clear_process
from ..utils.io import save_json, load_json
from ..utils.rich import get_rich_progress_download, add_tasks

from rich.progress import Progress

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

HUGGINGFACE_HEADER_X_LINKED_SIZE = "X-Linked-Size"

def get_hf_headers():
    user_agent_header = http_user_agent()
    token = os.getenv("HF_TOKEN", None)
    assert token is not None, "Please set HF_TOKEN in environment variable"
    headers = build_hf_headers(user_agent=user_agent_header, token=token)
    headers["Accept-Encoding"] = "identity"
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
        proxy=None,
        max_retries=10,
        timeout=10,
        verbose=True,
    ):
        print("=> Downloader Configuration:")
        print(
            json.dumps(
                {
                    "headers": headers,
                    "proxy": proxy,
                    "timeout": timeout,
                    "max_retries": max_retries,
                },
                indent=4,
            )
        )

        self.chunk_size_download = chunk_size_download
        self.headers = headers
        self.verbose = verbose
        self.max_retries = max_retries

        self.client = httpx.Client(
            headers=headers,
            follow_redirects=True,
            proxy=proxy,
            timeout=timeout,
        )

    @lru_cache()
    def get_file_headers(self, url):
        file_headers = self.client.head(url, follow_redirects=False).headers
        return file_headers

    def get_filesize(self, url):
        file_headers = self.get_file_headers(url)
        if HUGGINGFACE_HEADER_X_LINKED_SIZE in file_headers:
            return int(file_headers[HUGGINGFACE_HEADER_X_LINKED_SIZE])
        return int(file_headers["Content-Length"])

    def download(self, url, path):
        filesize = self.get_filesize(url)
        filedir, filename = osp.dirname(path), osp.basename(path)
        if osp.exists(path):
            f = open(path, "rb+")
        else:
            f = open(path, "wb+")

        progress = get_rich_progress_download(disable=(self.verbose == 0))
        progress.start()
        task_id = progress.add_task(filename, total=filesize)
        f.seek(0, os.SEEK_END)
        progress.update(task_id, advance=f.tell())

        with self.client.stream("GET", url) as r:
            for chunk in r.iter_bytes(chunk_size=self.chunk_size_download):
                if chunk:
                    f.write(chunk)
                    progress.update(task_id, advance=len(chunk))

        progress.stop()


def retry_wrapper(max_retries=10):
    def decorator(func):
        def wrapper(*args, **kwargs):
            thread_idx = kwargs["thread_idx"]
            retry_cnt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(
                        f"=> Thread {thread_idx} retry {retry_cnt+1}/{max_retries} failed: {e}"
                    )
                    retry_cnt += 1
                    if max_retries is not None and retry_cnt >= max_retries:
                        break

            raise Exception(
                f"=> Thread {thread_idx} retry {max_retries} times, still failed"
            )

        return wrapper

    return decorator


class MultiThreadDownloader(Downloader):
    def __init__(
        self,
        headers=None,
        num_threads=4,
        max_retries=10,
        timeout=10,
        proxy=None,
        verbose=1,
        chunk_size_download=1024 * 300,
        chunk_size_merge=1024 * 1024 * 500,
        # proxy=None,
    ):
        super().__init__(
            chunk_size_download=chunk_size_download,
            headers=headers,
            max_retries=max_retries,
            timeout=timeout,
            proxy=proxy,
            verbose=verbose,
        )
        self.num_threads = num_threads
        self.chunk_size_merge = chunk_size_merge

    def is_support_range(self, url):
        headers = self.get_file_headers(url)
        return "Accept-Ranges" in headers and headers["Accept-Ranges"] == "bytes"

    def range_merge(self, shard_path, output_file, s_pos, e_pos):
        output_file = open(output_file, "wb+")
        # each thread write to the same file
        cnt = 0
        with open(shard_path, "rb") as f:
            output_file.seek(s_pos)
            for chunk in iter(lambda: f.read(self.chunk_size_merge), b""):
                output_file.write(chunk)
                cnt += len(chunk)

            assert output_file.tell() == e_pos + 1

    def gather_task_progress(self, progress, path, ranges, task_ids):
        # resume task progress
        # don't put this logic inside range_download, which will be called multiple time when reconnect

        total_progress = 0
        for i, (s_pos, e_pos) in enumerate(ranges):
            shard_path = self.get_shard_path(path, i, s_pos, e_pos)
            if osp.exists(shard_path):
                buffer = open(shard_path, "rb+")
            else:
                buffer = open(shard_path, "wb+")
            buffer.seek(0, os.SEEK_END)
            thread_progress = buffer.tell()
            if self.verbose == 2:
                progress.update(task_ids[i + 1], completed=thread_progress)
            total_progress += buffer.tell()
            buffer.close()
        progress.update(task_ids[0], completed=total_progress)

        return total_progress

    def range_download(
        self,
        url,
        s_pos,
        e_pos,
        shard_path,
        thread_idx,
        total_task_id,
        thread_task_id,
        progress,
    ):

        if osp.exists(shard_path):
            buffer = open(shard_path, "rb+")
        else:
            buffer = open(shard_path, "wb+")

        buffer.seek(0, os.SEEK_END)
        skip_bytes = buffer.tell()
        s_pos += skip_bytes

        if s_pos == e_pos + 1:
            return

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
                if self.verbose == 2:
                    progress.update(thread_task_id, advance=len(chunk))
                progress.update(total_task_id, advance=len(chunk))
            assert buffer.tell() == e_pos + 1

    def get_cache_files(self, path):
        dirname, filename = osp.dirname(path), osp.basename(path)
        cache_pattern = osp.join(dirname, f".{filename}.*")
        filenames = [
            _.strip()
            for _ in run_cmd(f"ls {cache_pattern}").stdout.split("\n")
            if _.strip() != ""
        ]
        return filenames

    def clear_cache(self, path):
        filenames = self.get_cache_files(path)
        filename = osp.basename(path)

        try:
            assert all(force_delete(filename) for filename in filenames)
        except:
            import ipdb

            ipdb.set_trace()
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
                # make sure other process is not using same cache files
                cache_files = self.get_cache_files(path)
                for cache_file in cache_files:
                    clear_process(cache_file)

        save_json(download_meta, download_meta_path)

    def get_task_ids(self, progress, ranges, filesize):
        if self.verbose == 2:
            task_ids = add_tasks(
                progress,
                names=["Total"] + [f"Thread {i}" for i in range(self.num_threads)],
                totals=[filesize]
                + [ranges[i][1] - ranges[i][0] + 1 for i in range(self.num_threads)],
            )
        else:
            task_ids = add_tasks(
                progress,
                names=["Total"],
                totals=[filesize],
            )
        return task_ids

    def get_shard_path(self, path, i, s_pos, e_pos):
        file_dir, filename = osp.dirname(path), osp.basename(path)
        return osp.join(file_dir, f".{filename}.part{s_pos}-{e_pos}")

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
            print("=> Server does not support range, use single thread download")
            super().download(url, path)
            return

        progress: Progress = get_rich_progress_download(disable=(self.verbose == 0))
        progress.start()

        task_ids = self.get_task_ids(progress, ranges, filesize)
        self.gather_task_progress(
            progress,
            path=path,
            ranges=ranges,
            task_ids=task_ids,
        )

        executor = ThreadPoolExecutor(max_workers=self.num_threads)
        futures = []
        for i, (s_pos, e_pos) in enumerate(ranges):
            shard_path = self.get_shard_path(path, i, s_pos, e_pos)
            future = executor.submit(
                retry_wrapper(self.max_retries)(self.range_download),
                url=url,
                s_pos=s_pos,
                e_pos=e_pos,
                shard_path=shard_path,
                thread_idx=i,
                total_task_id=task_ids[0],
                thread_task_id=task_ids[i + 1] if self.verbose == 2 else None,
                progress=progress,
            )
            futures.append(future)

        wait(futures, return_when="ALL_COMPLETED")

        progress.stop()

        futures = []
        for i, (s_pos, e_pos) in enumerate(ranges):
            shard_path = self.get_shard_path(path, i, s_pos, e_pos)
            future = executor.submit(
                self.range_merge,
                shard_path=shard_path,
                output_file=path,
                s_pos=s_pos,
                e_pos=e_pos,
            )
            futures.append(future)

        executor.shutdown(wait=True)
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
