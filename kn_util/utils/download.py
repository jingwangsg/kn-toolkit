from kn_util.utils.logger import setup_logger_loguru

setup_logger_loguru(stdout=False, filename=None)
from concurrent.futures import ThreadPoolExecutor, wait
import httpx
from tqdm import tqdm
import json
from huggingface_hub.utils._headers import build_hf_headers
from huggingface_hub.utils._headers import _http_user_agent as http_user_agent
from loguru import logger

# from transformers.utils.hub import http_user_agent
from contextlib import nullcontext
import io
import os
import os.path as osp
import tempfile
import random
from functools import lru_cache
from queue import Queue
from queue import Empty as EmptyQueue

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
        chunk_size_download=1024 * 100,
        headers=None,
        proxy=None,
        max_retries=10,
        timeout=10,
        verbose=True,
    ):
        logger.info("=> Downloader Configuration:")
        logger.info(
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
        self.proxy = proxy
        self.timeout = timeout

    @lru_cache()
    def get_file_headers(self, client, url):
        file_headers = client.head(url, follow_redirects=False).headers
        return file_headers

    def get_filesize(self, client, url):
        file_headers = self.get_file_headers(client, url)
        if HUGGINGFACE_HEADER_X_LINKED_SIZE in file_headers:
            return int(file_headers[HUGGINGFACE_HEADER_X_LINKED_SIZE])
        return int(file_headers["Content-Length"])

    def download(self, url, path):

        client = httpx.Client(
            headers=self.headers,
            timeout=self.timeout,
            proxies=self.proxy,
        )

        filesize = self.get_filesize(client, url)
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

        try:
            with client.stream("GET", url) as r:
                for chunk in r.iter_bytes(chunk_size=self.chunk_size_download):
                    if chunk:
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
        except Exception as e:
            logger.error(f"=> Download failed: {e} Pos: {f.tell()}")

        progress.stop()


def retry_wrapper(max_retries=10):
    def decorator(func):
        def wrapper(*args, **kwargs):
            thread_id = kwargs["thread_id"]
            retry_cnt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # logger.info(f"=> Thread {thread_id} retry {retry_cnt+1}/{max_retries} failed: {e}")
                    retry_cnt += 1
                    if max_retries is not None and retry_cnt >= max_retries:
                        break

            raise Exception(f"=> Thread {thread_id} retry {max_retries} times, still failed")

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
        chunk_size_download=1024 * 10,
        chunk_size_merge=1024 * 1024 * 500,
        queue=None,
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
        self.message_queue = Queue() if queue is None else queue

    def is_support_range(self, client, url):
        headers = self.get_file_headers(client, url)
        return "Accept-Ranges" in headers and headers["Accept-Ranges"] == "bytes"

    def gather_for_resume(self, path, ranges, message_queue):
        # resume task progress
        # don't put this logic inside range_download, which will be called multiple time when reconnect
        for i, (s_pos, e_pos) in enumerate(ranges):
            shard_path = self.get_shard_path(path, i, s_pos, e_pos)
            if osp.exists(shard_path):
                buffer = open(shard_path, "rb+")
            else:
                buffer = open(shard_path, "wb+")
            buffer.seek(0, os.SEEK_END)

            message_queue.put_nowait(("advance", buffer.tell(), i))
            buffer.close()

    def range_download(
        self,
        client,
        url,
        s_pos,
        e_pos,
        shard_path,
        thread_id,
        message_queue: Queue,
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

        message_byte_cnt = 0
        nbytes_per_message = 1024 * 1024 * 1  # 1Mb

        def upload_progress_message():
            nonlocal message_byte_cnt, thread_id, nbytes_per_message, message_queue

            message_queue.put_nowait(("advance", message_byte_cnt, thread_id))
            message_byte_cnt = 0

        try:
            with client.stream(
                "GET",
                url,
                headers={
                    **self.headers,
                    "Range": f"bytes={s_pos}-{e_pos}",
                },
            ) as r:
                for chunk in r.iter_bytes(chunk_size=self.chunk_size_download):
                    # when status_code != 206, chunk will contain the error message
                    # the GET request is spoiled, give up and retry
                    if r.status_code != 206:
                        raise Exception(f"r.status_code: {r.status_code} r.content: {r.content}")

                    buffer.write(chunk)
                    message_byte_cnt += len(chunk)
                    if message_byte_cnt >= nbytes_per_message:
                        upload_progress_message()
                upload_progress_message()
        except Exception as e:
            # when using multiple threads, the exception often happens at 0
            logger.error(f"=> Thread {thread_id} failed: {e} Pos: {buffer.tell()}")

        assert buffer.tell() == e_pos + 1

    def get_cache_files(self, path):
        dirname, filename = osp.dirname(path), osp.basename(path)
        cache_pattern = osp.join(dirname, f".{filename}.*")
        filenames = [_.strip() for _ in run_cmd(f"ls {cache_pattern}").stdout.split("\n") if _.strip() != ""]
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

    def get_shard_path(self, path, i, s_pos, e_pos):
        file_dir, filename = osp.dirname(path), osp.basename(path)
        return osp.join(file_dir, f".{filename}.part{s_pos}-{e_pos}")

    def clear_message(self):
        q = self.message_queue
        while not q.empty():
            q.get_nowait()

    def download(self, url, path):
        # push all message to message queue by default
        # only deal with message queue in main thread when verbose > 0
        # when using MultiThreadDownloader as part of multi-process, set verbose=0

        # don't create client in self to make class picklable
        client = httpx.Client(
            headers=self.headers,
            timeout=self.timeout,
            proxies=self.proxy,
            follow_redirects=True,
        )

        filesize = self.get_filesize(client, url)

        self.message_queue.put_nowait(("filesize", filesize))

        file_chunk_size = (filesize + self.num_threads - 1) // self.num_threads
        ranges = [(i * file_chunk_size, min((i + 1) * file_chunk_size - 1, filesize - 1)) for i in range(self.num_threads)]
        if not self.is_support_range(client, url):
            print("=> Server does not support range, use single thread download")
            super().download(url, path)
            return

        # enter this step only when decided to use multi-thread download
        self.resolve_download_meta(url, path, filesize)

        progress: Progress = get_rich_progress_download() if self.verbose > 0 else nullcontext()
        if self.verbose >= 1:
            progress.add_task("Total", total=filesize)
        if self.verbose == 2:
            for i in range(self.num_threads):
                progress.add_task(f"Thread {i}", total=ranges[i][1] - ranges[i][0])

        with progress:
            self.gather_for_resume(
                path=path,
                ranges=ranges,
                message_queue=self.message_queue,
            )

            executor = ThreadPoolExecutor(max_workers=self.num_threads)
            not_done = []
            for i, (s_pos, e_pos) in enumerate(ranges):
                shard_path = self.get_shard_path(path, i, s_pos, e_pos)
                # use a message queue here
                # https://github.com/EleutherAI/tqdm-multiprocess/blob/master/tqdm_multiprocess/std.py

                future = executor.submit(
                    retry_wrapper(self.max_retries)(self.range_download),
                    client=client,
                    url=url,
                    s_pos=s_pos,
                    e_pos=e_pos,
                    shard_path=shard_path,
                    thread_id=i,
                    message_queue=self.message_queue,
                )
                not_done.append(future)

            while len(not_done) > 0:
                done, not_done = wait(
                    not_done,
                    return_when="FIRST_COMPLETED",
                    timeout=0.5,
                )  # this decides the frequency of message queue processing

                # process message queue for verbose > 0
                while True:
                    if self.verbose == 0:
                        break

                    message_cnt = self.message_queue.qsize()
                    if message_cnt > 10000:
                        print(f"=> Flooded! Message queue size: {message_cnt}")

                    try:
                        message = self.message_queue.get_nowait()
                        if message[0] != "advance":
                            continue
                        message_byte_cnt, thread_id = message[1], message[2]

                        if self.verbose >= 1:
                            progress.update(0, advance=message_byte_cnt)
                        if self.verbose == 2:
                            progress.update(thread_id + 1, advance=message_byte_cnt)
                    except (EmptyQueue, InterruptedError):
                        break
                if self.verbose > 0:
                    progress.refresh()

            shard_paths = [self.get_shard_path(path, i, s_pos, e_pos) for i, (s_pos, e_pos) in enumerate(ranges)]
            run_cmd(f"cat {' '.join(shard_paths)} > {path}", async_cmd=False)

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
    def download_wget(cls, url, out=None, headers=None, proxy=None, timeout=5, retries=3, verbose=True):
        if out == "auto":
            out = cls.get_output_path(url)

        if proxy == "auto":
            proxy = "127.0.0.1:8091"

        wget_args = ""
        if proxy:
            wget_args += f" --proxy=on --proxy http://{proxy}"

        wget_args += f" --tries {retries} --timeout {timeout} --no-check-certificate --continue"

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
