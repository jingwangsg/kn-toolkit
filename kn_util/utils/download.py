from kn_util.utils.logger import setup_logger_loguru
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
import socket
from io import BytesIO
import copy

from ..utils.system import run_cmd, force_delete, clear_process
from ..utils.io import save_json, load_json
from ..utils.rich import get_rich_progress_download, add_tasks
from ..utils.system import buffer_keep_open

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


# def is_proxy_valid(proxy):
#     logger.info(f"=> Testing proxy: {proxy}")
#     # r = httpx.get("https://www.google.com", proxy=proxy)

#     from pudb.remote import set_trace; set_trace()
#     try:
#         r = httpx.get("https://www.google.com", proxy=proxy)
#         return r.status_code == 200
#     except httpx.ConnectError as e:
#         return False
#     except Exception as e:
#         logger.error(f"=> Proxy test failed: {e}")
#         return False


def is_proxy_valid(
    proxy,
    timeout=None,
):
    try:
        _, address = proxy.split("//")
        hostname, port = address.split(":")
        port = int(port)
    except ValueError as e:
        logger.error(f"parse proxy failed: {e}")
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((hostname, port))
            return True
        except socket.error as e:
            logger.error(f"connect to proxy failed: {e}")
            return False


def retry_wrapper(max_retries=10, detect_proxy=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            thread_id = kwargs.get("thread_id", None)
            retry_cnt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.info(f"=> Thread {thread_id} retry {retry_cnt+1}/{max_retries} failed: {e}")

                    client = kwargs.get("client", None)
                    if detect_proxy and client is not None:

                        # check if proxy is valid
                        if len(client._mounts) > 0:
                            proxy_url = list(client._mounts.values())[0]._pool._proxy_url
                            proxy = f"{proxy_url.scheme.decode('utf-8')}://{proxy_url.host.decode('utf-8')}:{proxy_url.port}"
                            if not is_proxy_valid(proxy):
                                client._mounts = {}  # a hack to disable proxy
                                logger.info(f"=> Proxy {proxy} is invalid, disable it")

                    retry_cnt += 1
                    if max_retries is not None and retry_cnt >= max_retries:
                        break

            raise Exception(f"=> Thread {thread_id} retry {max_retries} times, still failed")

        return wrapper

    return decorator


def get_hf_headers():
    user_agent_header = http_user_agent()
    headers = build_hf_headers(user_agent=user_agent_header)
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
        if verbose:
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
        file_headers = client.head(url, follow_redirects=True).headers
        return file_headers

    @retry_wrapper(max_retries=3)
    def get_filesize(self, client, url):
        file_headers = self.get_file_headers(client=client, url=url)
        if HUGGINGFACE_HEADER_X_LINKED_SIZE in file_headers:
            return int(file_headers[HUGGINGFACE_HEADER_X_LINKED_SIZE])

        if "Content-Length" in file_headers:
            return int(file_headers["Content-Length"])

        return None

    def download(self, url, path):

        client = httpx.Client(
            headers=self.headers,
            timeout=self.timeout,
            proxy=self.proxy,
        )

        filesize = self.get_filesize(client=client, url=url)
        filedir, filename = osp.dirname(path), osp.basename(path)
        f = open(path, "wb+")

        progress = get_rich_progress_download(disable=(self.verbose == 0))
        progress.start()
        task_id = progress.add_task(filename, total=filesize)
        # f.seek(0, os.SEEK_END)
        # not really necessary, if it is resumable, it can be handled by multi-thread downloader
        # progress.update(task_id, completed=f.tell())

        try:
            with client.stream("GET", url) as r:
                for chunk in r.iter_bytes(chunk_size=self.chunk_size_download):
                    if r.status_code != 200:
                        raise Exception(f"r.status_code: {r.status_code} r.content: {r.content}")
                    if chunk:
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
        except Exception as e:
            logger.error(f"=> Download failed: {e} Pos: {f.tell()}")

        progress.stop()


class MultiThreadDownloader(Downloader):
    def __init__(
        self,
        headers={},
        num_threads=4,
        max_retries=10,
        timeout=10,
        disable_multithread=False,
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
        self.disable_multithread = disable_multithread

    @retry_wrapper(max_retries=3)
    def is_support_range(self, client, url):
        # headers = self.get_file_headers(client, url)
        # return "Accept-Ranges" in headers and headers["Accept-Ranges"] == "bytes"
        # download 1 bytes to check if server supports range

        with client.stream(
            "GET",
            url,
            headers={**self.headers, "Range": "bytes=0-0"},
        ) as r:
            if r.status_code in [503, 504]:
                raise Exception(f"r.status_code: {r.status_code} r.content: {r.content}")

            return r.status_code == 206

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
        client: httpx.Client,
        url,
        s_pos,
        e_pos,
        shard_path,
        thread_id,
        message_queue: Queue,
    ):
        # if not self.is_proxy_valid(self.proxy):
        #     client._mounts = {} # a hack to disable proxy

        if osp.exists(shard_path):
            buffer = open(shard_path, "rb+")
        else:
            buffer = open(shard_path, "wb+")

        s_pos_dl = s_pos
        buffer.seek(0, os.SEEK_END)
        skip_bytes = buffer.tell()
        s_pos_dl += skip_bytes

        # if skip_bytes > 0:
        #     logger.info(f"=> Thread {thread_id} resume from {skip_bytes} bytes")

        if s_pos_dl == e_pos + 1:
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
                    "Range": f"bytes={s_pos_dl}-{e_pos}",
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
            logger.error(f"=> thread {thread_id} failed: {e} Pos: {buffer.tell()}")

        # print(f" {buffer.tell()}, {e_pos - s_pos + 1} {e_pos}")
        assert buffer.tell() == e_pos - s_pos + 1

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
            from pudb.remote import set_trace

            set_trace()
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
            proxies=self.proxy,  # for backward compatibility
            follow_redirects=True,
        )

        if self.disable_multithread:
            print("=> Multi-thread download disabled")
            super().download(url, path)
            return

        filesize = self.get_filesize(client=client, url=url)

        self.message_queue.put_nowait(("filesize", filesize))

        file_chunk_size = (filesize + self.num_threads - 1) // self.num_threads
        ranges = [(i * file_chunk_size, min((i + 1) * file_chunk_size - 1, filesize - 1)) for i in range(self.num_threads)]
        if not self.is_support_range(client=client, url=url):
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
            dirname, filename = osp.dirname(path), osp.basename(path)
            cmds = "&&".join(
                [
                    f"cat {' '.join(shard_paths)} > {path}",
                    f"rm -rf {osp.join(dirname, f'.{filename}.*')}",
                ]
            )
            run_cmd(cmds)

            # self.clear_cache(path)


class MultiThreadDownloaderInMem(MultiThreadDownloader):
    """Multi-threaded Downloader without resume, downloading files into memory"""

    def range_download(self, client, url, s_pos, e_pos):
        buffer = BytesIO()
        with (
            buffer_keep_open(buffer),
            client.stream(
                "GET",
                url,
                headers={
                    **self.headers,
                    "Range": f"bytes={s_pos}-{e_pos}",
                },
            ) as r,
        ):
            for chunk in r.iter_bytes(chunk_size=self.chunk_size_download):
                buffer.write(chunk)
            byte_values = buffer.getvalue()

        return byte_values

    def _direct_download(self, client, url):
        buffer = BytesIO()
        with buffer_keep_open(buffer), client.stream("GET", url) as r:
            for chunk in r.iter_bytes(chunk_size=self.chunk_size_download):
                buffer.write(chunk)
            byte_values = buffer.getvalue()

        return byte_values

    def download(self, url):
        client = httpx.Client(
            headers=self.headers,
            timeout=self.timeout,
            proxies=self.proxy,
            follow_redirects=True,
        )
        if self.disable_multithread:
            return self._direct_download(client, url)

        filesize = self.get_filesize(client=client, url=url)
        if filesize is None:
            # head is not allowed or Content Length not available
            return self._direct_download(client, url)

        file_chunk_size = (filesize + self.num_threads - 1) // self.num_threads
        ranges = [(i * file_chunk_size, min((i + 1) * file_chunk_size - 1, filesize - 1)) for i in range(self.num_threads)]

        if not self.is_support_range(client=client, url=url):
            return self._direct_download(client, url)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for i, (s_pos, e_pos) in enumerate(ranges):
                future = executor.submit(
                    self.range_download,
                    client=client,
                    url=url,
                    s_pos=s_pos,
                    e_pos=e_pos,
                )
                futures.append(future)

        byte_values = b"".join([future.result() for future in futures])

        return byte_values


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


import aiohttp
import asyncio


class CoroutineDownloader(Downloader):
    def __init__(self, *args, queue=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_queue = Queue() if queue is None else queue

    async def is_support_range(self, session, url):
        async with session.get(url, headers={**self.headers, "Range": "bytes=0-0"}) as r:
            if r.status in [503, 504]:
                raise Exception(f"r.status_code: {r.status_code} r.content: {r.content}")

            return r.status == 206
    
    def clear_message(self):
        q = self.message_queue
        while not q.empty():
            q.get_nowait()
    
    async def get_filesize(self, session, url):
        async with session.head(url) as response:
            if HUGGINGFACE_HEADER_X_LINKED_SIZE in response.headers:
                return int(response.headers[HUGGINGFACE_HEADER_X_LINKED_SIZE])

            if "Content-Length" in response.headers:
                return int(response.headers["Content-Length"])

            return None
    
    

    async def _download(self, url, path):
        progress = get_rich_progress_download(disable=(self.verbose == 0))
        progress.start()

        async with aiohttp.ClientSession(headers=self.headers) as session:
            filesize = await self.get_filesize(session, url)
            self.message_queue.put_nowait(("filesize", filesize))

            is_support_range = await retry_wrapper(self.max_retries)(self.is_support_range)(session, url)

            progress.add_task("Total", total=filesize)

            if osp.exists(path):
                buffer = open(path, "rb+")
            else:
                buffer = open(path, "wb+")

            if is_support_range and osp.exists(path):
                buffer.seek(0, os.SEEK_END)
                pos = buffer.tell()
            else:
                buffer.seek(0, os.SEEK_SET)
                pos = 0

            if pos == filesize:
                progress.stop()
                return

            headers = copy.deepcopy(self.headers)
            if is_support_range and pos > 0:
                headers["Range"] = f"bytes={pos}-{filesize}"
                progress.update(0, advance=pos)
                self.message_queue.put_nowait(("advance", pos))

            update_queue_interval = 1024 * 1024 * 10  # 1Mb
            update_queue_nbytes = 0

            async with session.get(url, headers=headers) as response:
                if response.status in [200, 206]:
                    # Open the file in binary write mode
                    while True:
                        chunk = await response.content.read(1024)
                        progress.update(0, advance=len(chunk))
                        update_queue_nbytes += len(chunk)
                        if update_queue_nbytes >= update_queue_interval or not chunk:
                            self.message_queue.put_nowait(("advance", update_queue_nbytes))
                            update_queue_nbytes = 0
                        if not chunk:
                            break
                        buffer.write(chunk)
                else:
                    raise Exception(f"Failed to download {url}, status code: {response.status}, content: {response.content}")
        progress.stop()

    def download(self, url, path):
        func = retry_wrapper(max_retries=self.max_retries)(self._download)(url, path)
        asyncio.run(func)


class CoroutineDownloaderInMem(CoroutineDownloader):
    async def _download(self, url):
        progress = get_rich_progress_download(disable=(self.verbose == 0))
        progress.start()

        async with aiohttp.ClientSession(headers=self.headers) as session:
            filesize = await self.get_filesize(session, url)

            progress.add_task("Total", total=filesize)
            buffer = BytesIO()

            async with session.get(url, headers=self.headers) as response:
                if response.status in [200, 206]:
                    # Open the file in binary write mode
                    while True:
                        chunk = await response.content.read(1024)
                        progress.update(0, advance=len(chunk))
                        if not chunk:
                            break
                        buffer.write(chunk)
                else:
                    raise Exception(f"Failed to download {url}, status code: {response.status}, content: {response.content}")
        progress.stop()

        return buffer.getvalue()

    def download(self, url):
        func = retry_wrapper(max_retries=self.max_retries)(self._download)(url)
        return asyncio.run(func)
