import asyncio
import httpx
import requests
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from huggingface_hub.utils._headers import build_hf_headers
from transformers.utils.hub import http_user_agent
from contextlib import nullcontext
import io
from functools import partial

import nest_asyncio

nest_asyncio.apply()


def head_with_redirects(url, verbose=False, headers=None):
    with httpx.Client(follow_redirects=False, timeout=None) as client:
        response = client.head(url, headers=headers)
        while response.status_code in (301, 302):
            if verbose:
                print("Redirected to:", response.headers['Location'])
            response = client.head(response.headers['Location'])
        return response


# async download
def get_headers(from_hf=False):
    # only for huggingface
    user_agent_header = http_user_agent()
    if from_hf:
        headers = build_hf_headers(user_agent=user_agent_header, token="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")
    else:
        headers = {"User-Agent": user_agent_header}
    return headers


class Downloader:

    @classmethod
    async def _async_range_download(
        cls,
        url,
        s_pos,
        e_pos,
        client,
        chunk_size,
        out=None,
        headers=None,
        pbar=None,
        max_retries=5,
    ):
        #! should initiate separate file handler for each coroutine, or speed will be slow
        to_buffer = (out is None)

        headers = headers or {}

        if to_buffer:
            buffer = io.BytesIO()
            buffer.seek(0)
        else:
            buffer = open(out, "rb+")
            buffer.seek(s_pos)

        retries = 0
        written_bytes = 0

        while retries < max_retries:
            try:
                headers["Range"] = f"bytes={s_pos + written_bytes}-{e_pos}"
                async with client.stream('GET', url=url, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        if chunk:  # prevent keep-alive chunks
                            chunk_size = len(chunk)
                            buffer.write(chunk)
                            written_bytes += chunk_size
                            if pbar:
                                pbar.update(chunk_size)
                break  # Break the retry loop if download is successful
            except httpx.NetworkError:
                retries += 1
                if retries < max_retries:
                    print(
                        f"=> Network error at {written_bytes}/{e_pos-s_pos + 1}, retrying ({retries}/{max_retries})...")
                else:
                    print("=> Max retries reached. Download failed.")
                    if not to_buffer:
                        buffer.close()
                    raise

        if not to_buffer:
            buffer.close()
        else:
            return buffer

    @staticmethod
    def _calc_divisional_range(filesize, chuck=10):
        step = filesize // chuck
        ranges = [[i, i + step - 1] for i in range(0, filesize, step)]
        ranges[-1][-1] = filesize - 1
        return ranges

    @classmethod
    def async_sharded_download(cls,
                               url,
                               out=None,
                               chunk_size=1024 * 100,
                               num_shards=10,
                               headers=None,
                               proxy=None,
                               verbose=True):
        to_buffer = (out is None)

        if out == "_AUTO":
            out = url.split("/")[-1]

        if not to_buffer:
            with open(out, "wb"):
                pass
        # resolve redirect
        res = head_with_redirects(url, headers=headers)

        if res.headers.get("Accept-Ranges", None) != "bytes":
            print("File does not support range download, use direct download")
            cls.download(url, out=out, headers=headers, proxy=proxy, verbose=verbose)
            return

        # get filesize
        url = res.url
        filesize = int(res.headers["Content-Length"])
        divisional_ranges = cls._calc_divisional_range(filesize, num_shards)

        transport = httpx.AsyncHTTPTransport(retries=5)
        proxy = httpx.Proxy(url=f"http://{proxy}") if proxy else None
        client = httpx.AsyncClient(transport=transport, timeout=None, proxies=proxy)

        pbar = tqdm_asyncio(total=filesize,
                            dynamic_ncols=True,
                            desc=f"Downloading",
                            unit="B",
                            unit_scale=True,
                            smoothing=0.1,
                            miniters=1,
                            ascii=True) if verbose else None
        context = pbar if verbose else nullcontext()

        loop = asyncio.get_event_loop()

        async def download_shard(s_pos, e_pos):
            return await cls._async_range_download(url=url,
                                                   s_pos=s_pos,
                                                   e_pos=e_pos,
                                                   client=client,
                                                   chunk_size=chunk_size,
                                                   out=out,
                                                   headers=headers,
                                                   pbar=pbar)

        with context:
            # https://zhuanlan.zhihu.com/p/575243634
            tasks = asyncio.gather(*[download_shard(s_pos, e_pos) for (s_pos, e_pos) in divisional_ranges])
            result = loop.run_until_complete(tasks)

        # loop.close()

        if to_buffer:
            ret_buffer = io.BytesIO()
            for buffer in result:
                ret_buffer.write(buffer.getvalue())

            return ret_buffer

    @classmethod
    def download(cls, url, out=None, chunk_size=1024 * 100, headers=None, proxy=None, verbose=True):
        if out == "_AUTO":
            out = url.split("/")[-1]

        to_buffer = (out is None)

        # resolve redirect
        res = head_with_redirects(url, headers=headers)
        url = res.url
        filesize = int(res.headers["Content-Length"])

        pbar = tqdm(total=filesize,
                    dynamic_ncols=True,
                    desc=f"Downloading",
                    unit="B",
                    unit_scale=True,
                    smoothing=0.1,
                    miniters=1,
                    ascii=True) if verbose else None

        context = pbar if verbose else nullcontext()

        buffer = io.BytesIO() if to_buffer else open(out, "wb")

        with context:
            proxy = httpx.Proxy(url=f"http://{proxy}") if proxy else None
            client = httpx.Client(timeout=None, proxies=proxy)
            with client.stream('GET', url=url, headers=headers) as r:
                for chunk in r.iter_bytes(chunk_size=chunk_size):
                    if chunk:
                        buffer.write(chunk)
                        if pbar:
                            pbar.update(len(chunk))

        if not to_buffer:
            buffer.close()
        else:
            return buffer
