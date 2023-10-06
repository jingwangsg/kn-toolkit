import asyncio
import httpx
import requests
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from huggingface_hub.utils._headers import build_hf_headers
from transformers.utils.hub import http_user_agent
from contextlib import nullcontext
import io

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

    @staticmethod
    async def _async_range_download(url, s_pos, e_pos, client, chunk_size, out=None, headers=None, pbar=None):
        #! should initiate separate file handler for each coroutine, or speed will be slow
        to_buffer = (out is None)

        range_headers = {"Range": f"bytes={s_pos}-{e_pos}"}
        if headers:
            range_headers.update(headers)
        headers = range_headers

        if to_buffer:
            buffer = io.BytesIO()
            buffer.seek(0)
        else:
            buffer = open(out, "rb+")
            buffer.seek(s_pos)

        async with client.stream('GET', url=url, headers=headers) as r:
            async for chunk in r.aiter_bytes():
                if chunk:  # prevent keep-alive chunks
                    chunk_size = len(chunk)
                    buffer.write(chunk)
                    if pbar:
                        pbar.update(chunk_size)

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
    def async_sharded_download(cls, url, out=None, chunk_size=1024 * 100, num_shards=10, headers=None, verbose=True):
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
            cls.download(url, out=out, headers=headers, verbose=verbose)
            return

        # get filesize
        url = res.url
        filesize = int(res.headers["Content-Length"])
        divisional_ranges = cls._calc_divisional_range(filesize, num_shards)

        transport = httpx.AsyncHTTPTransport(retries=5)
        client = httpx.AsyncClient(transport=transport, timeout=None)

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

        with context:
            tasks = [
                cls._async_range_download(url,
                                          s_pos,
                                          e_pos,
                                          out=out,
                                          chunk_size=chunk_size,
                                          headers=headers,
                                          client=client,
                                          pbar=pbar) for s_pos, e_pos in divisional_ranges
            ]

            # https://zhuanlan.zhihu.com/p/575243634
            tasks = asyncio.gather(*tasks)
            result = loop.run_until_complete(tasks)

        loop.close()

        if to_buffer:
            ret_buffer = io.BytesIO()
            for buffer in result:
                ret_buffer.write(buffer.getvalue())

            return ret_buffer

    @classmethod
    def download(cls, url, out=None, chunk_size=1024 * 100, headers=None, verbose=True):
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
            client = httpx.Client(timeout=None)
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
