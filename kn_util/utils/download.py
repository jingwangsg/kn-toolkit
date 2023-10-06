import asyncio
import httpx
import requests
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from huggingface_hub.utils._headers import build_hf_headers
from transformers.utils.hub import http_user_agent
from contextlib import nullcontext

import nest_asyncio

nest_asyncio.apply()


def head_with_redirects(url, verbose=False, headers=None):
    with httpx.Client(follow_redirects=False) as client:
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
    async def _async_range_download(url, save_name, s_pos, e_pos, client, chunk_size, headers=None, pbar=None):
        range_headers = {"Range": f"bytes={s_pos}-{e_pos}"}
        if headers:
            range_headers.update(headers)
        headers = range_headers

        f = open(save_name, "rb+")
        async with client.stream('GET', url=url, headers=headers) as r:
            f.seek(s_pos)
            async for chunk in r.aiter_bytes():
                if chunk:  # prevent keep-alive chunks
                    f.write(chunk)
                    if pbar:
                        pbar.update(len(chunk))
        f.close()

    @staticmethod
    def _calc_divisional_range(filesize, chuck=10):
        step = filesize // chuck
        arr = list(range(0, filesize, step))
        result = []
        for i in range(len(arr) - 1):
            s_pos, e_pos = arr[i], arr[i + 1] - 1
            result.append([s_pos, e_pos])
        result[-1][-1] = filesize - 1
        return result

    @classmethod
    def async_sharded_download(cls,
                               url,
                               save_name=None,
                               chunk_size=1024 * 100,
                               num_shards=10,
                               headers=None,
                               verbose=True):
        if save_name is None:
            save_name = url.split("/")[-1]

        # resolve redirect
        res = head_with_redirects(url, headers=headers)

        if res.headers.get("Accept-Ranges", None) != "bytes":
            print("File does not support range download, use direct download")
            cls.download(url, save_name, headers=headers, verbose=verbose)
            return

        # get filesize
        url = res.url
        filesize = int(res.headers["Content-Length"])
        divisional_ranges = cls._calc_divisional_range(filesize, num_shards)

        with open(save_name, "wb") as f:
            pass

        transport = httpx.AsyncHTTPTransport(retries=5)
        client = httpx.AsyncClient(transport=transport, timeout=60)
        loop = asyncio.get_event_loop()

        pbar = tqdm_asyncio(total=filesize,
                            dynamic_ncols=True,
                            desc=f"Downloading {save_name}",
                            unit="B",
                            unit_scale=True,
                            smoothing=0.1,
                            miniters=1,
                            ascii=True) if verbose else None
        context = pbar if verbose else nullcontext()

        with context:
            tasks = [
                cls._async_range_download(url,
                                          save_name,
                                          s_pos,
                                          e_pos,
                                          chunk_size=chunk_size,
                                          headers=headers,
                                          client=client,
                                          pbar=pbar) for s_pos, e_pos in divisional_ranges
            ]
            loop.run_until_complete(asyncio.wait(tasks))

    @classmethod
    def download(cls, url, chunk_size=1024 * 100, save_name=None, headers=None, verbose=True):
        if save_name is None:
            save_name = url.split("/")[-1]

        # resolve redirect
        res = get_head_with_redirects(url, headers=headers)
        url = res.url
        filesize = int(res.headers["Content-Length"])

        pbar = tqdm(total=filesize,
                    dynamic_ncols=True,
                    desc=f"Downloading {save_name}",
                    unit="B",
                    unit_scale=True,
                    smoothing=0.1,
                    miniters=1,
                    ascii=True) if verbose else None

        context = pbar if verbose else nullcontext()

        with context:
            client = httpx.Client()
            with client.stream('GET', url=url, headers=headers) as r:
                with open(save_name, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
