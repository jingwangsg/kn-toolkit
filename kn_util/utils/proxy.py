import os.path as osp
import random

import requests
from loguru import logger

from kn_util.utils.multiproc import map_async_with_thread

HTTPS_PROXY_URLS = [
    "https://raw.githubusercontent.com/zloi-user/hideip.me/main/https.txt",
]
HTTP_PROXY_URLS = [
    "https://raw.githubusercontent.com/zloi-user/hideip.me/main/http.txt",
]


HTTP_PROXY_PARSERS = [
    lambda x: [_.rsplit(":", 1)[0] for _ in x.decode("unicode_escape").splitlines()],
]
HTTPS_PROXY_PARSERS = [
    lambda x: [_.rsplit(":", 1)[0] for _ in x.decode("unicode_escape").splitlines()],
]


class ProxyPool:
    def __init__(
        self,
        proxy_urls=HTTPS_PROXY_URLS,
        proxy_parsers=HTTPS_PROXY_PARSERS,
        target_url="www.baidu.com",
        domain="https",
        include_file=None,
        exclude_file=None,
    ):
        assert not target_url.startswith("http") and not target_url.startswith("https"), "Please remove http or https from target_url"

        self.proxy_urls = proxy_urls
        self.proxy_parsers = proxy_parsers
        self.target_url = target_url
        self.domain = domain
        self.failed_proxies = set()

        if exclude_file and osp.exists(exclude_file):
            with open(exclude_file, "r") as f:
                self.failed_proxies = set(f.read().splitlines())

        if include_file and osp.exists(include_file):
            with open(include_file, "r") as f:
                self.proxies = f.read().splitlines()
        else:
            self.refresh_proxies()

    def refresh_proxies(self):
        self.proxies = []
        for i in range(len(self.proxy_urls)):
            try:
                response = requests.get(self.proxy_urls[i])
                proxies = self.proxy_parsers[i](response.content)
            except Exception as e:
                print(e)
            self.proxies.extend(proxies)

        self.validate_proxies()

    def validate_proxy(self, proxy, url):
        if proxy in self.failed_proxies:
            return False

        try:
            response = requests.get(
                url,
                proxies={self.domain: f"{self.domain}://{proxy}"},
                allow_redirects=True,
                verify=False,
                timeout=60,
            )

            if response.status_code == 200:
                print(f"Validated proxy: {proxy}")
                return True
        except Exception:
            # print(f"Failed to validate proxy: {proxy}")
            self.failed_proxies.add(proxy)

        self.failed_proxies.add(proxy)
        return False

    def validate_proxies(self, url=None):
        if url is None:
            url = self.target_url

        url = f"{self.domain}://{url}"

        validate_results = map_async_with_thread(
            iterable=self.proxies,
            func=lambda proxy: self.validate_proxy(proxy=proxy, url=url),
            num_thread=32,
        )

        self.proxies = [proxy for proxy, result in zip(self.proxies, validate_results) if result]
        logger.info(f"Validated {len(self.proxies)} proxies")

    def get_proxy(self, choice="random"):
        if choice == "random":
            proxy = random.choice(self.proxies)
        elif choice == "order":
            proxy = self.proxies[0]
        else:
            raise ValueError(f"Invalid choice: {choice}")

        # time_elapsed = time.time() - self.refresh_timestamp
        if len(self.proxies) < 500:
            self.refresh_proxies()
        return proxy

    def get(self, url):
        proxy = self.get_proxy()
        return requests.get(
            url,
            proxies={self.domain: f"{self.domain}://{proxy}"},
            allow_redirects=True,
            verify=False,
            timeout=60,
        )

    def remove(self, proxy):
        self.proxies.remove(proxy)
        self.failed_proxies.add(proxy)

    def update_exclude_file(self, exclude_file):
        with open(exclude_file, "w") as f:
            f.write("\n".join(self.failed_proxies))

    def update_include_file(self, include_file):
        with open(include_file, "w") as f:
            f.write("\n".join(self.proxies))

    def __len__(self):
        return len(self.proxies)
