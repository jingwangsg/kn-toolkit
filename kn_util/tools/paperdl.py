from fire import Fire
from googlesearch import search

from kn_util.utils.download import MultiThreadDownloader
from kn_util.utils.multiproc import map_async_with_thread

# progress = get_rich_progress_mofn()
# progress.start()
# progress.add_task("Downloading", total=1)


def url_finder(results, match_func):
    for result in results:
        if match_func(result):
            return result
    return None


def arxiv_finder(results):
    arxiv_url = url_finder(results, lambda x: "arxiv.org/abs" in x)
    if arxiv_url is None:
        return None

    pdf_url = arxiv_url.replace("abs", "pdf") + ".pdf"
    return pdf_url


def cvf_finder(results):
    cvf_url = url_finder(results, lambda x: "openaccess.thecvf" in x and "pdf" in x)
    if cvf_url is None:
        return None

    return cvf_url


def openreview_finder(results):
    openreview_url = url_finder(results, lambda x: "openreview.net" in x)
    if openreview_url is None:
        return None

    openreview_url = openreview_url.replace("forum", "pdf")
    return openreview_url


def download_paper(title):
    results = search(title, num_results=5, sleep_interval=0.5)
    results = list(results)
    url = None
    if res := arxiv_finder(results):
        url = res
    elif res := openreview_finder(results):
        url = res
    elif res := cvf_finder(results):
        url = res
    else:
        url = None
    return url


def _main(file_path):
    with open(file_path) as f:
        papers = f.readlines()
    papers = [paper.strip() for paper in papers if paper.strip() != ""]
    urls = map_async_with_thread(
        iterable=papers,
        func=download_paper,
        num_thread=8,
        desc="Getting URLs",
    )
    downloader = MultiThreadDownloader(num_threads=4, max_retries=10)
    for idx, url in enumerate(urls):
        downloader.download(url, f"{idx:03d}.pdf")


def main():
    Fire(_main)
