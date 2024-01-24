import time
from pathos.multiprocessing import Pool

from pathos.helpers import mp
from pathos.threading import ThreadPool
from pathos.pools import ProcessPool
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext


def _run_sequential(iterable, func, desc=""):
    pbar = tqdm(total=len(iterable), desc=desc)
    ret = []
    for it in iterable:
        ret.append(func(it))
        pbar.update(1)
    pbar.close()
    return ret


def map_async_with_coroutine(iterable, func, desc="", wrap_func=True):
    pbar = tqdm(total=len(iterable), desc=desc)

    async def func_async(item):
        loop = asyncio.get_running_loop()
        if wrap_func:
            result = await loop.run_in_executor(None, func, item)  # 在默认执行器中运行非异步函数
        else:
            result = await func(item)
        pbar.update(1)
        return result

    async def _map_async_with_coroutine(iterable):
        tasks = [func_async(item) for item in iterable]
        results = await asyncio.gather(*tasks)
        pbar.close()
        return results

    return asyncio.run(_map_async_with_coroutine(iterable))


def map_async(
    iterable,
    func,
    num_process=30,
    chunksize=1,
    desc: object = "",
    test_flag=False,
    verbose=True,
):
    """while test_flag=True, run sequentially"""
    if test_flag:
        return _run_sequential(iterable, func, desc=desc)
    else:
        p = Pool(num_process)
        # ret = []
        # for it in tqdm(iterable, desc=desc):
        #     ret.append(p.apply_async(func, args=(it,)))
        ret = p.map_async(
            func=func,
            iterable=iterable,
            chunksize=chunksize,
        )
        total = ret._number_left

        pbar = tqdm(total=total, desc=desc) if verbose else None

        while ret._number_left > 0:
            if pbar:
                pbar.n = total - ret._number_left
                pbar.refresh()
            time.sleep(0.1)
        if pbar:
            p.close()

        return ret.get()


def map_async_with_thread(
    iterable,
    func,
    num_thread=30,
    desc="",
    verbose=True,
    test_flag=False,
):
    if test_flag:
        return _run_sequential(iterable, func, desc=desc)

    with ThreadPoolExecutor(num_thread) as executor:
        pbar = tqdm(total=len(iterable), desc=desc) if verbose else None
        context = pbar if pbar else nullcontext()

        results = []

        with context:
            futures = {executor.submit(func, x): x for x in iterable}

            for future in as_completed(futures):
                if pbar:
                    pbar.update(1)
                try:
                    result = future.result()  # Get the result from the future
                    results.append(result)  # Append the result to the results list
                except Exception as e:
                    # If there is an exception, you can handle it here
                    # For now, we'll just print it
                    print(f"Exception in thread: {e}")

        return results


def map_async_with_tolerance(iterable, func, num_workers=32, level="thread", is_ready=lambda x: x):

    if level == "thread":
        p = ThreadPool(num_workers)
    elif level == "process":
        p = ProcessPool(num_workers)
    p.restart()

    data_queue = mp.Queue()
    for x in iterable:
        data_queue.put(x)

    running = []

    total = data_queue.qsize()
    pbar = tqdm(total=total)

    while not (data_queue.empty() and len(running) == 0):
        if not data_queue.empty():
            cur_item = data_queue.get()
            cur_thread = p.apipe(func, cur_item)
            running.append(cur_thread)

        # update running processes whose state in unready
        new_running = []
        for item in running:
            if not item.ready():
                new_running.append(item)
            elif is_ready(item.get()):
                pbar.n = pbar.n + 1
                pbar.refresh()
                time.sleep(0.1)
        running.clear()
        del running
        running = new_running

    p.close()
