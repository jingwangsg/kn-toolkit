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
from threading import Semaphore
from multiprocessing.pool import ThreadPool as ThreadPoolVanilla
from queue import Queue
import threading


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


def map_async_with_shard(
    iterable,
    func=lambda x: x,
    loader=lambda x: x,
    num_process=32,
    num_thread=32,
    max_semaphore=32,
    max_retries=5,
    shard_size=1000,
    verbose=True,
    is_ready=lambda x: True,
):
    """
    producer-consumer model, split iterable to shards.
    For each shard, use thread to load data and process data iteratively
    (Here, func should be compute-bound, and loader should be io-bound)

    Q1: Why only split iterable to shards when using process pool?
    A1: Frequently switching between processes is much more expensive than switching between threads.
    """
    iterable = list(iterable)  # make sure it is shardable

    def _process_shard(shard):
        semaphore = Semaphore(max_semaphore)
        task_queue = Queue()
        failed_queue = Queue()

        for item in shard:
            task_queue.put((0, item))

        def locked_iterable():
            while not task_queue.empty():
                semaphore.acquire()
                yield task_queue.get()

        def wrapped_loader(*args, **kwargs):
            try:
                return True, item, loader(*args, **kwargs)
            except Exception as e:
                return False, item, None
        
        def deal_with_error(retry_cnt):
            if retry_cnt >= max_retries:
                failed_queue.put(item)
            else:
                task_queue.put((retry_cnt + 1, item))
            semaphore.release()
        
        with ThreadPoolVanilla(num_thread) as thread_pool:
            for retry_cnt, success, item, output in thread_pool.imap_unordered(
                    lambda x: (x[0], *wrapped_loader(x[1])),
                    locked_iterable(),
            ):
                if not success:
                    deal_with_error(retry_cnt)
                    continue

                try:
                    ret = func(output)
                except:
                    deal_with_error(retry_cnt)
                    continue

                if not is_ready(ret):
                    deal_with_error(retry_cnt)
                    continue
                    
                semaphore.release()
        
        return failed_queue.qsize()

    iterable_shards = [iterable[i:i + shard_size] for shard_idx, i in enumerate(range(0, len(iterable), shard_size))]
    if verbose:
        print(f"Total {len(iterable_shards)} shards")

    map_async(
        iterable=iterable_shards,
        func=_process_shard,
        num_process=num_process,
        verbose=verbose,
    )
