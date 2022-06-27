from tqdm import tqdm
import time
from pathos.multiprocessing import Pool


def map_async(iterable, func, num_process=30, desc: object = ""):
    p = Pool(num_process)
    # ret = []
    # for it in tqdm(iterable, desc=desc):
    #     ret.append(p.apply_async(func, args=(it,)))
    ret = p.map_async(func=func, iterable=iterable)
    total = ret._number_left
    pbar = tqdm(total=total, desc=desc)
    while ret._number_left > 0:
        pbar.n = total - ret._number_left
        pbar.refresh()
        time.sleep(0.1)
    p.close()

    return ret.get()
