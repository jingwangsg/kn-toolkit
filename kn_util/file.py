import json

# import joblib
import numpy as np
import dill
import pickle
import csv
import os
import h5py
from typing import Sequence, Mapping


def get_filename_from_absolute_path(url):
    return url.split("/")[-1]


def load_json(fn):
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, fn):
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def load_pickle(fn):
    # return joblib.load(fn)
    with open(fn, "rb") as f:
        obj = dill.load(f)
    return obj


def save_pickle(obj, fn):
    # return joblib.dump(obj, fn, protocol=pickle.HIGHEST_PROTOCOL)
    with open(fn, "wb") as f:
        dill.dump(obj, f, protocol=dill.HIGHEST_PROTOCOL)


def run_or_load(func, file_path, cache=True, overwrite=False, args=dict()):
    # def run_or_loader_wrapper(**kwargs):
    #     if os.path.exists()
    #     return fn(**kwargs)
    # if os.path.exists(file_path)
    if overwrite or not os.path.exists(file_path):
        obj = func(**args)
        if cache:
            save_pickle(obj, file_path)
    else:
        obj = load_pickle(obj)

    return obj


def load_csv(fn, delimiter=",", has_header=True):
    fr = open(fn, "r")
    read_csv = csv.reader(fr, delimiter=delimiter)

    ret_list = []

    for idx, x in enumerate(read_csv):
        if has_header and idx == 0:
            header = x
            continue
        if has_header:
            ret_list += [{k: v for k, v in zip(header, x)}]
        else:
            ret_list += [x]

    return ret_list

def save_hdf5(kv, cur_handler):
    """convenient saving hierarchical data recursively to hdf5"""
    if isinstance(kv, Sequence):
        kv = {str(idx): v for idx, v in enumerate(kv)}
    for k, v in kv.items():
        if isinstance(v, np.ndarray):
            cur_handler.create_dataset(k, data=v)
        elif  isinstance(v, Mapping):
            next_handler = cur_handler.create_group(k)
            save_hdf5(v, next_handler)