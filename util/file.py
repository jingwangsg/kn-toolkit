import json
import joblib
import pickle


def get_filename_from_absolute_path(url):
    return url.split("/")[-1]


def load_json(fn):
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, fn):
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(obj)


def load_pickle(fn):
    return joblib.load(fn)


def save_pickle(obj, fn):
    return joblib.dump(obj, fn, protocol=pickle.HIGHEST_PROTOCOL)
