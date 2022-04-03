from genericpath import samefile
import json

def load_json(path: str, silent=True):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not silent:
        print(f"=>load {path}")
    return obj

def save_json(obj, path: str, silent=True, sort_keys=True):
    with open(path, "w", encoding="utf-8") as f:
        # https://github.com/IsaacChanghau/VSLNet/blob/master/util/data_util.py
        f.write(json.dumps(obj, indent=4, sort_keys=sort_keys))
    if not silent:
        print(f"=>saved to {path}")