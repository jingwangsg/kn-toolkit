import joblib
import pickle


def load_pickle(file_path, silent=True):
    with open(file_path, "rb") as f:
        obj = joblib.load(f)
    if not silent:
        print(f"=> loaded {file_path}")
    return obj


def save_pickle(obj, file_path, silent=True):
    with open(file_path, "wb") as f:
        joblib.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    if not silent:
        print(f"=> saved to {file_path}")
