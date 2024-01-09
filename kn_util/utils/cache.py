from .io import save_pickle, load_pickle
import os


def cached_func(cache_path, verbose=False):

    def wrapper(func):

        def inner_wrapper(*args, **kwargs):
            if os.path.exists(cache_path):
                if verbose:
                    print(f"Result loaded from {cache_path}")
                return load_pickle(cache_path)
            else:
                result = func(*args, **kwargs)
                save_pickle(result, cache_path)
                if verbose:
                    print(f"Result saved to {cache_path}")
                return result

        return inner_wrapper

    return wrapper
