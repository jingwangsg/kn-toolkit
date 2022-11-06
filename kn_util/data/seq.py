import numpy as np


def generate_sample_indices(tot_len, max_len=None, stride=None):
    assert (max_len is not None) ^ (stride is not None)
    if max_len is not None:
        stride = int(np.ceil((tot_len - 1) / (max_len - 1)))

    indices = list(range(0, tot_len - 1, stride)) + [tot_len - 1]
    return indices
