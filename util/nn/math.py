import numpy as np


def gaussian(mean, sigma, n_position):
    """
    discrete gaussian over `n_position` indices
    return [n_position, ]
    """

    sigma = sigma
    mean = mean

    idx_tensor = np.arange(n_position)
    # (bsz, n_position)
    gauss = np.exp(-np.square(idx_tensor - mean) / (2 * np.square(sigma)))

    return gauss
