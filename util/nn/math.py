import torch


def gaussian(mean, sigma, n_position):
    """
    discrete gaussian over `n_position` indices
    * mean: (bsz, )
    * sigma (bsz, )
    return (bsz, n_position)
    """

    bsz = mean.shape[0]

    sigma = sigma.unsqueeze(1)
    mean = mean.unsqueeze(1)

    idx_tensor = torch.arange(n_position).unsqueeze(0).repeat(bsz, 1)
    # (bsz, n_position)
    gauss = torch.exp(-torch.square(idx_tensor - mean) / (2 * torch.square(sigma)))

    return gauss
