import torch


def batch_gather(x, inds):
    """
    N-axis is selected by inds
    output[b, m, k] = x[b, m, inds[m, k]]
    Args:
        x: (B, M, N)
        inds: (M, K)
    return: (B, M, K)
    """

    inds = inds[None, :].expand(x.shape[0], -1, -1)
    return torch.gather(x, 2, inds)
