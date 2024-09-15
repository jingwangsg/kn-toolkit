import numpy as np


def mask_safe(mask):
    """ generate 0/1 that make sure at least one 0 and one 1 exist in the mask
    """
    tot_len = mask.shape[0]
    mask_cnt = np.sum(mask.astype(np.int32))
    range_i = np.arange(tot_len)
    
    if tot_len == mask_cnt or mask_cnt == 0:
        idx = np.random.choice(range_i)
        mask[idx] = 1 - mask[idx]
    
    return mask