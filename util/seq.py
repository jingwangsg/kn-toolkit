import torch.nn.utils.rnn as rnn
import torch
from typing import List
import numpy as np


def get_word_mask(range_i, n_position):
    ret_mask = np.zeros((n_position,))

    word_mask = [
        (idx, 1.0) if np.random.uniform(0, 1) < 0.15 else (idx, 0.0) for idx in range_i
    ]
    for idx, v in word_mask:
        ret_mask[idx] = v
    num_mask = [v for idx, v in word_mask]

    if np.sum(num_mask) == 0:
        mask_i = np.random.choice(range_i)
        ret_mask[mask_i] = 1.0
    if np.sum(num_mask) == len(range_i):
        unmask_i = np.random.choice(range_i)
        ret_mask[unmask_i] = 0.0

    return ret_mask


def pad_sequence(sents: List[torch.Tensor]):
    return rnn.pad_sequence(sents, batch_first=True)


def get_mask_from_lengths(lens: torch.Tensor):
    max_len = torch.max(lens)
    mask = torch.arange(max_len).unsqueeze(0) < lens.unsqueeze(1)

    return mask
