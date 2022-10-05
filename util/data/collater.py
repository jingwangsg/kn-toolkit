import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import numpy as np


def collate_tensor(feat_list: List[Dict],
                   keys: List[str] = None):
    if keys is None:
        keys = feat_list[0].keys()

    temp_dict = dict()

    for feat in feat_list:
        for k in keys:
            temp_dict[k] = temp_dict.get(k, []) + [feat[k]]
    
    for k in keys:
        temp_dict[k] = np.stack(temp_dict[k])
        temp_dict[k] = torch.from_numpy(temp_dict[k])
    
    return temp_dict[k]
