import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from kn_util.debug import explore_content as EC

def _to_numpy(x):
    if torch.is_tensor(x):
        x = x.clone().detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        x = x
    else:
        raise ValueError(f"Unknown type \n {EC(x, print_str=False)}")
    
    assert x.ndim == 2, "Only support 2D array"
    return x

def plot_heatmap(x, fig="./figs/heatmap.png", **kwargs):
    x = _to_numpy(x)
    plt.figure()
    sns.heatmap(x, **kwargs)
    plt.savefig(fig)
    plt.close()