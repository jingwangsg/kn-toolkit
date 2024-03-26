from typing import Any
import torch.utils.data as torch_data


class Subset(torch_data.Subset):
    def __getattr__(self, name):
        method = self.dataset.__getattribute__(name)
        return method
