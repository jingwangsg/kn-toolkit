from typing import Any
from torch.utils.data import Subset


class Subset(Subset):
    def __getattr__(self, name):
        method = self.dataset.__getattribute__(name)
        return method
