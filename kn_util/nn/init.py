import torch
import torch.nn as nn
import math
from functools import partial


def init_weight(m, method="kaiming"):
    _uniform_dict = {
        "kaiming": partial(nn.init.kaiming_uniform_, nonlinearity="relu"),
        "xavier": nn.init.xavier_uniform_,
    }
    uniform_ = _uniform_dict[method]
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if method == "xavier":
            uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif method == "kaiming":
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
