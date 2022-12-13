import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers=2,
                 activation=F.relu,
                 has_ln=True,
                 dropout=0.1) -> None:
        super().__init__()
        assert num_layers > 1, "this class is intended for multiple linear layers"
        dims = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.activation = activation
        self.do = nn.Dropout(dropout)
        if has_ln:
            self.lns = nn.ModuleList(
                [nn.LayerNorm(dims[i]) for i in range(num_layers)])

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            if hasattr(self, "lns"):
                x = self.lns[idx](x)
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = self.activation(x)
                x = self.do(x)
        return x
