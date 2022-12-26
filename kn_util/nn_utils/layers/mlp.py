import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, activation="relu", dropout=0.1) -> None:
        super().__init__()
        assert num_layers > 1, "this class is intended for multiple linear layers"
        dims = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.activation = self.build_activation(activation)
        self.do = nn.Dropout(dropout)
    
    def build_activation(self, name):
        if name == "prelu":
            activation = nn.PReLU(device=next(self.parameters()).device)
        elif name == "relu":
            activation = F.relu
        elif name == "gelu":
            activation = nn.GELU()
        else:
            activation = None
            raise Exception(f"no such activation as {name}")
        
        return activation

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = self.activation(x)
                x = self.do(x)
        return x
