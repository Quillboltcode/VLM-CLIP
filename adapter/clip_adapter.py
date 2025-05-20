import torch.nn as nn


class LinearAdapter(nn.Module):
    def __init__(self, dim, adapter_dim):
        super().__init__()
        self.down = nn.Linear(dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))
