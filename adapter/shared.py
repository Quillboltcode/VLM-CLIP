import torch.nn as nn
from utillayer import MHSA, LayerNorm


class SharedAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(SharedAdapter, self).__init__()
        self.mhsa = MHSA(input_dim, num_heads)
        self.layer_norm = LayerNorm(input_dim)

    def forward(self, x):
        x = self.mhsa(x)
        x = self.layer_norm(x)
        return x
