# adapter.py

import torch.nn as nn


class TextualAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TextualAdapter, self).__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.gelu(x)
        x = self.up_proj(x)
        return x + residual


class TemporalDynamicAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalDynamicAdapter, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.scaling_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        scaled = self.scaling_layer(gru_out)
        return scaled + x  # Residual connection


class SharedAdapter(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SharedAdapter, self).__init__()
        self.mhsa = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_output, _ = self.mhsa(x, x, x)
        x = self.layer_norm(attn_output + x)  # Residual connection
        return x
