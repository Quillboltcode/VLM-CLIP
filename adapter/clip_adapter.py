import torch.nn as nn


class TextAdapter(nn.Module):
    """
    Adapter module for CLIP text encoder.
    Adds a lightweight, trainable layer at the end of the text encoder.
    """

    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.up_project(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states


# need
# class SharedCrossAttentionAdapter(nn.Module):
#     """
#     Shared cross-attention adapter after MHSA in the CLIP text encoder.
#     """
#     def __init__(self, hidden_size, num_heads=8, dropout=0.1):
#         super().__init__()
#         self.cross_attention = CLIPAttention(
#             hidden_size,
#             num_heads,
#             dropout=dropout,
#             is_cross_attention=True
#          )
#         self.layer_norm1 = nn.LayerNorm(hidden_size)
#         self.layer_norm2 = nn.LayerNorm(hidden_size)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 4),
#             nn.GELU(),
#             nn.Linear(hidden_size * 4, hidden_size),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, hidden_states, encoder_hidden_states):
#         residual = hidden_states
#
#         # Cross attention block
#         hidden_states = self.layer_norm1(hidden_states)
#         encoder_hidden_states = self.layer_norm2(encoder_hidden_states)
#         hidden_states = self.cross_attention(
#             hidden_states=hidden_states,
#             encoder_hidden_states=encoder_hidden_states,
#             attention_mask=None
#         )[0]
#
#         # Feed forward block
#         hidden_states = residual + hidden_states
#         residual = hidden_states
#         hidden_states = self.mlp(self.layer_norm1(hidden_states))
#         hidden_states = residual + hidden_states
#
#         return hidden_states


class SharedMHSAttentionAdapter(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states, encoder_hidden_states):
        # Norm before attention
        q = k = v = self.norm1(encoder_hidden_states)
        hidden_states = self.norm2(hidden_states)

        # Cross attention
        attn_output, _ = self.cross_attn(query=hidden_states, key=q, value=v)

        # Residual connection
        hidden_states = hidden_states + attn_output
        residual = hidden_states

        # FFN block
        hidden_states = self.mlp(self.norm3(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states


class VisionAdapter(nn.Module):
    """
    Adapter module for CLIP vision encoder.
    Adds a lightweight, trainable layer at the end of the vision encoder.
    """

    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.up_project(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states
