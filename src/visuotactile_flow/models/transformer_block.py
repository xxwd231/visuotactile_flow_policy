"""Bidirectional action self-attention with optional condition cross-attention."""

import torch
from torch import nn

from .layers import modulate


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_ratio: int,
        *,
        cross_attention: bool,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)
        if cross_attention:
            self.norm_ca = nn.LayerNorm(hidden_dim)
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
            )
        else:
            self.norm_ca = None
            self.cross_attention = None

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        condition_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = self.modulation(time_embedding).chunk(6, dim=-1)
        h = modulate(self.norm_sa(x), shift_sa, scale_sa)
        sa_out = self.self_attention(h, h, h, need_weights=False)[0]
        x = x + gate_sa[:, None, :] * sa_out
        if self.cross_attention is not None:
            ca_out = self.cross_attention(
                self.norm_ca(x), condition_tokens, condition_tokens,
                key_padding_mask=key_padding_mask, need_weights=False,
            )[0]
            x = x + ca_out
        h = modulate(self.norm_ffn(x), shift_ffn, scale_ffn)
        return x + gate_ffn[:, None, :] * self.ffn(h)
