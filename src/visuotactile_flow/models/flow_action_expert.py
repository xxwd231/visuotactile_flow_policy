"""Action velocity network v_theta(x_t, t, condition); no training objective."""

import torch
from torch import nn

from .layers import modulate
from .timestep import TimestepEmbedder
from .transformer_block import TransformerBlock


class FlowActionExpert(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 14,
        num_heads: int = 16,
        ffn_ratio: int = 4,
        action_horizon: int = 16,
        action_dim: int = 10,
        cross_attention_every: int = 2,
        time_embedding_dim: int = 256,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        adaln_zero: bool = True,
    ) -> None:
        super().__init__()
        if min(hidden_dim, num_layers, num_heads, ffn_ratio, action_horizon, action_dim,
               cross_attention_every) < 1 or hidden_dim % num_heads:
            raise ValueError("Positive dimensions required and hidden_dim must divide by num_heads")
        if not adaln_zero:
            raise ValueError("This baseline requires adaln_zero=True")
        if not (0 <= dropout < 1 and 0 <= attention_dropout < 1):
            raise ValueError("Dropout probabilities must lie in [0,1)")
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.action_input_projection = nn.Linear(action_dim, hidden_dim)
        self.action_position_embedding = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        nn.init.normal_(self.action_position_embedding, std=0.02)
        self.timestep_embedder = TimestepEmbedder(hidden_dim, time_embedding_dim)
        self.cross_attention_block_indices = tuple(
            i for i in range(num_layers) if (i + 1) % cross_attention_every == 0
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim, num_heads, ffn_ratio,
                cross_attention=i in self.cross_attention_block_indices,
                dropout=dropout, attention_dropout=attention_dropout,
            ) for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.final_output = nn.Linear(hidden_dim, action_dim)
        for layer in (self.final_modulation[1], self.final_output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        noisy_action: torch.Tensor,
        flow_time: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_action.ndim != 3 or noisy_action.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError(f"noisy_action must have shape [B,{self.action_horizon},{self.action_dim}]")
        batch = noisy_action.shape[0]
        if batch < 1 or condition_tokens.ndim != 3 or condition_tokens.shape[0] != batch or condition_tokens.shape[1] < 1 or condition_tokens.shape[2] != self.hidden_dim:
            raise ValueError(f"condition_tokens must have shape [B,N,{self.hidden_dim}] with B,N>0")
        if condition_valid_mask.shape != condition_tokens.shape[:2] or condition_valid_mask.dtype != torch.bool:
            raise ValueError("condition_valid_mask must be bool [B,N]")
        if flow_time.shape not in ((batch,), (batch, 1)):
            raise ValueError("flow_time must have shape [B] or [B,1]")
        if any(t.device != noisy_action.device for t in (flow_time, condition_tokens, condition_valid_mask)):
            raise ValueError("All inputs must be on the same device")
        if not noisy_action.is_floating_point() or not condition_tokens.is_floating_point() or not flow_time.is_floating_point():
            raise ValueError("Action, condition, and flow_time must be floating point")
        if not torch.isfinite(noisy_action).all() or not torch.isfinite(condition_tokens).all():
            raise ValueError("Action and condition must be finite")
        if not condition_valid_mask.any(dim=1).all():
            raise ValueError("Each sample must contain at least one valid condition token")
        time_embedding = self.timestep_embedder(flow_time)
        x = self.action_input_projection(noisy_action) + self.action_position_embedding
        time_embedding = time_embedding.to(dtype=x.dtype)
        # ConditionAdapter: True=valid. PyTorch MHA key_padding_mask: True=ignore.
        key_padding_mask = ~condition_valid_mask
        for block in self.blocks:
            x = block(x, time_embedding, condition_tokens, key_padding_mask)
        shift, scale = self.final_modulation(time_embedding).chunk(2, dim=-1)
        return self.final_output(modulate(self.final_norm(x), shift, scale))
