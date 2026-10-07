"""Deterministic sinusoidal flow-time features followed by a learned MLP."""

import math

import torch
from torch import nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int = 1024, time_embedding_dim: int = 256) -> None:
        super().__init__()
        if hidden_dim < 1 or time_embedding_dim < 2 or time_embedding_dim % 2:
            raise ValueError("hidden_dim must be positive and time_embedding_dim must be positive and even")
        self.time_embedding_dim = time_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, flow_time: torch.Tensor) -> torch.Tensor:
        if flow_time.ndim == 2 and flow_time.shape[1] == 1:
            flow_time = flow_time[:, 0]
        if flow_time.ndim != 1 or flow_time.shape[0] < 1 or not flow_time.is_floating_point():
            raise ValueError("flow_time must be floating [B] or [B,1]")
        if not torch.isfinite(flow_time).all() or bool(torch.any(flow_time < 0)) or bool(torch.any(flow_time > 1)):
            raise ValueError("flow_time must be finite and within [0,1]")
        # Form trigonometric features in float32 even under mixed-precision autocast.
        t = flow_time.float()
        half = self.time_embedding_dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        phase = t[:, None] * frequencies[None, :]
        return self.mlp(torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1))
