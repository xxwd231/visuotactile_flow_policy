"""Normalized-time sinusoidal features followed by a learned MLP."""

import math

import torch
from torch import nn


def sinusoidal_time_features(
    flow_time: torch.Tensor, dimension: int,
    min_period: float = 4e-3, max_period: float = 4.0,
) -> torch.Tensor:
    """Return [B,dimension] float32 features ordered as sin, then cos."""
    if type(dimension) is not int or dimension < 2 or dimension % 2:
        raise ValueError("dimension must be positive and even")
    if not (math.isfinite(min_period) and math.isfinite(max_period)
            and 0 < min_period < max_period):
        raise ValueError("Require finite 0 < min_period < max_period")
    if flow_time.ndim != 1 or flow_time.shape[0] < 1 or not flow_time.is_floating_point():
        raise ValueError("flow_time must be floating [B]")
    if not torch.isfinite(flow_time).all() or bool(torch.any((flow_time < 0) | (flow_time > 1))):
        raise ValueError("flow_time must be finite and within [0,1]")
    t = flow_time.to(dtype=torch.float32)
    fraction = torch.linspace(0, 1, dimension // 2, device=t.device, dtype=torch.float32)
    periods = min_period * (max_period / min_period) ** fraction
    phase = (2 * math.pi) * t[:, None] / periods[None, :]
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(
        self, hidden_dim: int = 1024, time_embedding_dim: int = 256,
        min_period: float = 4e-3, max_period: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or time_embedding_dim < 2 or time_embedding_dim % 2:
            raise ValueError("hidden_dim must be positive and time_embedding_dim must be positive and even")
        if not (math.isfinite(min_period) and math.isfinite(max_period)
                and 0 < min_period < max_period):
            raise ValueError("Require finite 0 < min_period < max_period")
        self.time_embedding_dim = time_embedding_dim
        self.min_period = min_period
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, flow_time: torch.Tensor) -> torch.Tensor:
        if flow_time.ndim == 2 and flow_time.shape[1] == 1:
            flow_time = flow_time[:, 0]
        features = sinusoidal_time_features(
            flow_time, self.time_embedding_dim, self.min_period, self.max_period
        )
        return self.mlp(features)
