"""Typed result of a conditional probability path sample."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PathSample:
    x_t: torch.Tensor                # [B,H,D]
    velocity_target: torch.Tensor    # [B,H,D]
    source: torch.Tensor             # [B,H,D]
    target: torch.Tensor             # [B,H,D]
    time: torch.Tensor               # [B]
