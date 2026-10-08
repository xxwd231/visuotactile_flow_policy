"""Typed result of a conditional probability path sample."""

from dataclasses import dataclass
from typing import Protocol

import torch

from ..convention import FlowConvention


@dataclass(frozen=True)
class PathSample:
    x_t: torch.Tensor                # [B,H,D]
    velocity_target: torch.Tensor    # [B,H,D]
    source: torch.Tensor             # [B,H,D]
    target: torch.Tensor             # [B,H,D]
    time: torch.Tensor               # [B]


class FlowPath(Protocol):
    def sample(
        self, source: torch.Tensor, target: torch.Tensor,
        time: torch.Tensor, convention: FlowConvention,
    ) -> PathSample: ...
