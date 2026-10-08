"""Generic ODE velocity callback and integration result."""

from dataclasses import dataclass
from typing import Protocol

import torch


class VelocityField(Protocol):
    def __call__(self, x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class EulerResult:
    sample: torch.Tensor                    # [B,H,D]
    time_grid: torch.Tensor                 # float32 [num_steps+1]
    num_function_evaluations: int
    trajectory: torch.Tensor | None = None  # [num_steps+1,B,H,D] if requested
