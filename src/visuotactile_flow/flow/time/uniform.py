"""Uniform sampling of normalized model time."""

from dataclasses import dataclass
import math

import torch

from .base import validate_request


@dataclass(frozen=True)
class UniformTimeSampler:
    min_time: float = 0.0
    max_time: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.min_time) and math.isfinite(self.max_time)
                and 0 <= self.min_time < self.max_time <= 1):
            raise ValueError("Require 0 <= min_time < max_time <= 1")

    def sample(
        self, batch_size: int, device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        validate_request(batch_size, device)
        u = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
        return self.min_time + (self.max_time - self.min_time) * u
