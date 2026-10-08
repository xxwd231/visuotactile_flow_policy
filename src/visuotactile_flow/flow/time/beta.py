"""Beta sampling; PyTorch distribution sampling uses the global torch RNG."""

from dataclasses import dataclass
import math

import torch

from .base import validate_request


@dataclass(frozen=True)
class BetaTimeSampler:
    alpha: float
    beta: float
    scale: float = 1.0
    offset: float = 0.0
    complement: bool = False

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.alpha, self.beta, self.scale, self.offset)):
            raise ValueError("Beta parameters must be finite")
        if self.alpha <= 0 or self.beta <= 0 or self.scale <= 0:
            raise ValueError("alpha, beta, and scale must be positive")
        if self.offset < 0 or self.offset + self.scale > 1:
            raise ValueError("Require 0 <= offset and offset + scale <= 1")
        if type(self.complement) is not bool:
            raise ValueError("complement must be bool")

    def sample(
        self, batch_size: int, device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        validate_request(batch_size, device)
        if generator is not None:
            raise ValueError("BetaTimeSampler uses global torch RNG; generator is unsupported")
        distribution = torch.distributions.Beta(
            torch.tensor(self.alpha, device=device, dtype=torch.float32),
            torch.tensor(self.beta, device=device, dtype=torch.float32),
        )
        time = self.offset + self.scale * distribution.sample((batch_size,))
        return 1 - time if self.complement else time
