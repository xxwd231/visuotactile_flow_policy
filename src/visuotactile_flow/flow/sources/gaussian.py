"""Target-independent standard Gaussian flow source."""

from typing import Any

import torch

from .base import FlowTensorSpec


class GaussianSource:
    """Standard N(0,I) source. Condition and history are reserved inputs."""

    def sample(
        self,
        spec: FlowTensorSpec,
        condition: Any = None,
        history: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del condition, history
        if not isinstance(spec, FlowTensorSpec):
            raise ValueError("Expected FlowTensorSpec")
        return torch.randn(spec.shape, dtype=spec.dtype, device=spec.device, generator=generator)
