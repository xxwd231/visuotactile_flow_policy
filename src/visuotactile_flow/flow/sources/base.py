"""Action-space source contract. A2A sources may be added in a later phase."""

from typing import Any, Protocol

import torch


class FlowSource(Protocol):
    def sample(
        self,
        target_action: torch.Tensor,
        condition: Any = None,
        history: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...


class GaussianSource:
    def sample(
        self,
        target_action: torch.Tensor,
        condition: Any = None,
        history: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del condition, history
        if not target_action.is_floating_point():
            raise ValueError("target_action must be floating point")
        # torch.randn_like does not accept generator in every supported torch.
        return torch.randn(target_action.shape, dtype=target_action.dtype,
                           device=target_action.device, generator=generator)
