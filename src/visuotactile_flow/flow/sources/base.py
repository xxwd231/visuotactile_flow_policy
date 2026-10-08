"""Action-space source contract, independent of clean targets."""

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class FlowTensorSpec:
    """Tensor allocation specification usable during training and inference."""

    shape: tuple[int, int, int]
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(type(dim) is not int or dim < 1 for dim in self.shape):
            raise ValueError("shape must be positive [B,H,D]")
        if not isinstance(self.device, torch.device) or not isinstance(self.dtype, torch.dtype):
            raise ValueError("device and dtype must be torch.device and torch.dtype")
        if not self.dtype.is_floating_point:
            raise ValueError("dtype must be floating point")

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "FlowTensorSpec":
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Expected a tensor")
        return cls(tuple(tensor.shape), tensor.device, tensor.dtype)


class FlowSource(Protocol):
    def sample(
        self,
        spec: FlowTensorSpec,
        condition: Any = None,
        history: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...
