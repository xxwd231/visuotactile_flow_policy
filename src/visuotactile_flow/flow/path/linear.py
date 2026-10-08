"""Linear conditional path from source to clean target."""

import torch

from ..convention import FlowConvention
from .base import PathSample


class LinearConditionalFlowPath:
    def sample(
        self, source: torch.Tensor, target: torch.Tensor, time: torch.Tensor,
        convention: FlowConvention,
    ) -> PathSample:
        if not isinstance(convention, FlowConvention):
            raise ValueError("Expected FlowConvention")
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("source and target must be tensors")
        if source.ndim != 3 or any(dim < 1 for dim in source.shape) or source.shape != target.shape:
            raise ValueError("source and target must have matching positive [B,H,D] shape")
        if not source.is_floating_point() or source.dtype != target.dtype:
            raise ValueError("source and target must have matching floating dtype")
        if not isinstance(time, torch.Tensor) or time.shape != (source.shape[0],) or not time.is_floating_point():
            raise ValueError("time must be floating [B]")
        if source.device != target.device or source.device != time.device:
            raise ValueError("source, target, and time must share a device")
        if not all(bool(torch.isfinite(x).all()) for x in (source, target, time)):
            raise ValueError("source, target, and time must be finite")
        if bool(torch.any((time < 0) | (time > 1))):
            raise ValueError("time must lie in [0,1]")
        expanded_time = time.to(dtype=source.dtype)[:, None, None]
        source_weight = (convention.target_time - expanded_time) * convention.integration_sign
        x_t = source_weight * source + (1 - source_weight) * target
        velocity_target = (target - source) * convention.integration_sign
        return PathSample(x_t=x_t, velocity_target=velocity_target,
                          source=source, target=target, time=time)
