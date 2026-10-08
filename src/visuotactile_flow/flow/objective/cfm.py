"""Standard conditional flow matching in normalized action space."""

from typing import Protocol

import torch

from ..convention import FlowConvention
from ..path import FlowPath
from ..sources import FlowSource, FlowTensorSpec
from ..time import TimeSampler
from .base import CFMObjectiveOutput, CFMTrainingBatch


class VelocityExpert(Protocol):
    def __call__(
        self, x_t: torch.Tensor, time: torch.Tensor,
        condition_tokens: torch.Tensor, condition_valid_mask: torch.Tensor,
    ) -> torch.Tensor: ...


class ConditionalFlowMatchingObjective:
    """Connect an injected source, time sampler, path and velocity expert."""

    def __init__(
        self, convention: FlowConvention, source: FlowSource,
        path: FlowPath, time_sampler: TimeSampler,
    ) -> None:
        if not isinstance(convention, FlowConvention):
            raise ValueError("convention must be FlowConvention")
        self.convention = convention
        self.source = source
        self.path = path
        self.time_sampler = time_sampler

    def prepare_training_batch(
        self, normalized_target_action: torch.Tensor, *,
        generator: torch.Generator | None = None,
        source_override: torch.Tensor | None = None,
        time_override: torch.Tensor | None = None,
    ) -> CFMTrainingBatch:
        """The target must already be normalized; this method never normalizes it."""
        spec = FlowTensorSpec.from_tensor(normalized_target_action)
        if not bool(torch.isfinite(normalized_target_action).all()):
            raise ValueError("normalized_target_action must be finite")
        source = (
            self.source.sample(spec, generator=generator)
            if source_override is None else source_override
        )
        if (not isinstance(source, torch.Tensor) or source.shape != spec.shape
                or source.dtype != spec.dtype or source.device != spec.device
                or not bool(torch.isfinite(source).all())):
            raise ValueError("source must match target shape, dtype and device and be finite")
        time = (
            self.time_sampler.sample(spec.shape[0], spec.device, generator=generator)
            if time_override is None else time_override
        )
        if (not isinstance(time, torch.Tensor) or time.shape != (spec.shape[0],)
                or time.dtype != torch.float32 or time.device != spec.device
                or not bool(torch.isfinite(time).all())
                or bool(torch.any((time < 0) | (time > 1)))):
            raise ValueError("time must be finite float32 [B] on target device in [0,1]")
        path_sample = self.path.sample(source, normalized_target_action, time, self.convention)
        return CFMTrainingBatch(
            x_t=path_sample.x_t, time=path_sample.time,
            velocity_target=path_sample.velocity_target,
            source=path_sample.source, target=path_sample.target,
        )

    def compute_loss(
        self, expert: VelocityExpert, batch: CFMTrainingBatch,
        condition_tokens: torch.Tensor, condition_valid_mask: torch.Tensor, *,
        action_valid_mask: torch.Tensor | None = None,
    ) -> CFMObjectiveOutput:
        if not isinstance(batch, CFMTrainingBatch):
            raise ValueError("batch must be CFMTrainingBatch")
        if (not isinstance(batch.x_t, torch.Tensor)
                or batch.x_t.ndim != 3 or any(dim < 1 for dim in batch.x_t.shape)
                or not batch.x_t.is_floating_point()):
            raise ValueError("batch x_t must be floating [B,H,D] with positive dimensions")
        batch_size, horizon, action_dim = batch.x_t.shape
        for name, tensor in (
            ("x_t", batch.x_t), ("velocity_target", batch.velocity_target),
            ("source", batch.source), ("target", batch.target),
        ):
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != batch.x_t.shape
                    or tensor.dtype != batch.x_t.dtype or tensor.device != batch.x_t.device
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError(f"batch {name} must match x_t shape, dtype and device and be finite")
        if (not isinstance(batch.time, torch.Tensor) or batch.time.shape != (batch_size,)
                or batch.time.dtype != torch.float32 or batch.time.device != batch.x_t.device
                or not bool(torch.isfinite(batch.time).all())
                or bool(torch.any((batch.time < 0) | (batch.time > 1)))):
            raise ValueError("batch time must be finite float32 [B] on action device in [0,1]")
        if (not isinstance(condition_tokens, torch.Tensor) or condition_tokens.ndim != 3
                or condition_tokens.shape[0] != batch_size or condition_tokens.shape[1] < 1
                or condition_tokens.device != batch.x_t.device
                or not condition_tokens.is_floating_point()
                or not bool(torch.isfinite(condition_tokens).all())):
            raise ValueError("condition_tokens must be finite floating [B,N,C] on action device")
        if (not isinstance(condition_valid_mask, torch.Tensor)
                or condition_valid_mask.shape != condition_tokens.shape[:2]
                or condition_valid_mask.dtype != torch.bool
                or condition_valid_mask.device != batch.x_t.device):
            raise ValueError("condition_valid_mask must be bool [B,N] on action device")
        if action_valid_mask is not None:
            if (not isinstance(action_valid_mask, torch.Tensor)
                    or action_valid_mask.shape != (batch_size, horizon)
                    or action_valid_mask.dtype != torch.bool
                    or action_valid_mask.device != batch.x_t.device):
                raise ValueError("action_valid_mask must be bool [B,H] on action device")
            if not bool(action_valid_mask.any()):
                raise ValueError("action_valid_mask cannot be all invalid")
        prediction = expert(batch.x_t, batch.time, condition_tokens, condition_valid_mask)
        if (not isinstance(prediction, torch.Tensor) or prediction.shape != batch.velocity_target.shape
                or prediction.device != batch.x_t.device or not prediction.is_floating_point()
                or not bool(torch.isfinite(prediction).all())):
            raise ValueError("expert prediction must be finite floating [B,H,D] on action device")
        squared_error = (prediction - batch.velocity_target).square()
        if action_valid_mask is None:
            loss = squared_error.mean()
        else:
            masked_error = torch.where(action_valid_mask[..., None], squared_error, 0)
            loss = masked_error.sum() / (action_valid_mask.sum() * action_dim)
        return CFMObjectiveOutput(
            loss=loss, prediction=prediction, velocity_target=batch.velocity_target,
            x_t=batch.x_t, time=batch.time, source=batch.source,
        )
