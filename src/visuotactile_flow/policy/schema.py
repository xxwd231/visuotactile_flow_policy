"""Stage-1 batched observation and policy result contracts."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolicyObservationBatch:
    """Policy-facing RGB, normalized tactile depth, and measured robot state."""

    external_rgb: torch.Tensor          # uint8 RGB [B,3,3,240,320]
    wrist_rgb: torch.Tensor             # uint8 RGB [B,3,3,240,320]
    tactile_depth_0: torch.Tensor       # normalized float [B,3,1,288,384]
    tactile_depth_1: torch.Tensor       # normalized float [B,3,1,288,384]
    agent_pos: torch.Tensor             # float [B,3,7]
    external_rgb_valid: torch.Tensor | None = None       # bool [B,3]
    wrist_rgb_valid: torch.Tensor | None = None
    tactile_depth_0_valid: torch.Tensor | None = None
    tactile_depth_1_valid: torch.Tensor | None = None
    agent_pos_valid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.external_rgb, torch.Tensor):
            raise ValueError("external_rgb must be a tensor")
        batch = self.external_rgb.shape[0] if self.external_rgb.ndim else 0
        if batch < 1:
            raise ValueError("Observation batch must be nonempty")
        device = self.external_rgb.device
        for name in ("external_rgb", "wrist_rgb"):
            value = getattr(self, name)
            if (not isinstance(value, torch.Tensor)
                    or value.shape != (batch, 3, 3, 240, 320)
                    or value.dtype != torch.uint8 or value.device != device):
                raise ValueError(f"{name} must be RGB uint8 [B,3,3,240,320] on one device")
        for name in ("tactile_depth_0", "tactile_depth_1"):
            value = getattr(self, name)
            if (not isinstance(value, torch.Tensor)
                    or value.shape != (batch, 3, 1, 288, 384)
                    or value.device != device or not value.is_floating_point()
                    or not bool(torch.isfinite(value).all())
                    or bool(torch.any((value < -1.00001) | (value > 1.00001)))):
                raise ValueError(f"{name} must be finite policy_normalized depth [B,3,1,288,384]")
        state = self.agent_pos
        if (not isinstance(state, torch.Tensor) or state.shape != (batch, 3, 7)
                or state.device != device or not state.is_floating_point()
                or not bool(torch.isfinite(state).all())):
            raise ValueError("agent_pos must be finite floating [B,3,7] on observation device")
        for name in (
            "external_rgb_valid", "wrist_rgb_valid", "tactile_depth_0_valid",
            "tactile_depth_1_valid", "agent_pos_valid",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, torch.Tensor) or value.shape != (batch, 3)
                or value.dtype != torch.bool or value.device != device
            ):
                raise ValueError(f"{name} must be bool [B,3] on observation device")


@dataclass(frozen=True)
class PolicyTrainingOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    velocity_target: torch.Tensor
    x_t: torch.Tensor
    time: torch.Tensor
    normalized_target_action: torch.Tensor
    condition_tokens: torch.Tensor
    condition_valid_mask: torch.Tensor


@dataclass(frozen=True)
class PolicySampleOutput:
    normalized_action: torch.Tensor    # [B,16,10] float32
    encoded_action: torch.Tensor       # denormalized fixed-anchor [B,16,10]
    num_function_evaluations: int
    time_grid: torch.Tensor
    trajectory: torch.Tensor | None
