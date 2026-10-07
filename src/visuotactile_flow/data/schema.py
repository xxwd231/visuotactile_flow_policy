"""Schema for newly collected episodes and future training windows.

Arrays here represent aligned samples. Acquisition must also retain its raw
sensor streams and provenance; this module does not perform synchronization.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

import torch


class ActionLabelSource(StrEnum):
    MEASURED_FUTURE = "measured_future"
    COMMANDED_TARGET = "commanded_target"


class ActionSource(StrEnum):
    POLICY = "policy"
    HUMAN = "human"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ActionSpec:
    horizon: int = 16
    action_dim: int = 10
    translation_dim: int = 3
    rotation_dim: int = 6
    gripper_dim: int = 1
    rotation_representation: str = "matrix_first_two_rows"
    rotation_flatten: str = "row_major"
    delta_frame: str = "anchor_tcp_frame"
    chunk_anchor: str = "single_latest_observation"
    gripper_mode: str = "absolute_normalized"

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.action_dim != self.translation_dim + self.rotation_dim + self.gripper_dim:
            raise ValueError("Invalid action dimensions or horizon")
        expected = ("matrix_first_two_rows", "row_major", "anchor_tcp_frame",
                    "single_latest_observation", "absolute_normalized")
        actual = (self.rotation_representation, self.rotation_flatten, self.delta_frame,
                  self.chunk_anchor, self.gripper_mode)
        if actual != expected or (self.translation_dim, self.rotation_dim, self.gripper_dim) != (3, 6, 1):
            raise ValueError("Unsupported robot action geometry")


@dataclass(frozen=True)
class DataConfig:
    action_label_source: ActionLabelSource = ActionLabelSource.MEASURED_FUTURE
    observation_history: int = 3
    action_horizon: int = 16
    action_dim: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_label_source", ActionLabelSource(self.action_label_source))
        spec = ActionSpec()
        if (self.observation_history, self.action_horizon, self.action_dim) != (3, spec.horizon, spec.action_dim):
            raise ValueError("This interface requires Tobs=3, H=16, D=10")


@dataclass
class Episode:
    """Aligned episode, retaining measured and commanded targets separately.

    Images and depth may use acquisition-native spatial dimensions. Their first
    axis is the aligned frame axis; no specific storage format is implied.
    """

    episode_id: str
    task_id: str
    timestamps: torch.Tensor
    external_rgb: torch.Tensor
    wrist_rgb: torch.Tensor
    tactile_depth_0: torch.Tensor
    tactile_depth_1: torch.Tensor
    agent_pos: torch.Tensor
    tcp_pose: torch.Tensor
    gripper_state: torch.Tensor
    source_timestamp: Mapping[str, torch.Tensor] = field(default_factory=dict)
    source_age: Mapping[str, torch.Tensor] = field(default_factory=dict)
    frame_reused: Mapping[str, torch.Tensor] = field(default_factory=dict)
    commanded_tcp_target: torch.Tensor | None = None
    commanded_gripper_target: torch.Tensor | None = None
    success: bool | None = None
    failure: bool | None = None
    intervention: torch.Tensor | None = None
    action_source: tuple[ActionSource, ...] | None = None

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if n < 1 or not self.episode_id or not self.task_id:
            raise ValueError("Episode needs identifiers and frames")
        if self.timestamps.ndim != 1 or not bool(torch.all(self.timestamps[1:] > self.timestamps[:-1])):
            raise ValueError("timestamps must be a strictly increasing vector")
        for name in ("external_rgb", "wrist_rgb", "tactile_depth_0", "tactile_depth_1"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} frame count differs from timestamps")
        for name, tail in (("agent_pos", (7,)), ("tcp_pose", (4, 4)),
                           ("gripper_state", (1,)), ("commanded_tcp_target", (4, 4)),
                           ("commanded_gripper_target", (1,))):
            value = getattr(self, name)
            if value is not None and tuple(value.shape) != (n, *tail):
                raise ValueError(f"{name} must have shape {(n, *tail)}")
        for name in ("source_timestamp", "source_age", "frame_reused"):
            if any(len(value) != n for value in getattr(self, name).values()):
                raise ValueError(f"{name} entries must align to episode frames")
        if self.intervention is not None and tuple(self.intervention.shape) != (n,):
            raise ValueError("intervention must have one value per frame")
        if self.action_source is not None:
            self.action_source = tuple(ActionSource(item) for item in self.action_source)
            if len(self.action_source) != n:
                raise ValueError("action_source must have one value per frame")


@dataclass
class TrainingSample:
    """One future training window; action_history is reserved for later A2A."""

    episode_id: str
    observation_timestamps: torch.Tensor  # [3]
    external_rgb: torch.Tensor             # [3, ...]
    wrist_rgb: torch.Tensor                # [3, ...]
    tactile_depth_0: torch.Tensor          # [3, ...]
    tactile_depth_1: torch.Tensor          # [3, ...]
    agent_pos: torch.Tensor                # [3, 7]
    anchor_tcp_pose: torch.Tensor          # [4, 4]
    target_action: torch.Tensor            # [16, 10], encoded by ActionCodec
    action_label_source: ActionLabelSource
    action_history: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.action_label_source = ActionLabelSource(self.action_label_source)
        spec = ActionSpec()
        if tuple(self.observation_timestamps.shape) != (3,) or tuple(self.agent_pos.shape) != (3, 7):
            raise ValueError("Training observation must have three 7-D state frames")
        if any(len(getattr(self, name)) != 3 for name in
               ("external_rgb", "wrist_rgb", "tactile_depth_0", "tactile_depth_1")):
            raise ValueError("Training images and depth require three frames")
        if tuple(self.anchor_tcp_pose.shape) != (4, 4) or tuple(self.target_action.shape) != (spec.horizon, spec.action_dim):
            raise ValueError("Training action must be anchored [16, 10]")
        if self.action_history is not None and (self.action_history.ndim != 2 or self.action_history.shape[-1] != spec.action_dim):
            raise ValueError("action_history must have shape [Th, 10]")
