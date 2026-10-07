"""UR5e TCP chunk geometry, independent of the old DP runtime.

Contract checked against the former implementation at
ros2_teleop_dataset/vision_RL/offline_RL/usb_insertion/vendor/rl_100/
dataset/usb_tactile_dataset.py (to_delta/from_delta, lines 15-38) and
models/epoch800/action_spec.json. No old package is imported.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .schema import ActionSpec


def _poses(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2:] != (4, 4) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite homogeneous transforms [...,4,4]")
    if not np.allclose(array[..., 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = array[..., :3, :3]
    identity = np.eye(3)
    if not np.allclose(rotation @ np.swapaxes(rotation, -1, -2), identity, atol=1e-5) or not np.allclose(
        np.linalg.det(rotation), 1, atol=1e-5
    ):
        raise ValueError(f"{name} rotation must be in SO(3)")
    return array


def _rotation_from_first_two_rows(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """The old DP codec uses row-wise Gram-Schmidt then a right-handed cross.

    This also projects imperfect model outputs onto SO(3). Degenerate row
    pairs are rejected because they do not define a unique rotation.
    """
    first, second = rows[..., :3], rows[..., 3:6]
    norm_first = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm_first < 1e-8):
        raise ValueError("First rotation row is degenerate")
    first = first / norm_first
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    norm_second = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm_second < 1e-8):
        raise ValueError("Rotation rows are linearly dependent")
    second = second / norm_second
    return np.stack((first, second, np.cross(first, second)), axis=-2)


class ActionCodec:
    def __init__(self, spec: ActionSpec | None = None) -> None:
        self.spec = spec or ActionSpec()

    def encode_chunk(
        self, anchor_pose: ArrayLike, future_target_poses: ArrayLike, future_gripper: ArrayLike
    ) -> NDArray[np.float32]:
        """Encode T_delta = inverse(T_anchor) @ T_target for every step.

        Shapes: anchor [4,4] or [B,4,4]; targets [H,4,4] or [B,H,4,4];
        gripper [H,1] or [B,H,1]. The same anchor is used for every H step.
        """
        targets = _poses(future_target_poses, "future_target_poses")
        if targets.ndim < 3 or targets.shape[-3] != self.spec.horizon:
            raise ValueError(f"Expected {self.spec.horizon} future poses")
        anchor = _poses(anchor_pose, "anchor_pose")
        try:
            anchor = np.broadcast_to(anchor[..., None, :, :], targets.shape)
        except ValueError as exc:
            raise ValueError("Anchor batch shape does not match target poses") from exc
        gripper = np.asarray(future_gripper, dtype=np.float64)
        if gripper.shape != targets.shape[:-2] + (1,) or not np.isfinite(gripper).all():
            raise ValueError("future_gripper must have shape [...,H,1] and finite values")
        anchor_rotation = anchor[..., :3, :3]
        delta_rotation = np.swapaxes(anchor_rotation, -1, -2) @ targets[..., :3, :3]
        delta_translation = np.einsum(
            "...ji,...j->...i", anchor_rotation, targets[..., :3, 3] - anchor[..., :3, 3]
        )
        rotation6d = delta_rotation[..., :2, :].reshape(*targets.shape[:-2], 6)
        return np.concatenate((delta_translation, rotation6d, gripper), axis=-1).astype(np.float32)

    def decode_chunk(
        self, anchor_pose: ArrayLike, action_chunk: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        """Decode all steps against the fixed anchor; no delta accumulation."""
        actions = np.asarray(action_chunk, dtype=np.float64)
        if actions.ndim < 2 or actions.shape[-2:] != (self.spec.horizon, self.spec.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"action_chunk must be finite [...,{self.spec.horizon},{self.spec.action_dim}]")
        anchor = _poses(anchor_pose, "anchor_pose")
        try:
            anchor = np.broadcast_to(anchor[..., None, :, :], actions.shape[:-1] + (4, 4))
        except ValueError as exc:
            raise ValueError("Anchor batch shape does not match action chunk") from exc
        anchor_rotation = anchor[..., :3, :3]
        delta_rotation = _rotation_from_first_two_rows(actions[..., 3:9])
        targets = np.broadcast_to(np.eye(4), actions.shape[:-1] + (4, 4)).copy()
        targets[..., :3, :3] = anchor_rotation @ delta_rotation
        targets[..., :3, 3] = np.einsum("...ij,...j->...i", anchor_rotation, actions[..., :3]) + anchor[..., :3, 3]
        return targets, actions[..., 9:10].astype(np.float32)
