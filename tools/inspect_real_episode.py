"""Read one converted real episode and inspect one complete training window."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visuotactile_flow.data.action_codec import ActionCodec  # noqa: E402
from visuotactile_flow.data.dataset import load_data_config  # noqa: E402
from visuotactile_flow.data.lerobot_v21 import (  # noqa: E402
    LeRobotV21EpisodeAdapter, pose6d_rotvec_to_matrix,
)


def _describe(name: str, value: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(value.shape)} dtype={value.dtype} "
          f"range=[{value.min().item():.9g}, {value.max().item():.9g}] "
          f"finite={bool(torch.isfinite(value).all())}")


def _diff_ms(values: np.ndarray, scale: float) -> str:
    delta = np.diff(values[:min(len(values), 101)].astype(np.float64)) * scale
    return (f"mean={delta.mean():.9g} median={np.median(delta):.9g} "
            f"min={delta.min():.9g} max={delta.max():.9g}")


def inspect(dataset_root: Path, episode: int, anchor: int) -> None:
    config_path = ROOT / "configs/data/insertion_task_3_lerobot.yaml"
    config = load_data_config(config_path)
    print(f"data_config: {config_path}")
    with LeRobotV21EpisodeAdapter(dataset_root, episode, config) as adapter:
        for key, value in adapter.metadata.items():
            print(f"{key}: {dict(value) if key == 'tactile_serial_mapping' else value}")
        print(f"valid_anchor_indices: {adapter.valid_anchor_indices.start}..{adapter.valid_anchor_indices.stop - 1}")
        sample = adapter.get_window(anchor)
        print(f"anchor: {anchor}; episode_id: {sample.episode_id}; label: {sample.action_label_source.value}")
        for name in ("observation_timestamps", "external_rgb", "wrist_rgb", "tactile_depth_0",
                     "tactile_depth_1", "agent_pos", "anchor_tcp_pose", "target_action"):
            _describe(name, getattr(sample, name))
        action = sample.target_action.numpy()
        for name, part in (("translation", action[:, :3]), ("rotation6d", action[:, 3:9]),
                           ("gripper", action[:, 9:10])):
            print(f"target_{name}_range: [{part.min():.9g}, {part.max():.9g}]")
        true_pose = pose6d_rotvec_to_matrix(adapter.state[anchor + 1:anchor + 17, :6])
        true_gripper = adapter.state[anchor + 1:anchor + 17, 6:7]
        decoded_pose, decoded_gripper = ActionCodec().decode_chunk(sample.anchor_tcp_pose.numpy(), action)
        rotation_delta = np.swapaxes(decoded_pose[:, :3, :3], -1, -2) @ true_pose[:, :3, :3]
        sin_angle = np.linalg.norm(np.stack((
            rotation_delta[:, 2, 1] - rotation_delta[:, 1, 2],
            rotation_delta[:, 0, 2] - rotation_delta[:, 2, 0],
            rotation_delta[:, 1, 0] - rotation_delta[:, 0, 1],
        ), axis=-1), axis=-1) / 2
        cos_angle = np.clip((np.trace(rotation_delta, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
        print(f"roundtrip_translation_max_m: {np.max(np.abs(decoded_pose[:, :3, 3] - true_pose[:, :3, 3])):.9g}")
        print(f"roundtrip_rotation_matrix_max_abs: {np.max(np.abs(decoded_pose[:, :3, :3] - true_pose[:, :3, :3])):.9g}")
        print(f"roundtrip_rotation_angle_max_rad: {np.max(np.arctan2(sin_angle, cos_angle)):.9g}")
        print(f"roundtrip_gripper_max: {np.max(np.abs(decoded_gripper - true_gripper)):.9g}")
        print(f"lerobot_action_next_measured_max_abs: "
              f"{np.max(np.abs(adapter.lerobot_action[:-1] - adapter.state[1:, :7])):.9g}")
        print(f"aligned_timestamp_first100_diff_ms: {_diff_ms(adapter.timestamps, 1000)}")
        if adapter.ros_timestamp_ns is not None:
            # Subtract before float conversion to preserve nanosecond-scale differences.
            delta = np.diff(adapter.ros_timestamp_ns[:min(adapter.episode_length, 101)]) * 1e-6
            print(f"source_ros_timestamp_first100_diff_ms: mean={delta.mean():.9g} "
                  f"median={np.median(delta):.9g} min={delta.min():.9g} max={delta.max():.9g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--anchor", type=int, required=True)
    args = parser.parse_args()
    inspect(args.dataset_root, args.episode, args.anchor)


if __name__ == "__main__":
    main()
