import numpy as np
import pytest
import torch

from visuotactile_flow.data.action_codec import ActionCodec
from visuotactile_flow.data.lerobot_v21 import (
    LeRobotV21EpisodeAdapter, assemble_agent_pos, build_training_sample,
    complete_anchor_indices, pose6d_rotvec_to_matrix, reconstruct_tactile_depth,
    resize_rgb_frames,
)
from visuotactile_flow.data.schema import ActionLabelSource, DataConfig


CONFIG = DataConfig(tactile_representation_source="legacy_video_reconstructed")


def test_rotvec_zero_and_small_angle():
    poses = np.zeros((2, 6))
    poses[1, 3:] = [1e-9, -2e-9, 3e-9]
    matrices = pose6d_rotvec_to_matrix(poses)
    np.testing.assert_array_equal(matrices[0], np.eye(4))
    np.testing.assert_allclose(matrices[1, :3, :3], np.eye(3) + np.array([
        [0, -3e-9, -2e-9], [3e-9, 0, -1e-9], [2e-9, 1e-9, 0],
    ]), atol=1e-16)


def test_rotvec_known_axis_and_so3():
    pose = np.array([1, 2, 3, 0, 0, np.pi / 2])
    transform = pose6d_rotvec_to_matrix(pose)
    np.testing.assert_allclose(transform[:3, :3], [
        [0, -1, 0], [1, 0, 0], [0, 0, 1],
    ], atol=1e-12)
    np.testing.assert_array_equal(transform[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)


def test_agent_pos_uses_measured_joints_and_normalized_gripper():
    joints = np.arange(6, dtype=np.float32)[None]
    state = np.zeros((1, 13), dtype=np.float32)
    state[0, 6] = 0.7
    np.testing.assert_allclose(assemble_agent_pos(joints, state), [[0, 1, 2, 3, 4, 5, 0.7]], atol=1e-7)
    state[0, 6] = 255
    with pytest.raises(ValueError, match="gripper"):
        assemble_agent_pos(joints, state)


def test_complete_anchor_range():
    anchors = complete_anchor_indices(913, CONFIG)
    assert (anchors.start, anchors.stop, len(anchors)) == (2, 897, 895)
    with pytest.raises(ValueError, match="19 frames"):
        complete_anchor_indices(18, CONFIG)


def test_tactile_rgb_difference_over_255():
    rgb = np.zeros((3, 288, 384, 3), dtype=np.uint8)
    rgb[0, :, :, 0] = 255
    rgb[1, :, :, 2] = 128
    depth = reconstruct_tactile_depth(rgb)
    assert depth.shape == (3, 1, 288, 384)
    assert depth.dtype == torch.float32
    assert depth[0, 0, 0, 0] == 1
    assert depth[1, 0, 0, 0] == pytest.approx(-128 / 255)
    assert depth[2, 0, 0, 0] == 0


def test_rgb_resize_preserves_uint8_rgb_channel_order():
    rgb = np.zeros((3, 8, 10, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 25, 100, 225
    result = resize_rgb_frames(rgb)
    assert result.shape == (3, 3, 240, 320)
    assert result.dtype == torch.uint8
    assert [int(result[0, c, 100, 100]) for c in range(3)] == [25, 100, 225]


def _synthetic_columns():
    n = 19
    state = np.zeros((n, 13), dtype=np.float32)
    state[:, 0] = np.arange(n) * 0.001
    state[:, 5] = np.arange(n) * 0.01
    state[:, 6] = np.linspace(0.1, 0.9, n)
    joints = np.tile(np.arange(6, dtype=np.float32), (n, 1))
    times = np.arange(n, dtype=np.float32) / 30
    rgb = np.full((3, 8, 10, 3), 64, dtype=np.uint8)
    tactile = np.zeros((3, 288, 384, 3), dtype=np.uint8)
    return state, joints, times, rgb, tactile


def _sample(anchor: int):
    state, joints, times, rgb, tactile = _synthetic_columns()
    return build_training_sample(
        episode_id="synthetic", anchor_index=anchor, timestamps=times,
        state=state, joint_position=joints, external_rgb=rgb, wrist_rgb=rgb,
        tactile_rgb_0=tactile, tactile_rgb_1=tactile, data_config=CONFIG,
    )


def test_training_sample_uses_fixed_anchor_measured_future_geometry():
    sample = _sample(2)
    state, _, _, _, _ = _synthetic_columns()
    assert sample.action_label_source is ActionLabelSource.MEASURED_FUTURE
    assert sample.action_history is None
    assert sample.target_action.shape == (16, 10)
    assert sample.agent_pos.shape == (3, 7)
    assert sample.external_rgb.shape == sample.wrist_rgb.shape == (3, 3, 240, 320)
    assert sample.tactile_depth_0.shape == sample.tactile_depth_1.shape == (3, 1, 288, 384)
    decoded, gripper = ActionCodec().decode_chunk(sample.anchor_tcp_pose.numpy(), sample.target_action.numpy())
    expected = pose6d_rotvec_to_matrix(state[3:19, :6])
    np.testing.assert_allclose(decoded, expected, atol=1e-7)
    np.testing.assert_allclose(gripper, state[3:19, 6:7], atol=1e-7)


@pytest.mark.parametrize("anchor", [1, 3])
def test_incomplete_window_rejected(anchor):
    with pytest.raises(IndexError, match="complete"):
        _sample(anchor)


def test_fallback_tactile_serial_order_is_sorted(tmp_path):
    adapter = object.__new__(LeRobotV21EpisodeAdapter)
    adapter.dataset_root = tmp_path
    adapter.episode_index = 0
    (tmp_path / "tactile_videos/depth/Z9").mkdir(parents=True)
    (tmp_path / "tactile_videos/depth/A1").mkdir(parents=True)
    serials, paths = adapter._tactile_paths({})
    assert serials == {"tactile_depth_0": "A1", "tactile_depth_1": "Z9"}
    assert paths["tactile_depth_0"].parent.name == "A1"
