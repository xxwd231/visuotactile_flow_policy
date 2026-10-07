import numpy as np
import pytest

from visuotactile_flow.data.action_codec import ActionCodec


H = 16


def transform(rotation=None, translation=None):
    pose = np.eye(4)
    if rotation is not None:
        pose[:3, :3] = rotation
    if translation is not None:
        pose[:3, 3] = translation
    return pose


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def assert_round_trip(anchor, targets, gripper):
    codec = ActionCodec()
    chunk = codec.encode_chunk(anchor, targets, gripper)
    restored, restored_gripper = codec.decode_chunk(anchor, chunk)
    assert chunk.shape == gripper.shape[:-1] + (10,)
    np.testing.assert_allclose(restored, targets, atol=1e-6, rtol=0)
    np.testing.assert_allclose(restored_gripper, gripper, atol=1e-6, rtol=0)
    return chunk


def test_identity_anchor_and_zero_motion():
    anchor = np.eye(4)
    targets = np.broadcast_to(anchor, (H, 4, 4)).copy()
    chunk = assert_round_trip(anchor, targets, np.zeros((H, 1)))
    np.testing.assert_allclose(chunk[0], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0])


def test_pure_translation_is_anchor_frame_translation():
    anchor = transform(rotation_z(np.pi / 2), [1, 2, 3])
    targets = np.broadcast_to(anchor, (H, 4, 4)).copy()
    targets[:, :3, 3] += [0, 1, 0]
    chunk = assert_round_trip(anchor, targets, np.full((H, 1), 0.6))
    np.testing.assert_allclose(chunk[:, :3], np.tile([1, 0, 0], (H, 1)), atol=1e-6)
    np.testing.assert_allclose(chunk[:, 9], 0.6)


def test_pure_rotation_uses_first_two_rows_in_row_major_order():
    anchor = np.eye(4)
    rotation = rotation_z(0.73)
    targets = np.broadcast_to(transform(rotation), (H, 4, 4)).copy()
    chunk = assert_round_trip(anchor, targets, np.zeros((H, 1)))
    np.testing.assert_allclose(chunk[0, 3:9], rotation[:2, :].reshape(6), atol=1e-6)
    assert not np.allclose(chunk[0, 3:9], rotation[:, :2].reshape(6))


def test_random_se3_batch_and_fixed_anchor_per_chunk():
    rng = np.random.default_rng(42)
    anchors = np.stack([transform(random_rotation(rng), rng.normal(size=3)) for _ in range(3)])
    targets = np.stack([
        np.stack([transform(random_rotation(rng), rng.normal(size=3)) for _ in range(H)])
        for _ in range(3)
    ])
    gripper = rng.uniform(size=(3, H, 1))
    chunk = assert_round_trip(anchors, targets, gripper)
    assert chunk.shape == (3, H, 10)
    # One anchor is used for all steps; identical targets yield identical deltas.
    repeated = np.broadcast_to(targets[:, :1], targets.shape).copy()
    encoded = ActionCodec().encode_chunk(anchors, repeated, gripper)
    np.testing.assert_allclose(encoded[:, 0, :9], encoded[:, -1, :9])


def test_degenerate_rotation_rows_rejected():
    actions = np.zeros((H, 10))
    actions[:, 3:6] = [1, 0, 0]
    actions[:, 6:9] = [1, 0, 0]
    with pytest.raises(ValueError, match="linearly dependent"):
        ActionCodec().decode_chunk(np.eye(4), actions)


def test_incorrect_horizon_rejected():
    with pytest.raises(ValueError, match="Expected 16"):
        ActionCodec().encode_chunk(np.eye(4), np.tile(np.eye(4), (15, 1, 1)), np.zeros((15, 1)))
