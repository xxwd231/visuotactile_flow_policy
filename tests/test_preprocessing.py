import pytest
import torch

from visuotactile_flow.encoders import DepthEncoding, RGBPreprocessor, TactilePreprocessor, raw_depth_to_policy_depth


def test_rgb_shape_dtype_finite_and_imagenet_values():
    x = torch.full((2, 3, 3, 240, 320), 255, dtype=torch.uint8)
    y = RGBPreprocessor()(x)
    assert y.shape == (2, 3, 3, 216, 216)
    assert y.dtype == torch.float32 and torch.isfinite(y).all()
    expected = torch.tensor([(1 - m) / s for m, s in zip(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    torch.testing.assert_close(y[0, 0, :, 100, 100], expected)


def test_tactile_resize_without_imagenet_normalization():
    x = torch.full((2, 3, 1, 288, 384), 0.5)
    y = TactilePreprocessor(DepthEncoding.POLICY_NORMALIZED)(x)
    assert y.shape == (2, 3, 1, 224, 224)
    torch.testing.assert_close(y, torch.full_like(y, 0.5))


def test_raw_tactile_normalization_clip_and_explicit_encoding():
    raw = torch.tensor([[[[[0.35, 0.7, -1.4]]]]])
    policy = raw_depth_to_policy_depth(raw, input_encoding="raw_sdk_depth")
    torch.testing.assert_close(policy.flatten(), torch.tensor([0.5, 1.0, -1.0]))
    with pytest.raises(ValueError, match="requires raw_sdk_depth"):
        raw_depth_to_policy_depth(policy, input_encoding="policy_normalized")
    with pytest.raises(ValueError, match="policy_normalized depth"):
        TactilePreprocessor("policy_normalized")(torch.full((1, 1, 1, 288, 384), 2.0))


def test_tactile_raw_and_normalized_modes_agree_after_one_conversion():
    raw = torch.full((1, 1, 1, 288, 384), 0.21)
    normalized = raw_depth_to_policy_depth(raw, input_encoding="raw_sdk_depth")
    a = TactilePreprocessor("raw_sdk_depth")(raw)
    b = TactilePreprocessor("policy_normalized")(normalized)
    torch.testing.assert_close(a, b)
