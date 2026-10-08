import json
from pathlib import Path

import pytest
import torch
import yaml

from visuotactile_flow.normalization import (
    LegacyDPLimitsNormalizer, MinMaxNormalizerV2,
    StructuredActionNormalizer, StructuredStateNormalizer,
)


ROOT = Path(__file__).resolve().parents[1]


def _legacy_reference(x: torch.Tensor, output_min=-1.0, output_max=1.0, range_eps=1e-4):
    # Directly reproduce the old DP limits branch, including its float32 fit.
    values = x.float().reshape(-1, x.shape[-1])
    input_min = values.min(axis=0).values
    input_max = values.max(axis=0).values
    input_mean = values.mean(axis=0)
    input_std = values.std(axis=0)
    input_range = input_max - input_min
    ignored = input_range < range_eps
    input_range[ignored] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignored] = (output_max + output_min) / 2 - input_min[ignored]
    return scale, offset, ignored, input_min, input_max, input_mean, input_std


def test_legacy_near_constant_retains_small_difference():
    x = torch.tensor([[1.000000], [1.000001]], dtype=torch.float32)
    legacy = LegacyDPLimitsNormalizer().fit(x)
    assert legacy.scale.item() == pytest.approx(1.0)
    assert legacy.offset.item() == pytest.approx(-1.0)
    torch.testing.assert_close(legacy.normalize(x), x - 1.0)
    torch.testing.assert_close(legacy.denormalize(legacy.normalize(x)), x)
    v2 = MinMaxNormalizerV2().fit(x)
    torch.testing.assert_close(v2.normalize(x), torch.zeros_like(x))
    assert legacy.normalize(x)[1, 0] > v2.normalize(x)[1, 0]


@pytest.mark.parametrize("shape", [(80, 7), (8, 10, 7)])
def test_legacy_matches_old_formula_per_channel(shape):
    x = torch.randn(shape, generator=torch.Generator().manual_seed(311))
    x[..., 0] = 1.0                    # constant
    x[..., 1] = 1.0 + x[..., 1] * 1e-6  # near-constant
    actual = LegacyDPLimitsNormalizer().fit(x)
    scale, offset, ignored, low, high, mean, std = _legacy_reference(x)
    torch.testing.assert_close(actual.scale, scale, rtol=0, atol=0)
    torch.testing.assert_close(actual.offset, offset, rtol=0, atol=0)
    torch.testing.assert_close(actual.floored_channels, ignored)
    for key, expected in (("raw_min", low), ("raw_max", high),
                          ("raw_mean", mean), ("raw_std", std)):
        torch.testing.assert_close(actual.stats[key], expected, rtol=0, atol=0)
    expected_normalized = x * scale + offset
    torch.testing.assert_close(actual.normalize(x), expected_normalized, rtol=0, atol=0)
    torch.testing.assert_close(actual.denormalize(actual.normalize(x)), x, atol=2e-6, rtol=1e-6)
    saved = json.loads(json.dumps(actual.state_dict()))
    assert saved["normalizer_version"] == 2
    assert saved["type"] == "legacy_dp_limits"
    assert saved["parameters"] == {"range_eps": 1e-4, "output_min": -1.0, "output_max": 1.0}
    restored = LegacyDPLimitsNormalizer().load_state_dict(saved)
    torch.testing.assert_close(restored.normalize(x), expected_normalized)


def test_nondefault_output_bounds_follow_old_formula():
    x = torch.tensor([[2.0, 1.0], [4.0, 1.000001]], dtype=torch.float32)
    n = LegacyDPLimitsNormalizer(output_min=0.0, output_max=2.0).fit(x)
    scale, offset, *_ = _legacy_reference(x, output_min=0.0, output_max=2.0)
    torch.testing.assert_close(n.scale, scale)
    torch.testing.assert_close(n.offset, offset)
    torch.testing.assert_close(n.denormalize(n.normalize(x)), x)


def test_legacy_yaml_modes_and_structured_round_trip():
    action_modes = yaml.safe_load((ROOT / "configs/normalization/action_legacy_minmax.yaml").read_text())
    state_modes = yaml.safe_load((ROOT / "configs/normalization/state_legacy_minmax.yaml").read_text())
    assert {part["mode"] for part in action_modes.values()} == {"legacy_limits"}
    assert {part["mode"] for part in state_modes.values()} == {"legacy_limits"}
    actions = torch.randn(12, 16, 10)
    states = torch.randn(12, 3, 7)
    action_normalizer = StructuredActionNormalizer(action_modes).fit(actions)
    state_normalizer = StructuredStateNormalizer(state_modes).fit(states)
    torch.testing.assert_close(action_normalizer.denormalize(action_normalizer.normalize(actions)),
                               actions, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(state_normalizer.denormalize(state_normalizer.normalize(states)),
                               states, atol=2e-6, rtol=1e-6)
    restored = StructuredActionNormalizer(action_modes).load_state_dict(
        json.loads(json.dumps(action_normalizer.state_dict())))
    torch.testing.assert_close(restored.normalize(actions), action_normalizer.normalize(actions))
