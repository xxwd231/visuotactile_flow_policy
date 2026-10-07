import json

import pytest
import torch

from visuotactile_flow.data.normalizer import MinMaxNormalizer


@pytest.mark.parametrize("shape", [(12, 16, 10), (12, 3, 7)])
def test_round_trip_for_action_and_state(shape):
    x = torch.randn(shape, generator=torch.Generator().manual_seed(3))
    normalizer = MinMaxNormalizer().fit(x)
    normalized = normalizer.normalize(x)
    restored = normalizer.denormalize(normalized)
    torch.testing.assert_close(restored, x)
    torch.testing.assert_close(normalized.flatten(0, -2).amin(0), -torch.ones(shape[-1]))
    torch.testing.assert_close(normalized.flatten(0, -2).amax(0), torch.ones(shape[-1]))


def test_constant_channel_has_finite_midpoint_and_exact_inverse():
    x = torch.tensor([[[1.0, 4.0]], [[3.0, 4.0]]])
    normalizer = MinMaxNormalizer().fit(x)
    normalized = normalizer.normalize(x)
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized[..., 1], torch.zeros((2, 1)))
    torch.testing.assert_close(normalizer.denormalize(normalized), x)


def test_json_round_trip_preserves_parameters():
    x = torch.randn((7, 16, 10))
    original = MinMaxNormalizer().fit(x)
    restored = MinMaxNormalizer().load_state_dict(json.loads(json.dumps(original.state_dict())))
    torch.testing.assert_close(restored.normalize(x), original.normalize(x))
    torch.testing.assert_close(restored.denormalize(restored.normalize(x)), x)


def test_unfitted_and_wrong_dimension_rejected():
    with pytest.raises(RuntimeError):
        MinMaxNormalizer().normalize(torch.zeros(2, 3))
    normalizer = MinMaxNormalizer().fit(torch.zeros(2, 3))
    with pytest.raises(ValueError):
        normalizer.normalize(torch.zeros(2, 4))
