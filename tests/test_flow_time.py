import pytest
import torch

from visuotactile_flow.flow import BetaTimeSampler, UniformTimeSampler


DEVICE = torch.device("cpu")


def test_uniform_range_dtype_and_seed():
    sampler = UniformTimeSampler()
    a = sampler.sample(4096, DEVICE, torch.Generator().manual_seed(42))
    b = sampler.sample(4096, DEVICE, torch.Generator().manual_seed(42))
    torch.testing.assert_close(a, b)
    assert a.shape == (4096,) and a.dtype == torch.float32 and a.device == DEVICE
    assert bool(((0 <= a) & (a < 1)).all())
    bounded = UniformTimeSampler(0.2, 0.8).sample(256, DEVICE)
    assert bool(((0.2 <= bounded) & (bounded <= 0.8)).all())


def test_beta_range_openpi_recipe_and_explicit_complement():
    sampler = BetaTimeSampler(1.5, 1.0, scale=0.999, offset=0.001)
    torch.manual_seed(29)
    time = sampler.sample(4096, DEVICE)
    assert time.shape == (4096,) and time.dtype == torch.float32
    assert bool(((0.001 <= time) & (time <= 1.0)).all())
    torch.manual_seed(29)
    mirrored = BetaTimeSampler(1.5, 1.0, scale=0.999, offset=0.001, complement=True).sample(4096, DEVICE)
    torch.testing.assert_close(mirrored, 1 - time)
    with pytest.raises(ValueError, match="global torch RNG"):
        sampler.sample(1, DEVICE, torch.Generator())


@pytest.mark.parametrize("sampler", [
    UniformTimeSampler(), BetaTimeSampler(1.5, 1),
])
def test_invalid_sample_request(sampler):
    with pytest.raises(ValueError, match="batch_size"):
        sampler.sample(0, DEVICE)


def test_invalid_time_sampler_configuration():
    with pytest.raises(ValueError):
        UniformTimeSampler(0.7, 0.2)
    with pytest.raises(ValueError):
        BetaTimeSampler(0, 1)
    with pytest.raises(ValueError):
        BetaTimeSampler(1, 1, scale=0.9, offset=0.2)
