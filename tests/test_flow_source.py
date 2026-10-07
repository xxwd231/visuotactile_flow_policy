import torch

from visuotactile_flow.flow.sources import GaussianSource


def test_gaussian_source_matches_target_and_reproduces_with_generator():
    target = torch.zeros((2, 16, 10), dtype=torch.float64)
    source = GaussianSource()
    first = source.sample(target, generator=torch.Generator().manual_seed(17))
    second = source.sample(target, generator=torch.Generator().manual_seed(17))
    assert first.shape == target.shape
    assert first.dtype == target.dtype
    assert first.device == target.device
    torch.testing.assert_close(first, second)


def test_gaussian_source_preserves_cpu_device():
    target = torch.zeros((1, 16, 10), device="cpu")
    assert GaussianSource().sample(target).device == target.device
