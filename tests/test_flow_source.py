import torch

from visuotactile_flow.flow.sources import FlowTensorSpec, GaussianSource


def test_gaussian_source_matches_spec_and_reproduces_with_generator():
    spec = FlowTensorSpec((2, 16, 10), torch.device("cpu"), torch.float64)
    source = GaussianSource()
    first = source.sample(spec, generator=torch.Generator().manual_seed(17))
    second = source.sample(spec, generator=torch.Generator().manual_seed(17))
    assert first.shape == spec.shape
    assert first.dtype == spec.dtype
    assert first.device == spec.device
    torch.testing.assert_close(first, second)


def test_gaussian_source_does_not_require_target():
    spec = FlowTensorSpec((1, 16, 10), torch.device("cpu"), torch.float32)
    assert GaussianSource().sample(spec).device == spec.device
    assert FlowTensorSpec.from_tensor(torch.zeros(spec.shape)).shape == spec.shape
