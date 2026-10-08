import pytest
import torch

from visuotactile_flow.flow import FlowConvention, LinearConditionalFlowPath


def samples():
    source = torch.arange(2 * 16 * 10, dtype=torch.float64).reshape(2, 16, 10)
    target = source + 3
    return source, target


def test_convention_endpoints_and_sign():
    zero, one = FlowConvention.NOISE_AT_ZERO, FlowConvention.NOISE_AT_ONE
    assert (zero.source_time, zero.target_time, zero.integration_sign) == (0, 1, 1)
    assert (one.source_time, one.target_time, one.integration_sign) == (1, 0, -1)


@pytest.mark.parametrize("convention", list(FlowConvention))
def test_linear_path_endpoints(convention):
    source, target = samples()
    path = LinearConditionalFlowPath()
    at_source = path.sample(source, target, torch.full((2,), convention.source_time), convention)
    at_target = path.sample(source, target, torch.full((2,), convention.target_time), convention)
    torch.testing.assert_close(at_source.x_t, source, rtol=0, atol=0)
    torch.testing.assert_close(at_target.x_t, target, rtol=0, atol=0)
    torch.testing.assert_close(at_source.velocity_target, (target - source) * convention.integration_sign)
    assert at_source.time.shape == (2,)


def test_mirrored_path_and_velocity_sign():
    source, target = samples()
    time = torch.tensor([0.2, 0.75], dtype=torch.float64)
    path = LinearConditionalFlowPath()
    forward = path.sample(source, target, time, FlowConvention.NOISE_AT_ZERO)
    reverse = path.sample(source, target, 1 - time, FlowConvention.NOISE_AT_ONE)
    torch.testing.assert_close(forward.x_t, reverse.x_t)
    torch.testing.assert_close(forward.velocity_target, -reverse.velocity_target)


@pytest.mark.parametrize("time", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_time_rejected(time):
    source, target = samples()
    with pytest.raises(ValueError, match="time"):
        LinearConditionalFlowPath().sample(
            source, target, torch.tensor([time, time]), FlowConvention.NOISE_AT_ZERO
        )


def test_path_shape_dtype_and_device_validation():
    source, target = samples()
    path = LinearConditionalFlowPath()
    with pytest.raises(ValueError, match="shape"):
        path.sample(source, target[:, :-1], torch.zeros(2), FlowConvention.NOISE_AT_ZERO)
    with pytest.raises(ValueError, match="dtype"):
        path.sample(source, target.float(), torch.zeros(2), FlowConvention.NOISE_AT_ZERO)
    with pytest.raises(ValueError, match="time"):
        path.sample(source, target, torch.zeros(2, 1), FlowConvention.NOISE_AT_ZERO)
