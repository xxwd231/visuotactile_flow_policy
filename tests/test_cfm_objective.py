import pytest
import torch

from visuotactile_flow.flow import (
    ConditionalFlowMatchingObjective, FlowConvention,
    GaussianSource, LinearConditionalFlowPath, UniformTimeSampler,
)
from visuotactile_flow.models import FlowActionExpert


class ConstantExpert:
    def __init__(self, velocity):
        self.velocity = velocity

    def __call__(self, x_t, time, condition_tokens, condition_valid_mask):
        return self.velocity


def objective(convention):
    return ConditionalFlowMatchingObjective(
        convention, GaussianSource(), LinearConditionalFlowPath(), UniformTimeSampler()
    )


def conditions(batch_size=2, hidden_dim=8):
    return torch.zeros(batch_size, 3, hidden_dim), torch.ones(batch_size, 3, dtype=torch.bool)


@pytest.mark.parametrize("convention", list(FlowConvention))
def test_exact_cfm_path_and_zero_loss(convention):
    source = torch.full((2, 4, 3), 2.0)
    target = torch.full((2, 4, 3), 5.0)
    time = torch.tensor([0.25, 0.75])
    obj = objective(convention)
    batch = obj.prepare_training_batch(
        target, source_override=source, time_override=time
    )
    weight = (convention.target_time - time[:, None, None]) * convention.integration_sign
    torch.testing.assert_close(batch.x_t, weight * source + (1 - weight) * target)
    torch.testing.assert_close(batch.velocity_target, (target - source) * convention.integration_sign)
    assert batch.time.dtype == torch.float32
    assert batch.source is source and batch.target is target
    output = obj.compute_loss(
        ConstantExpert(batch.velocity_target), batch, *conditions()
    )
    assert output.loss.item() == 0
    torch.testing.assert_close(output.prediction, batch.velocity_target)


def test_mirrored_convention_path_velocity_and_zero_loss():
    source = torch.randn(2, 4, 3)
    target = source + 2
    time = torch.tensor([0.25, 0.75])
    zero, one = objective(FlowConvention.NOISE_AT_ZERO), objective(FlowConvention.NOISE_AT_ONE)
    forward = zero.prepare_training_batch(target, source_override=source, time_override=time)
    reverse = one.prepare_training_batch(target, source_override=source, time_override=1 - time)
    torch.testing.assert_close(forward.x_t, reverse.x_t)
    torch.testing.assert_close(forward.velocity_target, -reverse.velocity_target)
    for obj, batch in ((zero, forward), (one, reverse)):
        assert obj.compute_loss(ConstantExpert(batch.velocity_target), batch, *conditions()).loss.item() == 0


def test_source_and_time_override_bypass_random_sampling():
    class NoSampling:
        def sample(self, *args, **kwargs):
            raise AssertionError("random sampling was not expected")

    obj = ConditionalFlowMatchingObjective(
        FlowConvention.NOISE_AT_ZERO, NoSampling(), LinearConditionalFlowPath(), NoSampling()
    )
    source = torch.zeros(1, 2, 3)
    target = torch.ones_like(source)
    batch = obj.prepare_training_batch(
        target, source_override=source, time_override=torch.tensor([0.5])
    )
    torch.testing.assert_close(batch.x_t, torch.full_like(target, 0.5))


@pytest.mark.parametrize("source,time", [
    (torch.zeros(1, 4, 3), torch.zeros(2)),
    (torch.zeros(2, 4, 3, dtype=torch.float64), torch.zeros(2)),
    (torch.full((2, 4, 3), float("nan")), torch.zeros(2)),
    (torch.zeros(2, 4, 3), torch.zeros(2, 1)),
    (torch.zeros(2, 4, 3), torch.zeros(2, dtype=torch.float64)),
    (torch.zeros(2, 4, 3), torch.tensor([0.0, 1.1])),
    (torch.zeros(2, 4, 3), torch.tensor([0.0, float("nan")])),
])
def test_invalid_overrides_rejected(source, time):
    with pytest.raises(ValueError):
        objective(FlowConvention.NOISE_AT_ZERO).prepare_training_batch(
            torch.ones(2, 4, 3), source_override=source, time_override=time
        )


def test_invalid_normalized_target_rejected():
    with pytest.raises(ValueError, match="normalized_target_action"):
        objective(FlowConvention.NOISE_AT_ZERO).prepare_training_batch(
            torch.full((1, 4, 3), float("inf"))
        )


def test_action_mask_excludes_invalid_steps_and_normalizes_by_valid_elements():
    obj = objective(FlowConvention.NOISE_AT_ZERO)
    batch = obj.prepare_training_batch(
        torch.zeros(1, 2, 3), source_override=torch.zeros(1, 2, 3),
        time_override=torch.tensor([0.5]),
    )
    mask = torch.tensor([[True, False]])
    cond = conditions(1)
    prediction = torch.tensor([[[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]])
    assert obj.compute_loss(ConstantExpert(prediction), batch, *cond, action_valid_mask=mask).loss.item() == 0
    prediction[:, 0] = 2
    assert obj.compute_loss(ConstantExpert(prediction), batch, *cond, action_valid_mask=mask).loss.item() == 4
    assert obj.compute_loss(ConstantExpert(prediction), batch, *cond).loss.item() > 4
    with pytest.raises(ValueError, match="all invalid"):
        obj.compute_loss(ConstantExpert(prediction), batch, *cond, action_valid_mask=torch.zeros_like(mask))
    with pytest.raises(ValueError, match="action_valid_mask"):
        obj.compute_loss(ConstantExpert(prediction), batch, *cond, action_valid_mask=mask.float())


def test_invalid_expert_output_rejected():
    obj = objective(FlowConvention.NOISE_AT_ZERO)
    batch = obj.prepare_training_batch(
        torch.zeros(1, 2, 3), source_override=torch.zeros(1, 2, 3),
        time_override=torch.tensor([0.5]),
    )
    for wrong in (torch.zeros(1, 2, 2), torch.full((1, 2, 3), float("nan"))):
        with pytest.raises(ValueError, match="prediction"):
            obj.compute_loss(ConstantExpert(wrong), batch, *conditions(1))


def test_real_expert_backward_smoke_with_tiny_architecture():
    torch.manual_seed(15)
    expert = FlowActionExpert(
        hidden_dim=64, num_layers=2, num_heads=4, ffn_ratio=2,
        action_horizon=16, action_dim=10, time_embedding_dim=32,
    )
    obj = objective(FlowConvention.NOISE_AT_ZERO)
    batch = obj.prepare_training_batch(torch.randn(1, 16, 10))
    condition = torch.randn(1, 15, 64)
    valid = torch.ones(1, 15, dtype=torch.bool)
    output = obj.compute_loss(expert, batch, condition, valid)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradient = expert.final_output.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
