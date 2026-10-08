import pytest
import torch

from visuotactile_flow.flow import (
    EulerSolver, FlowConvention, FlowTensorSpec, GaussianSource,
)
from visuotactile_flow.models import FlowActionExpert


@pytest.mark.parametrize("convention", list(FlowConvention))
@pytest.mark.parametrize("num_steps", [1, 2, 10])
def test_constant_velocity_reaches_clean_target(convention, num_steps):
    source = torch.randn(2, 16, 10, dtype=torch.float64)
    target = source + 3
    velocity = (target - source) * convention.integration_sign
    result = EulerSolver(convention, num_steps).solve(
        source, lambda x_t, time: velocity
    )
    torch.testing.assert_close(result.sample, target, atol=1e-6, rtol=0)
    assert result.time_grid.dtype == torch.float32
    assert result.time_grid[0].item() == convention.source_time
    assert result.time_grid[-1].item() == convention.target_time
    assert result.num_function_evaluations == num_steps
    assert result.trajectory is None


def test_mirrored_conventions_have_same_final_sample():
    source = torch.randn(1, 4, 3)
    target = source + 2
    results = []
    for convention in FlowConvention:
        velocity = (target - source) * convention.integration_sign
        results.append(EulerSolver(convention, 4).solve(source, lambda x, t: velocity).sample)
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(results[0], target)


@pytest.mark.parametrize("convention,expected", [
    (FlowConvention.NOISE_AT_ZERO, [0.0, 0.25, 0.5, 0.75]),
    (FlowConvention.NOISE_AT_ONE, [1.0, 0.75, 0.5, 0.25]),
])
def test_exact_nfe_and_evaluation_times(convention, expected):
    seen = []

    def field(x_t, time):
        assert time.shape == (2,) and time.dtype == torch.float32
        seen.append(time.detach().clone())
        return torch.zeros_like(x_t)

    result = EulerSolver(convention, 4).solve(torch.ones(2, 3, 4), field)
    assert len(seen) == result.num_function_evaluations == 4
    torch.testing.assert_close(torch.stack(seen)[:, 0], torch.tensor(expected))
    torch.testing.assert_close(torch.stack(seen)[:, 1], torch.tensor(expected))


def test_ten_steps_make_exactly_ten_field_calls():
    calls = 0

    def field(x_t, time):
        nonlocal calls
        calls += 1
        return torch.zeros_like(x_t)

    result = EulerSolver(FlowConvention.NOISE_AT_ZERO, 10).solve(torch.zeros(1, 2, 3), field)
    assert calls == result.num_function_evaluations == 10


@pytest.mark.parametrize("bad", [
    lambda x: torch.zeros(1, 2, 2),
    lambda x: torch.full_like(x, float("nan")),
    lambda x: torch.zeros_like(x, dtype=torch.float64),
])
def test_invalid_velocity_rejected(bad):
    source = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="velocity"):
        EulerSolver(FlowConvention.NOISE_AT_ZERO).solve(source, lambda x, t: bad(x))


@pytest.mark.parametrize("bad", [
    torch.zeros(0, 2, 3), torch.zeros(1, 2), torch.zeros(1, 2, 3, dtype=torch.int64),
    torch.full((1, 2, 3), float("inf")),
])
def test_invalid_source_rejected(bad):
    with pytest.raises(ValueError, match="initial_source"):
        EulerSolver(FlowConvention.NOISE_AT_ZERO).solve(bad, lambda x, t: x)


def test_trajectory_opt_in_and_default_grad_disabled():
    source = torch.ones(1, 2, 3, requires_grad=True)
    solver = EulerSolver(FlowConvention.NOISE_AT_ZERO, 2)
    result = solver.solve(source, lambda x, t: x)
    assert result.trajectory is None and not result.sample.requires_grad
    untracked = solver.solve(source, lambda x, t: x, return_intermediates=True)
    assert untracked.trajectory is not None and not untracked.trajectory.requires_grad
    tracked = solver.solve(
        source, lambda x, t: x, enable_grad=True, return_intermediates=True
    )
    assert tracked.trajectory.shape == (3, 1, 2, 3)
    torch.testing.assert_close(tracked.trajectory[0], source)
    torch.testing.assert_close(tracked.trajectory[-1], tracked.sample)
    tracked.sample.sum().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()


def test_invalid_step_count():
    with pytest.raises(ValueError, match="num_steps"):
        EulerSolver(FlowConvention.NOISE_AT_ZERO, 0)


def test_full_expert_euler_forward_smoke():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert = FlowActionExpert().to(device).eval()
    source = GaussianSource().sample(
        FlowTensorSpec((1, 16, 10), device, torch.float32),
        generator=torch.Generator(device=device).manual_seed(41),
    )
    condition = torch.randn(1, 15, 1024, device=device)
    mask = torch.ones(1, 15, dtype=torch.bool, device=device)
    result = EulerSolver(FlowConvention.NOISE_AT_ZERO, 2).solve(
        source, lambda x, t: expert(x, t, condition, mask)
    )
    assert result.sample.shape == (1, 16, 10)
    assert torch.isfinite(result.sample).all()
    torch.testing.assert_close(result.sample, source, rtol=0, atol=0)
    assert result.num_function_evaluations == 2
