"""Explicit Euler integration along a FlowConvention's endpoints."""

import torch

from ..convention import FlowConvention
from .base import EulerResult, VelocityField


class EulerSolver:
    def __init__(self, convention: FlowConvention, num_steps: int = 10) -> None:
        if not isinstance(convention, FlowConvention):
            raise ValueError("convention must be FlowConvention")
        if type(num_steps) is not int or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        self.convention = convention
        self.num_steps = num_steps

    def solve(
        self, initial_source: torch.Tensor, velocity_fn: VelocityField, *,
        enable_grad: bool = False, return_intermediates: bool = False,
    ) -> EulerResult:
        if (not isinstance(initial_source, torch.Tensor) or initial_source.ndim != 3
                or any(dim < 1 for dim in initial_source.shape)
                or not initial_source.is_floating_point()
                or not bool(torch.isfinite(initial_source).all())):
            raise ValueError("initial_source must be finite floating [B,H,D]")
        time_grid = torch.linspace(
            self.convention.source_time, self.convention.target_time,
            self.num_steps + 1, device=initial_source.device, dtype=torch.float32,
        )
        with torch.set_grad_enabled(enable_grad):
            x_t = initial_source
            trajectory = [x_t] if return_intermediates else None
            for index in range(self.num_steps):
                time_batch = time_grid[index].expand(x_t.shape[0])
                velocity = velocity_fn(x_t, time_batch)
                if (not isinstance(velocity, torch.Tensor) or velocity.shape != x_t.shape
                        or velocity.device != x_t.device or velocity.dtype != x_t.dtype
                        or not bool(torch.isfinite(velocity).all())):
                    raise ValueError("velocity must match source shape, device and dtype and be finite")
                dt = (time_grid[index + 1] - time_grid[index]).to(dtype=x_t.dtype)
                x_t = x_t + dt * velocity
                if trajectory is not None:
                    trajectory.append(x_t)
            stacked_trajectory = torch.stack(trajectory) if trajectory is not None else None
        return EulerResult(
            sample=x_t, time_grid=time_grid,
            num_function_evaluations=self.num_steps,
            trajectory=stacked_trajectory,
        )
