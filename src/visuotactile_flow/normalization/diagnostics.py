"""Pure-tensor diagnostics for action channels and flow velocity scale."""

import torch


def _summary(x: torch.Tensor, dim: int | tuple[int, ...]) -> dict[str, torch.Tensor]:
    return {"min": x.amin(dim=dim), "max": x.amax(dim=dim),
            "mean": x.mean(dim=dim), "std": x.std(dim=dim, unbiased=False),
            "q01": torch.quantile(x, 0.01, dim=dim) if isinstance(dim, int) else torch.quantile(x.flatten(0, 1), 0.01, dim=0),
            "q99": torch.quantile(x, 0.99, dim=dim) if isinstance(dim, int) else torch.quantile(x.flatten(0, 1), 0.99, dim=0)}


def analyze_action_distribution(actions: torch.Tensor, normalizer, *, noise_samples: int = 256,
                                generator: torch.Generator | None = None) -> dict:
    if (not isinstance(actions, torch.Tensor) or actions.ndim != 3 or actions.shape[-1] != 10
            or actions.shape[0] < 1 or actions.shape[1] < 1 or not actions.is_floating_point()
            or not bool(torch.isfinite(actions).all())):
        raise ValueError("Expected finite floating [N,H,10] actions")
    if noise_samples < 1:
        raise ValueError("noise_samples must be positive")
    normalized = normalizer.normalize(actions)
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("Normalized actions must be finite")
    raw = _summary(actions, (0, 1))
    norm = _summary(normalized, (0, 1))
    horizon = {f"per_horizon_{key}": value for key, value in _summary(normalized, 0).items()
               if key in {"mean", "std", "q01", "q99"}}
    # Monte Carlo helper uses a bounded subset; the analytic expectation is E[a²] + 1.
    subset = normalized.flatten(0, 1)[:noise_samples]
    noise = torch.randn(subset.shape, device=subset.device, dtype=subset.dtype, generator=generator)
    return {"raw": raw, "normalized": norm,
            "fraction_abs_gt_1": (normalized.abs() > 1).float().mean((0, 1)),
            "fraction_abs_gt_2": (normalized.abs() > 2).float().mean((0, 1)),
            "fraction_abs_gt_3": (normalized.abs() > 3).float().mean((0, 1)),
            **horizon,
            "action_rms_per_channel": actions.square().mean((0, 1)).sqrt(),
            "normalized_action_std_per_channel": norm["std"],
            "source_std": 1.0, "std_ratio": norm["std"],
            "velocity_mse_mc_per_channel": (subset - noise).square().mean(0),
            "velocity_mse_expected_per_channel": normalized.square().mean((0, 1)) + 1}
