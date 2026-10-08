"""Independent feature-wise normalization methods. No tactile preprocessing here."""

import math
from typing import Self

import torch

from .base import check_input, fit_values, read_flags, read_vector, tensor_list


class _Affine:
    kind = ""

    def __init__(self) -> None:
        self.dimension: int | None = None
        self.scale: torch.Tensor | None = None
        self.offset: torch.Tensor | None = None
        self.floored_channels: torch.Tensor | None = None
        self.stats: dict[str, torch.Tensor] = {}

    def _ready(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dimension is None or self.scale is None or self.offset is None:
            raise RuntimeError("Normalizer is not fitted")
        check_input(x, self.dimension)
        return self.scale.to(x), self.offset.to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        scale, offset = self._ready(x)
        return x * scale + offset

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        scale, offset = self._ready(x)
        return (x - offset) / scale

    def state_dict(self) -> dict:
        if self.dimension is None or self.scale is None or self.offset is None or self.floored_channels is None:
            raise RuntimeError("Normalizer is not fitted")
        return {"normalizer_version": 2, "type": self.kind, "dimension": self.dimension,
                "scale": tensor_list(self.scale), "offset": tensor_list(self.offset),
                "floored_channels": tensor_list(self.floored_channels),
                "stats": {key: tensor_list(value) for key, value in self.stats.items()},
                "parameters": self._parameters()}

    def _parameters(self) -> dict:
        return {}

    def _load(self, state: dict, required_stats: tuple[str, ...]) -> None:
        if state.get("normalizer_version") != 2 or state.get("type") != self.kind:
            raise ValueError("Unsupported normalizer state version or type")
        dim = state.get("dimension")
        if type(dim) is not int or dim < 1:
            raise ValueError("Invalid feature dimension")
        self.dimension = dim
        self.scale = read_vector(state, "scale", dim)
        self.offset = read_vector(state, "offset", dim)
        self.floored_channels = read_flags(state, dim)
        if bool(torch.any(self.scale <= 0)):
            raise ValueError("Scale must be positive")
        stats = state.get("stats")
        if not isinstance(stats, dict) or set(stats) != set(required_stats):
            raise ValueError("Invalid statistics")
        self.stats = {key: read_vector(stats, key, dim) for key in required_stats}


class IdentityNormalizer(_Affine):
    kind = "identity"

    def fit(self, train_tensor: torch.Tensor) -> Self:
        values = fit_values(train_tensor)
        self.dimension = values.shape[-1]
        self.scale = torch.ones(self.dimension, dtype=torch.float64)
        self.offset = torch.zeros_like(self.scale)
        self.floored_channels = torch.zeros(self.dimension, dtype=torch.bool)
        return self

    def load_state_dict(self, state: dict) -> Self:
        self._load(state, ())
        if not bool(torch.all(self.scale == 1) and torch.all(self.offset == 0) and not torch.any(self.floored_channels)):
            raise ValueError("Invalid identity transform")
        return self


class MeanStdNormalizer(_Affine):
    kind = "mean_std"

    def __init__(self, std_floor: float = 1e-4) -> None:
        super().__init__()
        if not math.isfinite(std_floor) or std_floor <= 0:
            raise ValueError("std_floor must be positive")
        self.std_floor = float(std_floor)

    def fit(self, train_tensor: torch.Tensor) -> Self:
        values = fit_values(train_tensor)
        mean = values.mean(0)
        raw_std = values.std(0, unbiased=False)
        effective_std = raw_std.clamp_min(self.std_floor)
        self.dimension = values.shape[-1]
        self.floored_channels = raw_std < self.std_floor
        self.scale = 1 / effective_std
        self.offset = -mean / effective_std
        self.stats = {"mean": mean, "raw_std": raw_std, "effective_std": effective_std}
        return self

    def _parameters(self) -> dict:
        return {"std_floor": self.std_floor}

    def load_state_dict(self, state: dict) -> Self:
        if state.get("normalizer_version") != 2 or state.get("type") != self.kind:
            raise ValueError("Unsupported normalizer state version or type")
        self.std_floor = float(state["parameters"]["std_floor"])
        if not math.isfinite(self.std_floor) or self.std_floor <= 0:
            raise ValueError("Invalid std_floor")
        self._load(state, ("mean", "raw_std", "effective_std"))
        if not torch.allclose(self.stats["effective_std"], self.stats["raw_std"].clamp_min(self.std_floor)):
            raise ValueError("Inconsistent standard deviation")
        return self


class MinMaxNormalizerV2(_Affine):
    kind = "minmax_v2"

    def __init__(self, range_eps: float = 1e-4) -> None:
        super().__init__()
        if not math.isfinite(range_eps) or range_eps <= 0:
            raise ValueError("range_eps must be positive")
        self.range_eps = float(range_eps)

    def fit(self, train_tensor: torch.Tensor) -> Self:
        values = fit_values(train_tensor)
        low, high = values.amin(0), values.amax(0)
        span = high - low
        floored = span < self.range_eps
        center = (low + high) / 2
        effective_span = torch.where(floored, torch.ones_like(span) * 2, span)
        self.dimension = values.shape[-1]
        self.floored_channels = floored
        self.scale = 2 / effective_span
        self.offset = torch.where(floored, -center, -1 - self.scale * low)
        self.stats = {"raw_min": low, "raw_max": high, "effective_span": effective_span, "center": center}
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        scale, offset = self._ready(x)
        result = x * scale + offset
        flags = self.floored_channels.to(device=x.device)
        return torch.where(flags, torch.zeros_like(result), result)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        scale, offset = self._ready(x)
        result = (x - offset) / scale
        flags = self.floored_channels.to(device=x.device)
        center = self.stats["center"].to(x)
        return torch.where(flags, center.expand_as(result), result)

    def _parameters(self) -> dict:
        return {"range_eps": self.range_eps}

    def load_state_dict(self, state: dict) -> Self:
        if state.get("normalizer_version") != 2 or state.get("type") != self.kind:
            raise ValueError("Unsupported normalizer state version or type")
        self.range_eps = float(state["parameters"]["range_eps"])
        if not math.isfinite(self.range_eps) or self.range_eps <= 0:
            raise ValueError("Invalid range_eps")
        self._load(state, ("raw_min", "raw_max", "effective_span", "center"))
        if bool(torch.any(self.stats["raw_max"] < self.stats["raw_min"])):
            raise ValueError("Invalid limits")
        return self


class QuantileNormalizer(_Affine):
    kind = "quantile"

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99, quantile_span_floor: float = 1e-4) -> None:
        super().__init__()
        if not (0 <= q_low < q_high <= 1) or not math.isfinite(quantile_span_floor) or quantile_span_floor <= 0:
            raise ValueError("Invalid quantiles or span floor")
        self.q_low, self.q_high = float(q_low), float(q_high)
        self.quantile_span_floor = float(quantile_span_floor)

    def fit(self, train_tensor: torch.Tensor) -> Self:
        values = fit_values(train_tensor)
        low = torch.quantile(values, self.q_low, dim=0)
        high = torch.quantile(values, self.q_high, dim=0)
        span = high - low
        effective_span = span.clamp_min(self.quantile_span_floor)
        self.dimension = values.shape[-1]
        self.floored_channels = span < self.quantile_span_floor
        self.scale = 2 / effective_span
        self.offset = -1 - self.scale * low
        self.stats = {"q_low_value": low, "q_high_value": high, "raw_span": span,
                      "effective_span": effective_span}
        return self

    def _parameters(self) -> dict:
        return {"q_low": self.q_low, "q_high": self.q_high,
                "quantile_span_floor": self.quantile_span_floor}

    def load_state_dict(self, state: dict) -> Self:
        if state.get("normalizer_version") != 2 or state.get("type") != self.kind:
            raise ValueError("Unsupported normalizer state version or type")
        params = state["parameters"]
        self.q_low, self.q_high = float(params["q_low"]), float(params["q_high"])
        self.quantile_span_floor = float(params["quantile_span_floor"])
        if not (0 <= self.q_low < self.q_high <= 1) or self.quantile_span_floor <= 0:
            raise ValueError("Invalid quantile parameters")
        self._load(state, ("q_low_value", "q_high_value", "raw_span", "effective_span"))
        if bool(torch.any(self.stats["raw_span"] < 0)):
            raise ValueError("Invalid quantile span")
        return self


class FixedRangeNormalizer(_Affine):
    kind = "fixed_range"

    def __init__(self, input_min: float = 0.0, input_max: float = 1.0, tolerance: float = 1e-5) -> None:
        super().__init__()
        if not all(math.isfinite(v) for v in (input_min, input_max, tolerance)) or input_min >= input_max or tolerance < 0:
            raise ValueError("Invalid physical range")
        self.input_min, self.input_max, self.tolerance = float(input_min), float(input_max), float(tolerance)

    def fit(self, train_tensor: torch.Tensor) -> Self:
        values = fit_values(train_tensor)
        if bool(torch.any(values < self.input_min - self.tolerance) or torch.any(values > self.input_max + self.tolerance)):
            raise ValueError("Training values exceed declared physical range")
        self.dimension = values.shape[-1]
        self.scale = torch.full((self.dimension,), 2 / (self.input_max - self.input_min), dtype=torch.float64)
        self.offset = torch.full((self.dimension,), -1 - 2 * self.input_min / (self.input_max - self.input_min), dtype=torch.float64)
        self.floored_channels = torch.zeros(self.dimension, dtype=torch.bool)
        self.stats = {"observed_min": values.amin(0), "observed_max": values.amax(0)}
        return self

    def _parameters(self) -> dict:
        return {"input_min": self.input_min, "input_max": self.input_max, "tolerance": self.tolerance}

    def load_state_dict(self, state: dict) -> Self:
        if state.get("normalizer_version") != 2 or state.get("type") != self.kind:
            raise ValueError("Unsupported normalizer state version or type")
        params = state["parameters"]
        self.input_min, self.input_max, self.tolerance = (float(params[k]) for k in
                                                          ("input_min", "input_max", "tolerance"))
        if not self.input_min < self.input_max or self.tolerance < 0:
            raise ValueError("Invalid physical range")
        self._load(state, ("observed_min", "observed_max"))
        return self
