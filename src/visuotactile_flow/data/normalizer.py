"""Normalizer interface; statistics must be fitted from the new train split."""

from typing import Protocol, Self

import torch


class FeatureNormalizer(Protocol):
    def fit(self, train_tensor: torch.Tensor) -> Self: ...
    def normalize(self, x: torch.Tensor) -> torch.Tensor: ...
    def denormalize(self, x: torch.Tensor) -> torch.Tensor: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> Self: ...


class MinMaxNormalizer:
    """Per-channel affine mapping, preserving constant channels exactly.

    For nonconstant channels, maps training minimum/maximum to output_min/max.
    A constant channel maps to the output midpoint; its inverse recovers the
    training constant. Values outside the fit range are not clipped.
    """

    def __init__(self, output_min: float = -1.0, output_max: float = 1.0) -> None:
        if not output_min < output_max:
            raise ValueError("output_min must be less than output_max")
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.minimum: torch.Tensor | None = None
        self.maximum: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.offset: torch.Tensor | None = None

    def fit(self, train_tensor: torch.Tensor) -> Self:
        if train_tensor.ndim not in (2, 3) or train_tensor.shape[-1] < 1 or train_tensor.numel() == 0:
            raise ValueError("Expected nonempty [N,D] or [N,T,D] tensor")
        if not torch.isfinite(train_tensor).all():
            raise ValueError("Training tensor contains nonfinite values")
        values = train_tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, train_tensor.shape[-1])
        minimum = values.amin(dim=0)
        maximum = values.amax(dim=0)
        span = maximum - minimum
        constant = span == 0
        scale = torch.where(constant, torch.ones_like(span), (self.output_max - self.output_min) / torch.where(constant, 1, span))
        midpoint = (self.output_min + self.output_max) / 2
        offset = torch.where(constant, midpoint - minimum, self.output_min - scale * minimum)
        self.minimum, self.maximum, self.scale, self.offset = minimum, maximum, scale, offset
        return self

    def _parameters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scale is None or self.offset is None:
            raise RuntimeError("Normalizer has not been fitted or loaded")
        if x.ndim < 1 or x.shape[-1] != len(self.scale) or not x.is_floating_point():
            raise ValueError("Input must be floating point with the fitted channel dimension")
        return self.scale.to(device=x.device, dtype=x.dtype), self.offset.to(device=x.device, dtype=x.dtype)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        scale, offset = self._parameters(x)
        return x * scale + offset

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        scale, offset = self._parameters(x)
        return (x - offset) / scale

    def state_dict(self) -> dict:
        if self.scale is None or self.offset is None or self.minimum is None or self.maximum is None:
            raise RuntimeError("Normalizer has not been fitted or loaded")
        return {"type": "minmax", "version": 1, "output_min": self.output_min,
                "output_max": self.output_max, "minimum": self.minimum.tolist(),
                "maximum": self.maximum.tolist(), "scale": self.scale.tolist(),
                "offset": self.offset.tolist()}

    def load_state_dict(self, state: dict) -> Self:
        if state.get("type") != "minmax" or state.get("version") != 1:
            raise ValueError("Unsupported normalizer state")
        low, high = float(state["output_min"]), float(state["output_max"])
        if not low < high:
            raise ValueError("Invalid output range")
        fields = {key: torch.as_tensor(state[key], dtype=torch.float64) for key in
                  ("minimum", "maximum", "scale", "offset")}
        sizes = {tuple(value.shape) for value in fields.values()}
        if len(sizes) != 1 or len(next(iter(sizes))) != 1 or not all(torch.isfinite(v).all() for v in fields.values()):
            raise ValueError("Invalid normalizer channel statistics")
        if bool(torch.any(fields["scale"] <= 0)) or bool(torch.any(fields["maximum"] < fields["minimum"])):
            raise ValueError("Invalid normalizer scale or limits")
        self.output_min, self.output_max = low, high
        self.minimum, self.maximum, self.scale, self.offset = (fields[k] for k in
            ("minimum", "maximum", "scale", "offset"))
        return self
