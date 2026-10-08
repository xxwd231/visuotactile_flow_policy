"""Separate action and robot-state normalization; sections never share statistics."""

from copy import deepcopy
from typing import Self

import torch

from visuotactile_flow.data.schema import DataConfig

from .affine import (FixedRangeNormalizer, IdentityNormalizer, LegacyDPLimitsNormalizer,
                     MeanStdNormalizer, MinMaxNormalizerV2, QuantileNormalizer)
from .base import check_input, fit_values
from .spec import contract_metadata, validate_metadata


MODES = {
    "identity": IdentityNormalizer,
    "mean_std": MeanStdNormalizer,
    "minmax": MinMaxNormalizerV2,
    "legacy_limits": LegacyDPLimitsNormalizer,
    "quantile": QuantileNormalizer,
    "fixed_range": FixedRangeNormalizer,
}


def _create(spec: dict, allowed: set[str]):
    if not isinstance(spec, dict) or spec.get("mode") not in allowed:
        raise ValueError("Unsupported normalization mode")
    mode = spec["mode"]
    kwargs = {key: value for key, value in spec.items() if key != "mode"}
    try:
        return MODES[mode](**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for {mode}") from exc


class _Structured:
    feature = ""
    sections: tuple[tuple[str, slice, set[str]], ...] = ()
    dim = 0

    def __init__(self, modes: dict, *, data_config: DataConfig | None = None,
                 dataset_fingerprint: str | None = None) -> None:
        if not isinstance(modes, dict) or set(modes) != {name for name, _, _ in self.sections}:
            raise ValueError("Wrong normalization sections")
        self.modes = deepcopy(modes)
        self.parts = {name: _create(modes[name], allowed) for name, _, allowed in self.sections}
        self.data_config = data_config or DataConfig()
        self.dataset_fingerprint = dataset_fingerprint
        self.fitted = False

    def fit(self, train_tensor: torch.Tensor) -> Self:
        values = fit_values(train_tensor)
        if values.shape[-1] != self.dim:
            raise ValueError(f"Expected {self.dim} features")
        for name, section, _ in self.sections:
            self.parts[name].fit(train_tensor[..., section])
        self.fitted = True
        return self

    def _transform(self, x: torch.Tensor, method: str) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Structured normalizer is not fitted")
        check_input(x, self.dim)
        return torch.cat([getattr(self.parts[name], method)(x[..., section])
                          for name, section, _ in self.sections], dim=-1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self._transform(x, "normalize")

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self._transform(x, "denormalize")

    def state_dict(self) -> dict:
        if not self.fitted:
            raise RuntimeError("Structured normalizer is not fitted")
        return {"metadata": contract_metadata(self.data_config, self.feature, self.modes,
                                               dataset_fingerprint=self.dataset_fingerprint),
                "sections": {name: self.parts[name].state_dict() for name, _, _ in self.sections}}

    def load_state_dict(self, state: dict) -> Self:
        if not isinstance(state, dict) or set(state) != {"metadata", "sections"}:
            raise ValueError("Invalid structured normalizer state")
        validate_metadata(state["metadata"], self.feature, self.modes)
        if set(state["sections"]) != set(self.parts):
            raise ValueError("Wrong section states")
        for name, section, _ in self.sections:
            self.parts[name].load_state_dict(state["sections"][name])
            if self.parts[name].dimension != section.stop - section.start:
                raise ValueError("Wrong section feature dimension")
        self.data_config = DataConfig(tactile_representation_source=state["metadata"]["tactile_representation_source"])
        self.dataset_fingerprint = state["metadata"].get("dataset_fingerprint")
        self.fitted = True
        return self


class StructuredActionNormalizer(_Structured):
    feature = "action"
    dim = 10
    sections = (
        ("translation", slice(0, 3), {"identity", "mean_std", "minmax", "quantile", "legacy_limits"}),
        ("rotation", slice(3, 9), {"identity", "mean_std", "minmax", "quantile", "legacy_limits"}),
        ("gripper", slice(9, 10), {"identity", "fixed_range", "mean_std", "minmax", "legacy_limits"}),
    )


class StructuredStateNormalizer(_Structured):
    feature = "state"
    dim = 7
    sections = (
        ("joints", slice(0, 6), {"mean_std", "minmax", "quantile", "legacy_limits"}),
        ("gripper", slice(6, 7), {"fixed_range", "minmax", "legacy_limits"}),
    )
