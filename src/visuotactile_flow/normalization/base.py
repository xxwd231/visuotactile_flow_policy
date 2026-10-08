"""Tensor-only per-feature normalizer contract."""

from typing import Protocol, Self

import torch


class Normalizer(Protocol):
    def fit(self, train_tensor: torch.Tensor) -> Self: ...
    def normalize(self, x: torch.Tensor) -> torch.Tensor: ...
    def denormalize(self, x: torch.Tensor) -> torch.Tensor: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> Self: ...


def fit_values(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.ndim not in (2, 3) or not x.is_floating_point() or x.numel() == 0:
        raise ValueError("Expected nonempty floating [N,D] or [N,T,D]")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Fit data must be finite")
    return x.detach().to(device="cpu", dtype=torch.float64).reshape(-1, x.shape[-1])


def check_input(x: torch.Tensor, dim: int) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim not in (2, 3) or x.shape[-1] != dim or not x.is_floating_point():
        raise ValueError(f"Expected floating [N,{dim}] or [N,T,{dim}]")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Input must be finite")


def tensor_list(x: torch.Tensor) -> list:
    return x.tolist()


def read_vector(state: dict, key: str, dim: int) -> torch.Tensor:
    value = torch.as_tensor(state[key], dtype=torch.float64)
    if value.shape != (dim,) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"Invalid {key}")
    return value


def read_flags(state: dict, dim: int) -> torch.Tensor:
    raw = state["floored_channels"]
    if not isinstance(raw, list) or len(raw) != dim or any(type(v) is not bool for v in raw):
        raise ValueError("Invalid floored_channels")
    return torch.tensor(raw, dtype=torch.bool)
