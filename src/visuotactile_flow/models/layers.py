"""Shared DiT-style modulation primitive."""

import torch


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]
