"""Sampling contract for normalized model time."""

from typing import Protocol

import torch


def validate_request(batch_size: int, device: torch.device) -> None:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device")


class TimeSampler(Protocol):
    def sample(
        self, batch_size: int, device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...
