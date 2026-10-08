"""Typed training and diagnostic outputs for velocity matching."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CFMTrainingBatch:
    x_t: torch.Tensor                 # [B,H,D]
    time: torch.Tensor                # float32 [B]
    velocity_target: torch.Tensor     # [B,H,D]
    source: torch.Tensor              # [B,H,D]
    target: torch.Tensor              # normalized action [B,H,D]


@dataclass(frozen=True)
class CFMObjectiveOutput:
    loss: torch.Tensor                # scalar with autograd
    prediction: torch.Tensor          # [B,H,D]
    velocity_target: torch.Tensor     # [B,H,D]
    x_t: torch.Tensor                 # [B,H,D]
    time: torch.Tensor                # float32 [B]
    source: torch.Tensor              # [B,H,D]
