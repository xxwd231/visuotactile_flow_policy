"""Load portable per-branch weights without importing the old repository."""

from pathlib import Path

import torch

from .resnet18 import ResNet18Encoder


def load_encoder_weights(
    encoder: ResNet18Encoder, weight_path: str | Path, *, strict: bool = True
) -> ResNet18Encoder:
    if not isinstance(encoder, ResNet18Encoder):
        raise TypeError("Expected a ResNet18Encoder branch")
    payload = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported standalone encoder weight format")
    if payload.get("in_channels") != encoder.in_channels or payload.get("output_dim") != encoder.output_dim:
        raise ValueError("Weight architecture does not match this encoder")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Standalone weights have no state_dict")
    encoder.backbone.load_state_dict(state, strict=strict)
    return encoder
