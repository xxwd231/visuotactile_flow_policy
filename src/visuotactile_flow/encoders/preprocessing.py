"""Inference preprocessing matching the legacy tactile DP image branches."""

from enum import StrEnum

import torch
from torch import nn
from torchvision import transforms as T


class DepthEncoding(StrEnum):
    RAW_SDK_DEPTH = "raw_sdk_depth"
    POLICY_NORMALIZED = "policy_normalized"


def raw_depth_to_policy_depth(
    raw_depth: torch.Tensor, *, input_encoding: DepthEncoding | str
) -> torch.Tensor:
    """Convert SDK depth once; reject already normalized policy depth."""
    if DepthEncoding(input_encoding) is not DepthEncoding.RAW_SDK_DEPTH:
        raise ValueError("raw_depth_to_policy_depth requires raw_sdk_depth input")
    if not raw_depth.is_floating_point() or not torch.isfinite(raw_depth).all():
        raise ValueError("Raw depth must be finite floating point")
    return torch.clamp(raw_depth / 0.7, -1.0, 1.0)


class RGBPreprocessor(nn.Module):
    """Accept RGB uint8 [B,T,3,H,W] already resized by acquisition to 320×240."""

    def __init__(self) -> None:
        super().__init__()
        self.resize = T.Resize((224, 224))
        self.crop = T.CenterCrop((216, 216))
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("Expected RGB uint8 [B,T,3,H,W]")
        if rgb.shape[0] == 0 or rgb.shape[1] == 0:
            raise ValueError("Batch and history dimensions must be nonempty")
        b, t, _, h, w = rgb.shape
        frames = rgb.reshape(b * t, 3, h, w).float() / 255.0
        frames = self.normalize(self.crop(self.resize(frames)))
        return frames.reshape(b, t, 3, 216, 216)


class TactilePreprocessor(nn.Module):
    """Accept depth [B,T,1,288,384] with an explicit encoding declaration."""

    def __init__(self, depth_encoding: DepthEncoding | str) -> None:
        super().__init__()
        self.depth_encoding = DepthEncoding(depth_encoding)
        self.resize = T.Resize((224, 224))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim != 5 or depth.shape[2:] != (1, 288, 384) or not depth.is_floating_point():
            raise ValueError("Expected floating depth [B,T,1,288,384]")
        if depth.shape[0] == 0 or depth.shape[1] == 0 or not torch.isfinite(depth).all():
            raise ValueError("Depth must have nonempty dimensions and finite values")
        if self.depth_encoding is DepthEncoding.RAW_SDK_DEPTH:
            depth = raw_depth_to_policy_depth(depth, input_encoding=self.depth_encoding)
        elif bool(torch.any(depth < -1.00001)) or bool(torch.any(depth > 1.00001)):
            raise ValueError("policy_normalized depth must lie in [-1,1]")
        b, t = depth.shape[:2]
        return self.resize(depth.reshape(b * t, 1, 288, 384)).reshape(b, t, 1, 224, 224)
