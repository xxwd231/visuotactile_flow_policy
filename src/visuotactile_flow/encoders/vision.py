"""Two independent RGB branches; no feature fusion."""

import torch
from torch import nn

from .preprocessing import RGBPreprocessor
from .resnet18 import ResNet18Encoder


class VisionEncoder(nn.Module):
    def __init__(self, *, freeze: bool = True) -> None:
        super().__init__()
        self.preprocess = RGBPreprocessor()
        self.external_rgb = ResNet18Encoder(3, freeze=freeze)
        self.wrist_rgb = ResNet18Encoder(3, freeze=freeze)

    def forward(self, *, external_rgb: torch.Tensor, wrist_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        if external_rgb.shape[:2] != wrist_rgb.shape[:2]:
            raise ValueError("RGB views must share batch and time dimensions")
        result = {}
        for name, frames in (("external_rgb", external_rgb), ("wrist_rgb", wrist_rgb)):
            prepared = self.preprocess(frames)
            b, t = prepared.shape[:2]
            result[name] = getattr(self, name)(prepared.reshape(b * t, 3, 216, 216)).reshape(b, t, 512)
        return result

    def freeze(self) -> "VisionEncoder":
        self.external_rgb.freeze()
        self.wrist_rgb.freeze()
        return self

    def unfreeze(self) -> "VisionEncoder":
        self.external_rgb.unfreeze()
        self.wrist_rgb.unfreeze()
        return self
