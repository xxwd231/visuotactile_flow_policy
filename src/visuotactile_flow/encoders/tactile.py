"""Two independent single-channel depth branches; no feature fusion."""

import torch
from torch import nn

from .preprocessing import DepthEncoding, TactilePreprocessor
from .resnet18 import ResNet18Encoder


class TactileEncoder(nn.Module):
    def __init__(self, *, depth_encoding: DepthEncoding | str, freeze: bool = True) -> None:
        super().__init__()
        self.preprocess = TactilePreprocessor(depth_encoding)
        self.tactile_depth_0 = ResNet18Encoder(1, freeze=freeze)
        self.tactile_depth_1 = ResNet18Encoder(1, freeze=freeze)

    def forward(self, *, tactile_depth_0: torch.Tensor,
                tactile_depth_1: torch.Tensor) -> dict[str, torch.Tensor]:
        if tactile_depth_0.shape[:2] != tactile_depth_1.shape[:2]:
            raise ValueError("Tactile views must share batch and time dimensions")
        result = {}
        for name, frames in (("tactile_depth_0", tactile_depth_0),
                             ("tactile_depth_1", tactile_depth_1)):
            prepared = self.preprocess(frames)
            b, t = prepared.shape[:2]
            result[name] = getattr(self, name)(prepared.reshape(b * t, 1, 224, 224)).reshape(b, t, 512)
        return result

    def freeze(self) -> "TactileEncoder":
        self.tactile_depth_0.freeze()
        self.tactile_depth_1.freeze()
        return self

    def unfreeze(self) -> "TactileEncoder":
        self.tactile_depth_0.unfreeze()
        self.tactile_depth_1.unfreeze()
        return self
