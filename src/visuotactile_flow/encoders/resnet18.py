"""Torchvision ResNet18 layout matching the legacy tactile DP backbone.

Audited source: tactile_RL/.../multi_image_obs_encoder.py lines 77-101 and
model_getter.py lines 13-16. No legacy runtime import is used here.
"""

import torch
from torch import nn
from torchvision.models import resnet18


def _replace_batch_norm_with_legacy_group_norm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # GroupNorm defaults match the old constructor: eps=1e-5, affine=True.
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            _replace_batch_norm_with_legacy_group_norm(child)


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels: int, output_dim: int = 512, *, freeze: bool = False) -> None:
        super().__init__()
        if in_channels not in (1, 3) or output_dim != 512:
            raise ValueError("Legacy encoder supports 1 or 3 input channels and 512 outputs")
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        if in_channels == 1:
            old = backbone.conv1
            conv = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride,
                             old.padding, bias=False)
            # Match old initialization; checkpoint loading replaces these weights.
            with torch.no_grad():
                conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = conv
        _replace_batch_norm_with_legacy_group_norm(backbone)
        self.backbone = backbone
        self.in_channels = in_channels
        self.output_dim = output_dim
        self._frozen = False
        if freeze:
            self.freeze()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != self.in_channels:
            raise ValueError(f"Expected [B,{self.in_channels},H,W] preprocessed frames")
        return self.backbone(frames)

    def freeze(self) -> "ResNet18Encoder":
        self._frozen = True
        self.requires_grad_(False)
        super().train(False)
        return self

    def unfreeze(self) -> "ResNet18Encoder":
        self._frozen = False
        self.requires_grad_(True)
        return self

    def train(self, mode: bool = True) -> "ResNet18Encoder":
        super().train(False if self._frozen else mode)
        return self
