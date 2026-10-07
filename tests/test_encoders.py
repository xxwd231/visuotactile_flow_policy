import pytest
import torch
from torch import nn

from visuotactile_flow.encoders import ResNet18Encoder, TactileEncoder, VisionEncoder


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_vision_shape_and_variable_history():
    model = VisionEncoder().eval()
    assert next(model.external_rgb.parameters()) is not next(model.wrist_rgb.parameters())
    with torch.inference_mode():
        for t in (3, 1, 4):
            inputs = torch.randint(0, 256, (2, t, 3, 240, 320), dtype=torch.uint8)
            outputs = model(external_rgb=inputs, wrist_rgb=inputs)
            assert {key: tuple(value.shape) for key, value in outputs.items()} == {
                "external_rgb": (2, t, 512), "wrist_rgb": (2, t, 512)
            }


def test_tactile_shape_independence_and_variable_history():
    model = TactileEncoder(depth_encoding="policy_normalized").eval()
    assert next(model.tactile_depth_0.parameters()) is not next(model.tactile_depth_1.parameters())
    with torch.inference_mode():
        for t in (3, 1, 4):
            depth = torch.rand((2, t, 1, 288, 384))
            outputs = model(tactile_depth_0=depth, tactile_depth_1=depth)
            assert {key: tuple(value.shape) for key, value in outputs.items()} == {
                "tactile_depth_0": (2, t, 512), "tactile_depth_1": (2, t, 512)
            }


@pytest.mark.parametrize("channels", [1, 3])
def test_group_norm_layout_and_freeze_is_sticky(channels):
    model = ResNet18Encoder(channels)
    assert isinstance(model.backbone.fc, nn.Identity)
    assert not any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
    for layer in model.modules():
        if isinstance(layer, nn.GroupNorm):
            assert layer.num_groups == layer.num_channels // 16
            assert layer.affine and layer.eps == 1e-5
    assert model.backbone.conv1.weight.shape[1] == channels
    model.freeze()
    model.train(True)
    assert not model.training and not model.backbone.training
    assert all(not p.requires_grad for p in model.parameters())
    model.unfreeze().train(True)
    assert model.training and all(p.requires_grad for p in model.parameters())


def test_wrappers_keep_frozen_branches_in_eval():
    model = VisionEncoder(freeze=True)
    model.train(True)
    assert not model.external_rgb.training and not model.wrist_rgb.training
    model.unfreeze().train(True)
    assert model.external_rgb.training and model.wrist_rgb.training
