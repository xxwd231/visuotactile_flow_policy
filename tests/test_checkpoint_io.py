import torch
import pytest

from visuotactile_flow.encoders import ResNet18Encoder, load_encoder_weights


def test_standalone_weight_save_load_round_trip(tmp_path):
    original = ResNet18Encoder(1)
    path = tmp_path / "tactile_depth_0.pt"
    torch.save({"format_version": 1, "encoder_name": "tactile_depth_0", "in_channels": 1,
                "output_dim": 512, "state_dict": original.backbone.state_dict()}, path)
    restored = load_encoder_weights(ResNet18Encoder(1), path)
    for key, value in original.backbone.state_dict().items():
        torch.testing.assert_close(restored.backbone.state_dict()[key], value)
    with pytest.raises(ValueError, match="architecture"):
        load_encoder_weights(ResNet18Encoder(3), path)


def test_strict_loader_rejects_missing_keys(tmp_path):
    path = tmp_path / "incomplete.pt"
    torch.save({"format_version": 1, "in_channels": 3, "output_dim": 512,
                "state_dict": {"conv1.weight": torch.zeros(64, 3, 7, 7)}}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_encoder_weights(ResNet18Encoder(3), path)
