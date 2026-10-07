import torch

from visuotactile_flow.condition import ConditionAdapter
from visuotactile_flow.encoders import TactileEncoder, VisionEncoder


def test_synthetic_encoder_to_condition_pipeline():
    torch.manual_seed(42)
    vision = VisionEncoder().eval()
    tactile = TactileEncoder(depth_encoding="policy_normalized").eval()
    adapter = ConditionAdapter().eval()
    rgb = torch.randint(0, 256, (1, 3, 3, 240, 320), dtype=torch.uint8)
    depth = torch.rand(1, 3, 1, 288, 384) * 2 - 1
    state = torch.randn(1, 3, 7)
    with torch.no_grad():
        features = vision(external_rgb=rgb, wrist_rgb=rgb)
        features.update(tactile(tactile_depth_0=depth, tactile_depth_1=depth))
        result = adapter(**features, agent_pos=state)
    assert all(value.shape == (1, 3, 512) for value in features.values())
    assert result.tokens.shape == (1, 15, 1024)
    assert result.valid_mask.shape == (1, 15)
    assert torch.isfinite(result.tokens).all()
