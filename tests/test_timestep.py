import pytest
import torch

from visuotactile_flow.models import TimestepEmbedder


def test_time_shapes_and_column_equivalence():
    embedder = TimestepEmbedder(hidden_dim=64, time_embedding_dim=32)
    t = torch.tensor([0.0, 0.5, 1.0])
    torch.testing.assert_close(embedder(t), embedder(t[:, None]))
    assert embedder(t).shape == (3, 64)
    assert torch.isfinite(embedder(t)).all()


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_flow_time(value):
    with pytest.raises(ValueError, match="flow_time"):
        TimestepEmbedder(64, 32)(torch.tensor([value]))


def test_cpu_bfloat16_autocast():
    embedder = TimestepEmbedder(64, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = embedder(torch.tensor([0.5], dtype=torch.float32))
    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()
