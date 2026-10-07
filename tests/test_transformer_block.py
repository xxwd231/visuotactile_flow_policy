import torch

from visuotactile_flow.models import TransformerBlock
from visuotactile_flow.models.layers import modulate


def test_modulation_formula_and_zero_init():
    x = torch.ones(2, 3, 4)
    shift = torch.full((2, 4), 3.0)
    scale = torch.full((2, 4), 2.0)
    torch.testing.assert_close(modulate(x, shift, scale), torch.full_like(x, 6.0))
    block = TransformerBlock(64, 4, 2, cross_attention=False)
    assert block.norm_sa.elementwise_affine is False
    assert block.norm_ffn.elementwise_affine is False
    assert torch.count_nonzero(block.modulation[1].weight) == 0
    assert torch.count_nonzero(block.modulation[1].bias) == 0
    assert block.cross_attention is None


def test_bidirectional_self_attention_and_cross_attention():
    torch.manual_seed(8)
    block = TransformerBlock(64, 4, 2, cross_attention=True).eval()
    # Open the SA gate so changes to a later action can affect an earlier action.
    with torch.no_grad():
        block.modulation[1].bias[128:192].fill_(1)
    x = torch.randn(1, 4, 64)
    condition = torch.randn(1, 5, 64)
    time = torch.randn(1, 64)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    a = block(x, time, condition, mask)
    changed = x.clone()
    changed[:, -1, 0] += 2
    b = block(changed, time, condition, mask)
    assert not torch.allclose(a[:, 0], b[:, 0])
    assert block.cross_attention is not None
    assert block.norm_ca is not None
