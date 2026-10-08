import pytest
import torch
import yaml
from pathlib import Path

from visuotactile_flow.models import FlowActionExpert


def tiny(**overrides):
    config = dict(hidden_dim=64, num_layers=4, num_heads=4, ffn_ratio=2,
                  cross_attention_every=2, action_horizon=16, action_dim=10,
                  time_embedding_dim=32)
    return FlowActionExpert(**(config | overrides))


def inputs(batch=2, n=15):
    return (torch.randn(batch, 16, 10), torch.rand(batch),
            torch.randn(batch, n, 64), torch.ones(batch, n, dtype=torch.bool))


@pytest.mark.parametrize("n", [5, 15, 20])
def test_forward_variable_condition_length_and_zero_output(n):
    model = tiny()
    output = model(*inputs(n=n))
    assert output.shape == (2, 16, 10)
    assert torch.count_nonzero(output) == 0


def test_full_config_cross_attention_placement_using_tiny_width():
    model = tiny(num_layers=14)
    assert list(model.cross_attention_block_indices) == [1, 3, 5, 7, 9, 11, 13]
    assert [i for i, block in enumerate(model.blocks) if block.cross_attention is not None] == [1, 3, 5, 7, 9, 11, 13]


def test_time_vector_and_column_match():
    model = tiny()
    action, time, condition, mask = inputs()
    torch.testing.assert_close(model(action, time, condition, mask),
                               model(action, time[:, None], condition, mask))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_time(value):
    action, _, condition, mask = inputs()
    with pytest.raises(ValueError, match="flow_time"):
        tiny()(action, torch.full((2,), value), condition, mask)


def test_shape_dtype_finite_and_validity_validation():
    model = tiny()
    action, time, condition, mask = inputs()
    bad_cases = (
        (action[:, :15], time, condition, mask),
        (action[:, :, :9], time, condition, mask),
        (action, time, condition[:, :, :63], mask),
        (action, time, condition, mask[:, :14]),
        (action, time, condition, mask.float()),
        (action, time, condition[:, :0], mask[:, :0]),
        (action, time, condition[:1], mask[:1]),
        (action, time, condition, torch.zeros_like(mask)),
        (action, time, condition.masked_fill(torch.ones_like(condition, dtype=torch.bool), float("nan")), mask),
    )
    for case in bad_cases:
        with pytest.raises(ValueError):
            model(*case)
    partial_mask = mask.clone()
    partial_mask[1] = False
    with pytest.raises(ValueError, match="Each sample must contain at least one valid condition token"):
        model(action, time, condition, partial_mask)


def test_mask_semantics_on_nonzero_output_head():
    torch.manual_seed(12)
    model = tiny().eval()
    with torch.no_grad():
        model.final_output.weight.normal_(std=0.02)
    action, time, condition, mask = inputs(batch=1, n=5)
    mask[:, 2] = False
    with torch.no_grad():
        baseline = model(action, time, condition, mask)
        changed_invalid = condition.clone()
        changed_invalid[:, 2] += 1e4
        ignored = model(action, time, changed_invalid, mask)
        changed_valid = condition.clone()
        changed_valid[:, 1] += 10
        used = model(action, time, changed_valid, mask)
    torch.testing.assert_close(baseline, ignored)
    assert not torch.allclose(baseline, used, atol=1e-6, rtol=1e-6)


def test_backward_zero_head_then_open_head():
    torch.manual_seed(13)
    model = tiny()
    batch = inputs()
    loss = model(*batch).square().mean() + model(*batch).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.final_output.weight.grad is not None
    assert torch.isfinite(model.final_output.weight.grad).all()
    assert torch.count_nonzero(model.final_output.weight.grad) > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.final_output.weight.normal_(std=0.02)
    model(*batch).square().mean().backward()
    for parameter in (model.action_input_projection.weight,
                      model.blocks[1].cross_attention.in_proj_weight,
                      model.blocks[0].modulation[1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_cpu_bfloat16_autocast_forward():
    model = tiny().eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(*inputs(batch=1))
    assert output.shape == (1, 16, 10)
    assert torch.isfinite(output).all()


def test_full_yaml_direct_constructor_and_parameter_count():
    config_path = Path(__file__).resolve().parents[1] / "configs/model/flow_300m.yaml"
    config = yaml.safe_load(config_path.read_text())
    with torch.device("meta"):
        model = FlowActionExpert(**config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 297_309_194
