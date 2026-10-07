import pytest
import torch

from visuotactile_flow.condition import (
    ConditionAdapter, ConditionOutput, MODALITY_ORDER, TOKEN_ORDER, Modality,
)


def inputs(batch=2, history=3):
    features = {name: torch.randn(batch, history, 512) for name in MODALITY_ORDER[:-1]}
    features["agent_pos"] = torch.randn(batch, history, 7)
    return features


@pytest.mark.parametrize("history", [1, 3, 4])
def test_shapes_and_default_validity(history):
    output = ConditionAdapter()(**inputs(history=history))
    assert isinstance(output, ConditionOutput)
    assert output.tokens.shape == (2, history * 5, 1024)
    assert output.valid_mask.shape == (2, history * 5)
    assert output.valid_mask.dtype == torch.bool
    assert output.valid_mask.all()
    assert torch.isfinite(output.tokens).all()


def test_explicit_time_major_order_and_ids():
    adapter = ConditionAdapter()
    features = inputs(batch=1, history=3)
    assert TOKEN_ORDER == "time_major"
    assert MODALITY_ORDER == (
        "external_rgb", "wrist_rgb", "tactile_depth_0", "tactile_depth_1", "agent_pos"
    )
    assert [int(m) for m in Modality] == [0, 1, 2, 3, 4]
    output = adapter(**features)
    assert output.modality_ids.tolist() == [0, 1, 2, 3, 4] * 3
    assert output.time_ids.tolist() == [0] * 5 + [1] * 5 + [2] * 5
    # Check actual token values against each specific time and modality branch.
    for time in range(3):
        for modality, name in enumerate(MODALITY_ORDER):
            projection = adapter.state_mlp if name == "agent_pos" else getattr(adapter, f"{name}_projection")
            expected = adapter.final_norm(
                projection(features[name][:, time])
                + adapter.modality_embedding(torch.tensor(modality))
                + adapter.temporal_embedding(torch.tensor(time))
            )
            torch.testing.assert_close(output.tokens[:, 5 * time + modality], expected)


def test_validity_follows_token_order():
    features = inputs()
    tactile1_valid = torch.ones(2, 3, dtype=torch.bool)
    tactile1_valid[:, 1] = False
    output = ConditionAdapter()(**features, tactile_depth_1_valid=tactile1_valid)
    expected = torch.ones(2, 15, dtype=torch.bool)
    expected[:, 8] = False  # t1 tactile_depth_1
    torch.testing.assert_close(output.valid_mask, expected)
    assert torch.isfinite(output.tokens).all()


def test_invalid_inputs():
    adapter = ConditionAdapter()
    base = inputs()
    cases = (
        ("history", {"external_rgb": torch.randn(2, 9, 512)}),
        ("shape", {"wrist_rgb": torch.randn(2, 2, 512)}),
        ("state dimension", {"agent_pos": torch.randn(2, 3, 8)}),
        ("feature dimension", {"tactile_depth_0": torch.randn(2, 3, 511)}),
    )
    for _, changed in cases:
        with pytest.raises(ValueError):
            adapter(**(base | changed))
    with pytest.raises(ValueError, match="valid"):
        adapter(**base, agent_pos_valid=torch.ones(2, 3))
    with pytest.raises(ValueError, match="valid"):
        adapter(**base, agent_pos_valid=torch.ones(2, 2, dtype=torch.bool))


def test_trainable_gradients():
    adapter = ConditionAdapter()
    adapter(**inputs()).tokens.square().mean().backward()
    for name in (*[f"{m}_projection" for m in MODALITY_ORDER[:-1]],
                 "state_mlp", "modality_embedding", "temporal_embedding", "final_norm"):
        module = getattr(adapter, name)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()), name
        assert any(torch.count_nonzero(p.grad) for p in module.parameters()), name
