import json

import torch

from tools.overfit_synthetic import (
    SEED, build_tiny_policy, learned_checkpoint_round_trip,
    make_synthetic_data, parameter_inventory, seeded_generator,
    target_from_condition, train_bounded, write_metrics,
)


def test_synthetic_target_is_deterministic_and_uses_all_modalities():
    data_a = make_synthetic_data(8, torch.device("cpu"))
    data_b = make_synthetic_data(8, torch.device("cpu"))
    torch.testing.assert_close(data_a.condition_scalars, data_b.condition_scalars)
    torch.testing.assert_close(data_a.encoded_target_action, data_b.encoded_target_action)
    baseline = target_from_condition(data_a.condition_scalars)
    for modality in range(5):
        changed = data_a.condition_scalars.clone()
        changed[:, :, modality] += 0.1
        assert not torch.allclose(target_from_condition(changed), baseline, atol=1e-6, rtol=0)
    assert not torch.allclose(baseline[:, 0], baseline[:, -1])
    assert bool((baseline[..., 9] >= 0).all() and (baseline[..., 9] <= 1).all())
    assert bool((baseline[..., :9].std(dim=(0, 1)) > 0).all())


def test_tiny_policy_one_update_and_euler_are_finite():
    data = make_synthetic_data(8, torch.device("cpu"))
    policy, _ = build_tiny_policy(data)
    inventory = parameter_inventory(policy)
    assert inventory["total_trainable_params"] == (
        inventory["condition_adapter_params"] + inventory["flow_expert_params"]
    )
    assert inventory["fake_encoder_trainable_params"] == 0
    training = train_bounded(
        policy, data.observations, data.encoded_target_action,
        steps=1, phase="test", generator=seeded_generator(torch.device("cpu"), SEED + 9),
    )
    assert training["all_finite"] and training["max_grad_norm"] >= 0
    policy.eval()
    output = policy.sample_encoded_action(
        data.observations, generator=seeded_generator(torch.device("cpu"), SEED + 10),
    )
    assert output.normalized_action.shape == (8, 16, 10)
    assert output.num_function_evaluations == 10
    assert torch.isfinite(output.normalized_action).all()


def test_metrics_serialization_and_learned_checkpoint_after_one_update(tmp_path):
    data = make_synthetic_data(8, torch.device("cpu"))
    policy, bundle = build_tiny_policy(data)
    train_bounded(
        policy, data.observations, data.encoded_target_action,
        steps=1, phase="test", generator=seeded_generator(torch.device("cpu"), SEED + 11),
    )
    metrics_path = tmp_path / "metrics.json"
    write_metrics(metrics_path, {"seed": SEED, "phase_a": {"passed": True}})
    assert json.loads(metrics_path.read_text())["seed"] == SEED
    assert learned_checkpoint_round_trip(
        policy, bundle, data, tmp_path / "tiny_policy.pt", SEED + 12,
    )
