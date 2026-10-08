from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from torch import nn

from visuotactile_flow.condition import ConditionAdapter
from visuotactile_flow.data.schema import DataConfig
from visuotactile_flow.flow import (
    ConditionalFlowMatchingObjective, EulerSolver, FlowConvention,
    FlowTensorSpec, GaussianSource, LinearConditionalFlowPath, UniformTimeSampler,
)
from visuotactile_flow.models import FlowActionExpert
from visuotactile_flow.normalization import StructuredActionNormalizer, StructuredStateNormalizer
from visuotactile_flow.policy import (
    PolicyConfigBundle, PolicyObservationBatch, VisuotactileFlowPolicy,
    load_policy_checkpoint, save_policy_checkpoint,
)


ACTION_MODES = {
    "translation": {"mode": "mean_std"},
    "rotation": {"mode": "identity"},
    "gripper": {"mode": "fixed_range", "input_min": 0.0, "input_max": 1.0},
}
STATE_MODES = {
    "joints": {"mode": "mean_std"},
    "gripper": {"mode": "fixed_range", "input_min": 0.0, "input_max": 1.0},
}


class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(8))

    def forward(self, *, external_rgb, wrist_rgb):
        def features(rgb):
            value = rgb.float().mean(dim=(2, 3, 4))[..., None] / 255
            return value * self.scale
        return {"external_rgb": features(external_rgb), "wrist_rgb": features(wrist_rgb)}


class FakeTactile(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(8))

    def forward(self, *, tactile_depth_0, tactile_depth_1):
        def features(depth):
            value = depth.mean(dim=(2, 3, 4))[..., None]
            return value * self.scale
        return {
            "tactile_depth_0": features(tactile_depth_0),
            "tactile_depth_1": features(tactile_depth_1),
        }


class RecordingAdapter(ConditionAdapter):
    def forward(self, **kwargs):
        self.last_agent_pos = kwargs["agent_pos"].detach().clone()
        return super().forward(**kwargs)


def make_policy(*, fitted=True, convention=FlowConvention.NOISE_AT_ZERO,
                solver_convention=None, num_steps=2, expert_horizon=16):
    data = DataConfig()
    action_normalizer = StructuredActionNormalizer(ACTION_MODES, data_config=data)
    state_normalizer = StructuredStateNormalizer(STATE_MODES, data_config=data)
    if fitted:
        torch.manual_seed(11)
        action_fit = torch.randn(6, 16, 10) * 0.1
        action_fit[..., 9] = torch.linspace(0.1, 0.9, 6)[:, None]
        state_fit = torch.randn(6, 3, 7) * 0.1 + 2
        state_fit[..., 6] = torch.linspace(0.1, 0.9, 6)[:, None]
        action_normalizer.fit(action_fit)
        state_normalizer.fit(state_fit)
    policy = VisuotactileFlowPolicy(
        vision_encoder=FakeVision(),
        tactile_encoder=FakeTactile(),
        condition_adapter=RecordingAdapter(model_dim=32, encoder_feature_dim=8, state_dim=7),
        flow_expert=FlowActionExpert(
            hidden_dim=32, num_layers=2, num_heads=4, ffn_ratio=2,
            action_horizon=expert_horizon, action_dim=10, time_embedding_dim=32,
        ),
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        cfm_objective=ConditionalFlowMatchingObjective(
            convention, GaussianSource(), LinearConditionalFlowPath(), UniformTimeSampler(),
        ),
        euler_solver=EulerSolver(solver_convention or convention, num_steps=num_steps),
        data_config=data,
    )
    bundle = PolicyConfigBundle(
        model={
            "hidden_dim": 32, "num_layers": 2, "num_heads": 4,
            "action_horizon": 16, "action_dim": 10,
            "time_min_period": 0.004, "time_max_period": 4.0,
        },
        flow={
            "convention": convention.value, "source": {"type": "gaussian"},
            "path": {"type": "linear"},
            "time_sampler": {"type": "uniform", "min_time": 0.0, "max_time": 1.0},
            "objective": {"type": "conditional_flow_matching", "loss": "mse"},
            "solver": {"type": "euler", "num_steps": num_steps},
        },
        data=asdict(data), action_normalization=ACTION_MODES,
        state_normalization=STATE_MODES,
    )
    return policy, bundle


def observation(*, device="cpu", **overrides):
    fields = dict(
        external_rgb=torch.zeros(1, 3, 3, 240, 320, dtype=torch.uint8, device=device),
        wrist_rgb=torch.zeros(1, 3, 3, 240, 320, dtype=torch.uint8, device=device),
        tactile_depth_0=torch.zeros(1, 3, 1, 288, 384, device=device),
        tactile_depth_1=torch.zeros(1, 3, 1, 288, 384, device=device),
        agent_pos=torch.zeros(1, 3, 7, device=device),
    )
    fields.update(overrides)
    return PolicyObservationBatch(**fields)


@pytest.mark.parametrize("field,bad", [
    ("external_rgb", torch.zeros(1, 2, 3, 240, 320, dtype=torch.uint8)),
    ("agent_pos", torch.zeros(1, 3, 6)),
    ("wrist_rgb", torch.zeros(1, 3, 3, 240, 320)),
    ("tactile_depth_0", torch.zeros(1, 3, 1, 288, 384, dtype=torch.int16)),
    ("tactile_depth_1", torch.full((1, 3, 1, 288, 384), 1.1)),
    ("agent_pos", torch.full((1, 3, 7), float("nan"))),
    ("tactile_depth_0", torch.full((1, 3, 1, 288, 384), float("nan"))),
    ("external_rgb_valid", torch.ones(1, 2, dtype=torch.bool)),
    ("wrist_rgb_valid", torch.ones(1, 3)),
])
def test_observation_contract_rejects_invalid_input(field, bad):
    with pytest.raises(ValueError):
        observation(**{field: bad})


def test_valid_observation_and_mask_do_not_sanitize_nan():
    obs = observation(agent_pos_valid=torch.zeros(1, 3, dtype=torch.bool))
    assert obs.agent_pos_valid.shape == (1, 3)
    with pytest.raises(ValueError, match="agent_pos"):
        observation(
            agent_pos=torch.full((1, 3, 7), float("nan")),
            agent_pos_valid=torch.zeros(1, 3, dtype=torch.bool),
        )


def test_policy_constructor_rejects_incompatible_modules():
    with pytest.raises(ValueError, match="H=16"):
        make_policy(expert_horizon=15)
    with pytest.raises(ValueError, match="conventions must match"):
        make_policy(solver_convention=FlowConvention.NOISE_AT_ONE)


def test_condition_normalizes_state_first_and_keeps_time_major_mask():
    policy, _ = make_policy()
    state = torch.full((1, 3, 7), 2.0)
    state[..., 6] = 0.5
    mask = torch.tensor([[False, True, True]])
    obs = observation(agent_pos=state, external_rgb_valid=mask)
    result = policy.encode_condition(obs)
    expected_state = policy.state_normalizer.normalize(state)
    torch.testing.assert_close(policy.condition_adapter.last_agent_pos, expected_state)
    assert result.tokens.shape == (1, 15, 32)
    assert result.valid_mask.shape == (1, 15)
    assert result.valid_mask[0, 0].item() is False
    assert result.valid_mask[0, 5].item() is True
    torch.testing.assert_close(result.time_ids, torch.arange(3).repeat_interleave(5))


def test_condition_casts_normalized_state_and_feature_dtypes_explicitly():
    class DoubleTactile(FakeTactile):
        def forward(self, **kwargs):
            return {name: value.double() for name, value in super().forward(**kwargs).items()}

    policy, _ = make_policy()
    policy.tactile_encoder = DoubleTactile()
    state = torch.full((1, 3, 7), 2.0, dtype=torch.float64)
    state[..., 6] = 0.5
    result = policy.encode_condition(observation(agent_pos=state))
    assert policy.condition_adapter.last_agent_pos.dtype == torch.float32
    assert result.tokens.dtype == torch.float32
    torch.testing.assert_close(
        policy.condition_adapter.last_agent_pos,
        policy.state_normalizer.normalize(state).float(),
    )


def test_training_normalizes_encoded_action_and_backpropagates():
    policy, _ = make_policy()
    target = torch.randn(1, 16, 10) * 0.1
    target[..., 9] = 0.5
    result = policy.compute_training_loss(
        observation(), target, generator=torch.Generator().manual_seed(3),
    )
    torch.testing.assert_close(
        result.normalized_target_action, policy.action_normalizer.normalize(target)
    )
    assert result.x_t.shape == (1, 16, 10)
    assert result.condition_tokens.shape == (1, 15, 32)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    grad = policy.flow_expert.final_output.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0


def test_inference_seeded_source_denormalization_and_eval_guard():
    policy, _ = make_policy()
    with pytest.raises(RuntimeError, match="policy.eval"):
        policy.sample_encoded_action(observation())
    policy.eval()
    result = policy.sample_encoded_action(
        observation(), generator=torch.Generator().manual_seed(5),
        return_intermediates=True,
    )
    initial = GaussianSource().sample(
        FlowTensorSpec((1, 16, 10), torch.device("cpu"), torch.float32),
        generator=torch.Generator().manual_seed(5),
    )
    assert result.normalized_action.shape == result.encoded_action.shape == (1, 16, 10)
    torch.testing.assert_close(result.normalized_action, initial, rtol=0, atol=0)
    torch.testing.assert_close(
        result.encoded_action, policy.action_normalizer.denormalize(result.normalized_action)
    )
    assert result.trajectory.shape == (3, 1, 16, 10)
    assert result.num_function_evaluations == 2
    assert result.normalized_action.dtype == torch.float32
    assert not result.normalized_action.requires_grad


def test_unfitted_normalizers_rejected():
    policy, _ = make_policy(fitted=False)
    with pytest.raises(RuntimeError, match="normalizer must be fitted or loaded"):
        policy.encode_condition(observation())
    policy.eval()
    with pytest.raises(RuntimeError, match="normalizer must be fitted or loaded"):
        policy.sample_encoded_action(observation())


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
                    reason="CUDA bf16 unavailable")
def test_inference_under_external_bf16_autocast_keeps_euler_fp32():
    policy, _ = make_policy()
    policy.cuda().eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = policy.sample_encoded_action(
            observation(device="cuda"),
            generator=torch.Generator(device="cuda").manual_seed(7),
        )
    assert result.normalized_action.dtype == torch.float32
    assert result.encoded_action.dtype == torch.float32
    assert torch.isfinite(result.encoded_action).all()


def test_checkpoint_round_trip_preserves_model_stats_config_and_sample(tmp_path):
    policy, bundle = make_policy()
    policy.eval()
    before = policy.sample_encoded_action(
        observation(), generator=torch.Generator().manual_seed(17)
    )
    path = tmp_path / "nested" / "policy.pt"
    save_policy_checkpoint(
        policy, path, config_bundle=bundle,
        metadata={"dataset_fingerprint": "sha256:test", "encoder_provenance": {
            "source_checkpoint_identifier": "legacy-epoch800",
            "standalone_export_version": 1, "legacy_source_commit": "abc123",
        }},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format_name"] == "visuotactile_flow_policy"
    assert payload["format_version"] == 1
    assert set(payload["model_state_dict"]) == set(policy.state_dict())
    assert any(key.startswith("vision_encoder.") for key in payload["model_state_dict"])
    assert any(key.startswith("tactile_encoder.") for key in payload["model_state_dict"])
    assert any(key.startswith("condition_adapter.") for key in payload["model_state_dict"])
    assert any(key.startswith("flow_expert.") for key in payload["model_state_dict"])
    restored, _ = make_policy(fitted=False)
    info = load_policy_checkpoint(restored, path, expected_config_bundle=bundle)
    assert info.config_bundle.canonical_json() == bundle.canonical_json()
    assert info.metadata["dataset_fingerprint"] == "sha256:test"
    assert restored.action_normalizer.state_dict() == policy.action_normalizer.state_dict()
    assert restored.state_normalizer.state_dict() == policy.state_normalizer.state_dict()
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    restored.eval()
    after = restored.sample_encoded_action(
        observation(), generator=torch.Generator().manual_seed(17)
    )
    torch.testing.assert_close(after.normalized_action, before.normalized_action)
    torch.testing.assert_close(after.encoded_action, before.encoded_action)


@pytest.mark.parametrize("mutation", [
    lambda p: p.__setitem__("format_version", 2),
    lambda p: p["config_bundle"]["flow"].__setitem__("convention", "noise_at_one"),
    lambda p: p["config_bundle"]["model"].__setitem__("hidden_dim", 64),
    lambda p: p["config_bundle"]["model"].__setitem__("num_layers", 3),
    lambda p: p["config_bundle"]["model"].__setitem__("num_heads", 8),
    lambda p: p["config_bundle"]["model"].__setitem__("time_min_period", 0.005),
    lambda p: p["config_bundle"]["model"].__setitem__("time_max_period", 5.0),
    lambda p: p["config_bundle"]["action_normalization"]["translation"].__setitem__("mode", "identity"),
    lambda p: p["config_bundle"]["state_normalization"]["joints"].__setitem__("mode", "minmax"),
    lambda p: p["config_bundle"]["data"].__setitem__("tactile_representation_source", "legacy_video_reconstructed"),
    lambda p: p["config_bundle"]["data"].__setitem__("observation_history", 4),
    lambda p: p.__setitem__("state_normalizer_state", None),
    lambda p: p["action_normalizer_state"].__setitem__("sections", {}),
])
def test_checkpoint_mismatch_and_corruption_rejected(tmp_path, mutation):
    policy, bundle = make_policy()
    source = tmp_path / "source.pt"
    save_policy_checkpoint(policy, source, config_bundle=bundle)
    payload = deepcopy(torch.load(source, map_location="cpu", weights_only=True))
    mutation(payload)
    corrupted = tmp_path / "corrupted.pt"
    torch.save(payload, corrupted)
    target, _ = make_policy(fitted=False)
    with pytest.raises(ValueError):
        load_policy_checkpoint(target, corrupted)


def test_explicit_expected_config_mismatch_and_unfitted_save_rejected(tmp_path):
    policy, bundle = make_policy()
    path = tmp_path / "policy.pt"
    save_policy_checkpoint(policy, path, config_bundle=bundle)
    changed = bundle.to_dict()
    changed["model"]["time_min_period"] = 0.005
    target, _ = make_policy(fitted=False)
    with pytest.raises(ValueError, match="config bundle mismatch"):
        load_policy_checkpoint(
            target, path, expected_config_bundle=PolicyConfigBundle.from_dict(changed)
        )
    unfitted, _ = make_policy(fitted=False)
    with pytest.raises(RuntimeError, match="normalizer must be fitted"):
        save_policy_checkpoint(unfitted, tmp_path / "bad.pt", config_bundle=bundle)
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        save_policy_checkpoint(
            policy, tmp_path / "path-as-identity.pt", config_bundle=bundle,
            metadata={"dataset_fingerprint": "/absolute/dataset/path"},
        )
