"""Bounded synthetic CFM learning gate for a tiny, fully wired Stage-1 policy.

This is an experiment, not a dataset loader or production training framework.
"""

import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visuotactile_flow.condition import ConditionAdapter  # noqa: E402
from visuotactile_flow.data.schema import DataConfig  # noqa: E402
from visuotactile_flow.flow import (  # noqa: E402
    ConditionalFlowMatchingObjective, EulerSolver, FlowConvention,
    GaussianSource, LinearConditionalFlowPath, UniformTimeSampler,
)
from visuotactile_flow.models import FlowActionExpert  # noqa: E402
from visuotactile_flow.normalization import (  # noqa: E402
    StructuredActionNormalizer, StructuredStateNormalizer,
)
from visuotactile_flow.policy import (  # noqa: E402
    PolicyConfigBundle, PolicyObservationBatch, VisuotactileFlowPolicy,
    load_policy_checkpoint, save_policy_checkpoint,
)


SEED = 20261008
PHASE_A_MAX_STEPS = 600
PHASE_B_MAX_STEPS = 1500
LR = 1e-3
GRAD_CLIP = 1.0
HORIZON = 16
ACTION_DIM = 10
CONDITION_DIM = 8
ACTION_MODES = {
    "translation": {"mode": "mean_std"},
    "rotation": {"mode": "mean_std"},
    "gripper": {"mode": "fixed_range", "input_min": 0.0, "input_max": 1.0},
}
STATE_MODES = {
    "joints": {"mode": "mean_std"},
    "gripper": {"mode": "fixed_range", "input_min": 0.0, "input_max": 1.0},
}
TINY_MODEL_CONFIG = {
    "hidden_dim": 64, "num_layers": 4, "num_heads": 4, "ffn_ratio": 2,
    "action_horizon": HORIZON, "action_dim": ACTION_DIM,
    "cross_attention_every": 1, "time_embedding_dim": 64,
    "time_min_period": 0.004, "time_max_period": 4.0,
    "dropout": 0.0, "attention_dropout": 0.0, "adaln_zero": True,
}
FLOW_CONFIG = {
    "convention": "noise_at_zero", "source": {"type": "gaussian"},
    "path": {"type": "linear"},
    "time_sampler": {"type": "uniform", "min_time": 0.0, "max_time": 1.0},
    "objective": {"type": "conditional_flow_matching", "loss": "mse"},
    "solver": {"type": "euler", "num_steps": 10},
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


class TinyVisionEncoder(nn.Module):
    """Read one RGB pixel per frame and apply fixed, branch-specific 8D maps."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("external_basis", torch.tensor(
            [0.7, -0.3, 0.9, -0.5, 0.4, 1.1, -0.8, 0.6]
        ))
        self.register_buffer("wrist_basis", torch.tensor(
            [-0.6, 1.0, 0.5, -0.9, 0.8, -0.4, 0.3, 0.7]
        ))

    def forward(self, *, external_rgb: torch.Tensor, wrist_rgb: torch.Tensor) -> dict:
        external = external_rgb[:, :, 0, 0, 0].float() / 127.5 - 1.0
        wrist = wrist_rgb[:, :, 0, 0, 0].float() / 127.5 - 1.0
        return {
            "external_rgb": external[..., None] * self.external_basis,
            "wrist_rgb": wrist[..., None] * self.wrist_basis,
        }


class TinyTactileEncoder(nn.Module):
    """Read one normalized depth pixel per frame and apply fixed 8D maps."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("depth_0_basis", torch.tensor(
            [0.8, 0.2, -0.7, 1.0, -0.5, 0.6, -0.9, 0.4]
        ))
        self.register_buffer("depth_1_basis", torch.tensor(
            [-0.4, 0.9, 0.6, -0.2, 1.0, -0.8, 0.5, -0.7]
        ))

    def forward(self, *, tactile_depth_0: torch.Tensor, tactile_depth_1: torch.Tensor) -> dict:
        depth_0 = tactile_depth_0[:, :, 0, 0, 0]
        depth_1 = tactile_depth_1[:, :, 0, 0, 0]
        return {
            "tactile_depth_0": depth_0[..., None] * self.depth_0_basis,
            "tactile_depth_1": depth_1[..., None] * self.depth_1_basis,
        }


@dataclass(frozen=True)
class SyntheticData:
    observations: PolicyObservationBatch
    encoded_target_action: torch.Tensor
    condition_scalars: torch.Tensor  # [N,3,5] as actually observable


def target_from_condition(condition_scalars: torch.Tensor) -> torch.Tensor:
    """Fixed deterministic mapping depending on all modalities and horizon."""
    if condition_scalars.ndim != 3 or condition_scalars.shape[1:] != (3, 5):
        raise ValueError("condition_scalars must have shape [N,3,5]")
    generator = torch.Generator(device="cpu").manual_seed(SEED + 101)
    weights = (torch.randn(15, 9, generator=generator) * 0.34).to(condition_scalars.device)
    bias = (torch.randn(9, generator=generator) * 0.08).to(condition_scalars.device)
    grip_weights = (torch.randn(15, generator=generator) * 0.42).to(condition_scalars.device)
    flattened = condition_scalars.reshape(condition_scalars.shape[0], 15)
    base = 0.3 * torch.tanh(flattened @ weights + bias)
    horizon = torch.arange(HORIZON, device=condition_scalars.device, dtype=torch.float32)
    channel = torch.arange(9, device=condition_scalars.device, dtype=torch.float32)
    horizon_term = 0.035 * torch.sin(
        2 * torch.pi * (horizon[:, None] + 1) * (1 + channel[None, :] / 15) / HORIZON
    )
    physical = base[:, None, :] + horizon_term[None, :, :]
    grip_logit = flattened @ grip_weights
    gripper = torch.sigmoid(
        grip_logit[:, None] + 0.24 * (horizon[None, :] / (HORIZON - 1) - 0.5)
    )
    return torch.cat((physical, gripper[..., None]), dim=-1)


def make_synthetic_data(count: int, device: torch.device) -> SyntheticData:
    if count < 1:
        raise ValueError("count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(SEED + 11)
    scalars = (2 * torch.rand(count, 3, 5, generator=generator) - 1).to(device)
    rgb_external = ((scalars[:, :, 0] + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    rgb_wrist = ((scalars[:, :, 1] + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    observed = scalars.clone()
    observed[:, :, 0] = rgb_external.float() / 127.5 - 1
    observed[:, :, 1] = rgb_wrist.float() / 127.5 - 1
    external_rgb = torch.zeros(count, 3, 3, 240, 320, dtype=torch.uint8, device=device)
    wrist_rgb = torch.zeros_like(external_rgb)
    external_rgb[:, :, 0, 0, 0] = rgb_external
    wrist_rgb[:, :, 0, 0, 0] = rgb_wrist
    depth_0 = torch.zeros(count, 3, 1, 288, 384, device=device)
    depth_1 = torch.zeros_like(depth_0)
    depth_0[:, :, 0, 0, 0] = observed[:, :, 2]
    depth_1[:, :, 0, 0, 0] = observed[:, :, 3]
    state = torch.zeros(count, 3, 7, device=device)
    state[:, :, 0] = observed[:, :, 4]
    state[:, :, 6] = 0.5
    observations = PolicyObservationBatch(
        external_rgb=external_rgb, wrist_rgb=wrist_rgb,
        tactile_depth_0=depth_0, tactile_depth_1=depth_1, agent_pos=state,
    )
    return SyntheticData(observations, target_from_condition(observed), observed)


def select_observations(
    observations: PolicyObservationBatch, indices: torch.Tensor,
) -> PolicyObservationBatch:
    return PolicyObservationBatch(**{
        field.name: (
            value.index_select(0, indices) if value is not None else None
        )
        for field in fields(PolicyObservationBatch)
        if (value := getattr(observations, field.name)) is not None
    })


def build_tiny_policy(
    train_data: SyntheticData, *, fit_normalizers: bool = True,
    initialization_seed: int = SEED + 201,
) -> tuple[VisuotactileFlowPolicy, PolicyConfigBundle]:
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    data_config = DataConfig()
    action_normalizer = StructuredActionNormalizer(ACTION_MODES, data_config=data_config)
    state_normalizer = StructuredStateNormalizer(STATE_MODES, data_config=data_config)
    if fit_normalizers:
        # Fit only the complete synthetic train set, never an evaluation batch.
        action_normalizer.fit(train_data.encoded_target_action)
        state_normalizer.fit(train_data.observations.agent_pos)
    convention = FlowConvention.NOISE_AT_ZERO
    policy = VisuotactileFlowPolicy(
        vision_encoder=TinyVisionEncoder(),
        tactile_encoder=TinyTactileEncoder(),
        condition_adapter=ConditionAdapter(
            model_dim=64, encoder_feature_dim=CONDITION_DIM, state_dim=7, max_history=3,
        ),
        flow_expert=FlowActionExpert(**TINY_MODEL_CONFIG),
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        cfm_objective=ConditionalFlowMatchingObjective(
            convention, GaussianSource(), LinearConditionalFlowPath(),
            UniformTimeSampler(),
        ),
        euler_solver=EulerSolver(convention, num_steps=10),
        data_config=data_config,
    )
    bundle = PolicyConfigBundle(
        model=TINY_MODEL_CONFIG, flow=FLOW_CONFIG, data=asdict(data_config),
        action_normalization=ACTION_MODES, state_normalization=STATE_MODES,
    )
    return policy.to(train_data.encoded_target_action.device), bundle


def parameter_inventory(policy: VisuotactileFlowPolicy) -> dict[str, int]:
    adapter = sum(parameter.numel() for parameter in policy.condition_adapter.parameters())
    expert = sum(parameter.numel() for parameter in policy.flow_expert.parameters())
    fake_encoder_trainable = sum(
        parameter.numel() for encoder in (policy.vision_encoder, policy.tactile_encoder)
        for parameter in encoder.parameters() if parameter.requires_grad
    )
    all_trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    if fake_encoder_trainable != 0 or all_trainable != adapter + expert:
        raise RuntimeError("Only ConditionAdapter and FlowActionExpert may be trainable")
    return {
        "condition_adapter_params": adapter,
        "flow_expert_params": expert,
        "total_trainable_params": all_trainable,
        "fake_encoder_trainable_params": fake_encoder_trainable,
    }


def fixed_cfm_loss(
    policy: VisuotactileFlowPolicy, observations: PolicyObservationBatch,
    encoded_target_action: torch.Tensor,
    fixed_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    was_training = policy.training
    policy.eval()
    with torch.inference_mode():
        losses = [
            policy.compute_training_loss(
                observations, encoded_target_action,
                source_override=source, time_override=time,
            ).loss.item()
            for source, time in fixed_pairs
        ]
    policy.train(was_training)
    return statistics.mean(losses)


def fixed_source_euler_rmse(
    policy: VisuotactileFlowPolicy, observations: PolicyObservationBatch,
    normalized_target_action: torch.Tensor, source: torch.Tensor,
) -> float:
    policy.eval()
    with torch.inference_mode():
        condition = policy.encode_condition(observations)
        result = policy.euler_solver.solve(
            source,
            lambda x, t: policy.flow_expert(
                x, t, condition.tokens, condition.valid_mask,
            ).to(dtype=x.dtype),
        )
        return torch.mean((result.sample - normalized_target_action).square()).sqrt().item()


def seeded_euler_metrics(
    policy: VisuotactileFlowPolicy, observations: PolicyObservationBatch,
    normalized_target_action: torch.Tensor, seeds: tuple[int, ...],
) -> dict:
    policy.eval()
    with torch.inference_mode():
        samples = torch.stack([
            policy.sample_encoded_action(
                observations,
                generator=seeded_generator(normalized_target_action.device, seed),
            ).normalized_action
            for seed in seeds
        ])
        squared = (samples - normalized_target_action[None]).square()
        per_channel = squared.mean(dim=(0, 1, 2)).sqrt()
        return {
            "rmse": squared.mean().sqrt().item(),
            "per_channel_rmse": per_channel.tolist(),
            "translation_rmse": squared[..., :3].mean().sqrt().item(),
            "rotation_rmse": squared[..., 3:9].mean().sqrt().item(),
            "gripper_rmse": squared[..., 9:].mean().sqrt().item(),
        }


def condition_permutation_metrics(
    policy: VisuotactileFlowPolicy, observations: PolicyObservationBatch,
    normalized_target_action: torch.Tensor, seed: int,
) -> tuple[float, float]:
    policy.eval()
    device = normalized_target_action.device
    indices = torch.arange(normalized_target_action.shape[0], device=device)
    shifted = select_observations(observations, torch.roll(indices, shifts=1))
    with torch.inference_mode():
        correct = policy.sample_encoded_action(
            observations, generator=seeded_generator(device, seed),
        ).normalized_action
        shuffled = policy.sample_encoded_action(
            shifted, generator=seeded_generator(device, seed),
        ).normalized_action
        correct_rmse = (correct - normalized_target_action).square().mean().sqrt().item()
        shuffled_rmse = (shuffled - normalized_target_action).square().mean().sqrt().item()
    return correct_rmse, shuffled_rmse


def _all_parameters_finite(parameters: list[nn.Parameter]) -> bool:
    with torch.no_grad():
        return bool(torch.stack([torch.isfinite(parameter).all() for parameter in parameters]).all())


def train_bounded(
    policy: VisuotactileFlowPolicy,
    observations: PolicyObservationBatch,
    encoded_target_action: torch.Tensor,
    *, steps: int, phase: str, generator: torch.Generator,
    source_override: torch.Tensor | None = None,
    evaluation=None,
) -> dict:
    parameters = list(policy.condition_adapter.parameters()) + list(policy.flow_expert.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=0.0)
    gradient_norms: list[float] = []
    curve: list[dict] = []
    loss_ema = None
    for step in range(1, steps + 1):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        output = policy.compute_training_loss(
            observations, encoded_target_action,
            generator=generator, source_override=source_override,
        )
        if not bool(torch.isfinite(output.loss)):
            raise FloatingPointError(f"{phase} step {step}: non-finite loss")
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"{phase} step {step}: non-finite gradient norm")
        optimizer.step()
        if not _all_parameters_finite(parameters):
            raise FloatingPointError(f"{phase} step {step}: non-finite parameters")
        loss_value = output.loss.detach().item()
        grad_value = grad_norm.detach().item()
        gradient_norms.append(grad_value)
        loss_ema = loss_value if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_value
        if step % 100 == 0 or step == steps:
            point = {
                "step": step, "train_loss": loss_value,
                "loss_ema": loss_ema, "grad_norm": grad_value,
            }
            if evaluation is not None and step % 250 == 0:
                point["fixed_eval_loss"] = evaluation(policy)
            curve.append(point)
            print(
                f"{phase} step={step} train_loss={loss_value:.5f} "
                f"ema={loss_ema:.5f} grad_norm={grad_value:.4f}"
                + (f" fixed_eval={point['fixed_eval_loss']:.5f}" if "fixed_eval_loss" in point else ""),
                flush=True,
            )
    return {
        "steps": steps, "lr": LR, "grad_clip": GRAD_CLIP,
        "max_grad_norm": max(gradient_norms),
        "typical_grad_norm": statistics.median(gradient_norms),
        "all_finite": True, "loss_curve": curve,
    }


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def learned_checkpoint_round_trip(
    policy: VisuotactileFlowPolicy, bundle: PolicyConfigBundle,
    train_data: SyntheticData, path: Path, seed: int,
) -> bool:
    policy.eval()
    before = policy.sample_encoded_action(
        train_data.observations,
        generator=seeded_generator(train_data.encoded_target_action.device, seed),
    )
    save_policy_checkpoint(
        policy, path, config_bundle=bundle,
        metadata={
            "dataset_fingerprint": "synthetic:20261008:N8",
            "encoder_provenance": "deterministic_pixel_projection_v1",
            "notes": "Tiny synthetic learning gate; no real robot data",
        },
    )
    fresh, fresh_bundle = build_tiny_policy(
        train_data, fit_normalizers=False, initialization_seed=SEED + 999,
    )
    load_policy_checkpoint(fresh, path, expected_config_bundle=fresh_bundle)
    fresh.eval()
    after = fresh.sample_encoded_action(
        train_data.observations,
        generator=seeded_generator(train_data.encoded_target_action.device, seed),
    )
    torch.testing.assert_close(after.normalized_action, before.normalized_action)
    torch.testing.assert_close(after.encoded_action, before.encoded_action)
    return True


def run(device: torch.device, output_dir: Path) -> dict:
    set_seed()
    data = make_synthetic_data(8, device)
    metrics: dict = {
        "seed": SEED, "device": str(device), "phase_a": None, "phase_b": None,
        "training": {
            "lr": LR, "grad_clip": GRAD_CLIP, "steps": {},
            "model_size": None,
        },
        "learned_checkpoint_roundtrip": False,
    }
    metrics_path = output_dir / "metrics.json"
    policy_a, _ = build_tiny_policy(data, initialization_seed=SEED + 201)
    inventory = parameter_inventory(policy_a)
    metrics["training"]["model_size"] = inventory
    print(f"Tiny model inventory: {inventory}", flush=True)
    sample_index = torch.tensor([0], device=device)
    observation_a = select_observations(data.observations, sample_index)
    target_a = data.encoded_target_action[:1]
    normalized_a = policy_a.action_normalizer.normalize(target_a)
    z_fixed = torch.randn(
        1, HORIZON, ACTION_DIM, device=device,
        generator=seeded_generator(device, SEED + 210),
    )
    time_points = torch.linspace(0.05, 0.95, 10, device=device, dtype=torch.float32)
    fixed_a = [(z_fixed, time.reshape(1)) for time in time_points]
    a_initial = fixed_cfm_loss(policy_a, observation_a, target_a, fixed_a)
    a_training = train_bounded(
        policy_a, observation_a, target_a,
        steps=PHASE_A_MAX_STEPS, phase="A",
        generator=seeded_generator(device, SEED + 211),
        source_override=z_fixed,
    )
    a_final = fixed_cfm_loss(policy_a, observation_a, target_a, fixed_a)
    a_endpoint = fixed_source_euler_rmse(policy_a, observation_a, normalized_a, z_fixed)
    a_passed = a_final < 0.10 * a_initial and a_endpoint < 0.20
    metrics["phase_a"] = {
        "initial_fixed_loss": a_initial, "final_fixed_loss": a_final,
        "endpoint_rmse": a_endpoint, "passed": a_passed,
        "training": a_training,
    }
    metrics["training"]["steps"]["phase_a"] = PHASE_A_MAX_STEPS
    write_metrics(metrics_path, metrics)
    print(f"Phase A: initial={a_initial:.6f} final={a_final:.6f} "
          f"endpoint_rmse={a_endpoint:.6f} passed={a_passed}", flush=True)
    if not a_passed:
        return metrics

    # Fresh model: Phase B cannot inherit the single-trajectory solution.
    policy_b, bundle_b = build_tiny_policy(data, initialization_seed=SEED + 202)
    normalized_target = policy_b.action_normalizer.normalize(data.encoded_target_action)
    eval_gen = seeded_generator(device, SEED + 310)
    fixed_b = [
        (
            torch.randn(8, HORIZON, ACTION_DIM, device=device, generator=eval_gen),
            torch.rand(8, device=device, generator=eval_gen),
        )
        for _ in range(4)
    ]
    euler_seeds = tuple(SEED + 401 + index for index in range(4))
    b_initial = fixed_cfm_loss(
        policy_b, data.observations, data.encoded_target_action, fixed_b,
    )
    euler_before = seeded_euler_metrics(
        policy_b, data.observations, normalized_target, euler_seeds,
    )
    b_training = train_bounded(
        policy_b, data.observations, data.encoded_target_action,
        steps=PHASE_B_MAX_STEPS, phase="B",
        generator=seeded_generator(device, SEED + 311),
        evaluation=lambda model: fixed_cfm_loss(
            model, data.observations, data.encoded_target_action, fixed_b,
        ),
    )
    b_final = fixed_cfm_loss(
        policy_b, data.observations, data.encoded_target_action, fixed_b,
    )
    euler_after = seeded_euler_metrics(
        policy_b, data.observations, normalized_target, euler_seeds,
    )
    correct, shuffled = condition_permutation_metrics(
        policy_b, data.observations, normalized_target, euler_seeds[0],
    )
    ratio = correct / shuffled if shuffled > 0 else float("inf")
    b_passed = (
        b_final < 0.25 * b_initial
        and euler_after["rmse"] < 0.50 * euler_before["rmse"]
        and correct < 0.75 * shuffled
        and b_training["all_finite"]
    )
    metrics["phase_b"] = {
        "initial_fixed_loss": b_initial, "final_fixed_loss": b_final,
        "initial_euler_rmse": euler_before["rmse"],
        "final_euler_rmse": euler_after["rmse"],
        "per_channel_rmse": euler_after["per_channel_rmse"],
        "translation_rmse": euler_after["translation_rmse"],
        "rotation_rmse": euler_after["rotation_rmse"],
        "gripper_rmse": euler_after["gripper_rmse"],
        "correct_condition_rmse": correct,
        "shuffled_condition_rmse": shuffled,
        "condition_permutation_ratio": ratio,
        "passed": b_passed,
        "training": b_training,
    }
    metrics["training"]["steps"]["phase_b"] = PHASE_B_MAX_STEPS
    write_metrics(metrics_path, metrics)
    print(
        f"Phase B: fixed_initial={b_initial:.6f} fixed_final={b_final:.6f} "
        f"euler_initial={euler_before['rmse']:.6f} euler_final={euler_after['rmse']:.6f} "
        f"correct={correct:.6f} shuffled={shuffled:.6f} ratio={ratio:.4f} "
        f"passed={b_passed}",
        flush=True,
    )
    if not b_passed:
        return metrics
    metrics["learned_checkpoint_roundtrip"] = learned_checkpoint_round_trip(
        policy_b, bundle_b, data, output_dir / "tiny_policy.pt", SEED + 490,
    )
    write_metrics(metrics_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/synthetic_overfit")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    metrics = run(torch.device(args.device), args.output_dir)
    print(f"Metrics: {args.output_dir / 'metrics.json'}", flush=True)
    if not metrics["phase_a"]["passed"] or not metrics["phase_b"] or not metrics["phase_b"]["passed"]:
        raise SystemExit(1)
    if not metrics["learned_checkpoint_roundtrip"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
