"""Inspect default Stage-1 module inventory; optional synthetic loaded-weight smoke."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visuotactile_flow.condition import ConditionAdapter  # noqa: E402
from visuotactile_flow.data.schema import DataConfig  # noqa: E402
from visuotactile_flow.encoders import TactileEncoder, VisionEncoder, load_encoder_weights  # noqa: E402
from visuotactile_flow.flow import (  # noqa: E402
    BetaTimeSampler, ConditionalFlowMatchingObjective, EulerSolver, FlowConvention,
    GaussianSource, LinearConditionalFlowPath, UniformTimeSampler,
)
from visuotactile_flow.models import FlowActionExpert  # noqa: E402
from visuotactile_flow.normalization import (  # noqa: E402
    StructuredActionNormalizer, StructuredStateNormalizer,
)
from visuotactile_flow.policy import (  # noqa: E402
    PolicyConfigBundle, PolicyObservationBatch, VisuotactileFlowPolicy,
    load_yaml_mapping,
)


def config(path: str) -> dict:
    return load_yaml_mapping(ROOT / "configs" / path)


def build(flow_config_path: str) -> tuple[VisuotactileFlowPolicy, PolicyConfigBundle]:
    model_cfg = config("model/flow_300m.yaml")
    adapter_cfg = config("model/condition_adapter.yaml")
    flow_cfg = config(flow_config_path)
    data_cfg = config("data/default.yaml")
    action_cfg = config("normalization/action_meanstd_candidate.yaml")
    state_cfg = config("normalization/state_meanstd_candidate.yaml")
    data = DataConfig(**data_cfg)
    sampler_cfg = dict(flow_cfg["time_sampler"])
    sampler_type = sampler_cfg.pop("type")
    if sampler_type == "uniform":
        sampler = UniformTimeSampler(**sampler_cfg)
    elif sampler_type == "beta":
        sampler = BetaTimeSampler(**sampler_cfg)
    else:
        raise ValueError("Unsupported time sampler in inspection config")
    convention = FlowConvention(flow_cfg["convention"])
    objective = ConditionalFlowMatchingObjective(
        convention, GaussianSource(), LinearConditionalFlowPath(), sampler,
    )
    adapter = ConditionAdapter(
        model_dim=adapter_cfg["model_dim"],
        encoder_feature_dim=adapter_cfg["encoder_feature_dim"],
        state_dim=adapter_cfg["state_dim"],
        max_history=adapter_cfg["max_history"],
    )
    policy = VisuotactileFlowPolicy(
        vision_encoder=VisionEncoder(freeze=True),
        tactile_encoder=TactileEncoder(depth_encoding=data.tactile_encoding, freeze=True),
        condition_adapter=adapter,
        flow_expert=FlowActionExpert(**model_cfg),
        action_normalizer=StructuredActionNormalizer(action_cfg, data_config=data),
        state_normalizer=StructuredStateNormalizer(state_cfg, data_config=data),
        cfm_objective=objective,
        euler_solver=EulerSolver(convention, flow_cfg["solver"]["num_steps"]),
        data_config=data,
    )
    bundle = PolicyConfigBundle(model_cfg, flow_cfg, data_cfg, action_cfg, state_cfg)
    return policy, bundle


def count(module: torch.nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in module.parameters()), sum(
        p.numel() for p in module.parameters() if p.requires_grad
    )


def smoke(policy: VisuotactileFlowPolicy, encoder_dir: Path) -> None:
    for container, names in (
        (policy.vision_encoder, ("external_rgb", "wrist_rgb")),
        (policy.tactile_encoder, ("tactile_depth_0", "tactile_depth_1")),
    ):
        for name in names:
            load_encoder_weights(getattr(container, name), encoder_dir / f"{name}.pt")
    action_fit = torch.randn(4, 16, 10) * 0.1
    action_fit[..., 9] = torch.linspace(0.2, 0.8, 4)[:, None]
    state_fit = torch.randn(4, 3, 7) * 0.1
    state_fit[..., 6] = torch.linspace(0.2, 0.8, 4)[:, None]
    policy.action_normalizer.fit(action_fit)
    policy.state_normalizer.fit(state_fit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()
    obs = PolicyObservationBatch(
        external_rgb=torch.randint(0, 256, (1, 3, 3, 240, 320), dtype=torch.uint8, device=device),
        wrist_rgb=torch.randint(0, 256, (1, 3, 3, 240, 320), dtype=torch.uint8, device=device),
        tactile_depth_0=torch.rand(1, 3, 1, 288, 384, device=device) * 2 - 1,
        tactile_depth_1=torch.rand(1, 3, 1, 288, 384, device=device) * 2 - 1,
        agent_pos=torch.zeros(1, 3, 7, device=device),
    )
    output = policy.sample_encoded_action(obs)
    print("Synthetic loaded-weight smoke shape:", list(output.encoded_action.shape))
    print("Synthetic loaded-weight smoke finite:", bool(torch.isfinite(output.encoded_action).all()))
    print("NFE:", output.num_function_evaluations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-config", default="flow/standard_cfm.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--encoder-dir", type=Path, help="directory of four exported encoder .pt files")
    args = parser.parse_args()
    policy, bundle = build(args.flow_config)
    for name in ("vision_encoder", "tactile_encoder", "condition_adapter", "flow_expert"):
        total, trainable = count(getattr(policy, name))
        print(f"{name}: total={total}, trainable={trainable}")
    total, trainable = count(policy)
    policy.train()
    frozen_branches = (
        policy.vision_encoder.external_rgb, policy.vision_encoder.wrist_rgb,
        policy.tactile_encoder.tactile_depth_0, policy.tactile_encoder.tactile_depth_1,
    )
    if any(branch.training or any(parameter.requires_grad for parameter in branch.parameters())
           for branch in frozen_branches):
        raise RuntimeError("Frozen encoder branches changed behavior under policy.train()")
    policy.eval()
    print(f"Policy total params: {total}")
    print(f"Policy trainable params: {trainable}")
    print("Frozen encoder branches stay eval under policy.train(): True")
    print(f"Flow convention: {policy.cfm_objective.convention.value}")
    print(f"Action H/D: {policy.flow_expert.action_horizon}/{policy.flow_expert.action_dim}")
    print(f"Tactile encoding: {policy.data_config.tactile_encoding}")
    print(f"Tactile representation source: {policy.data_config.tactile_representation_source}")
    print(f"Action normalization modes: {bundle.action_normalization}")
    print(f"State normalization modes: {bundle.state_normalization}")
    if (total, trainable) != (344_652_810, 299_959_306):
        raise RuntimeError("Default Stage-1 parameter inventory changed")
    if args.smoke:
        if args.encoder_dir is None:
            parser.error("--smoke requires --encoder-dir")
        smoke(policy, args.encoder_dir)


if __name__ == "__main__":
    main()
