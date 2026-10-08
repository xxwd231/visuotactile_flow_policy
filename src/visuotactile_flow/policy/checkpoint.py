"""Portable model/inference checkpoint contract v1 (no optimizer state)."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from visuotactile_flow.flow import BetaTimeSampler, UniformTimeSampler

from .config import PolicyConfigBundle
from .core import VisuotactileFlowPolicy


FORMAT_NAME = "visuotactile_flow_policy"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class LoadedPolicyCheckpoint:
    config_bundle: PolicyConfigBundle
    metadata: dict


def _same(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _validate_policy_config(
    policy: VisuotactileFlowPolicy, bundle: PolicyConfigBundle,
) -> None:
    """Reject declared settings that disagree with the constructed policy."""
    if not _same(bundle.data, asdict(policy.data_config)):
        raise ValueError("DataConfig mismatch")
    if not _same(bundle.action_normalization, policy.action_normalizer.modes):
        raise ValueError("Action normalization config mismatch")
    if not _same(bundle.state_normalization, policy.state_normalizer.modes):
        raise ValueError("State normalization config mismatch")
    expert = policy.flow_expert
    required_model = {
        "hidden_dim": expert.hidden_dim,
        "num_layers": len(expert.blocks),
        "num_heads": expert.blocks[0].self_attention.num_heads,
        "action_horizon": expert.action_horizon,
        "action_dim": expert.action_dim,
        "time_min_period": expert.timestep_embedder.min_period,
        "time_max_period": expert.timestep_embedder.max_period,
    }
    for key, expected in required_model.items():
        if key not in bundle.model or not _same(bundle.model[key], expected):
            raise ValueError(f"Model config mismatch: {key}")
    flow = bundle.flow
    if flow.get("convention") != policy.cfm_objective.convention.value:
        raise ValueError("FlowConvention mismatch")
    if flow.get("source") != {"type": "gaussian"} or flow.get("path") != {"type": "linear"}:
        raise ValueError("Flow source/path config mismatch")
    if flow.get("objective") != {"type": "conditional_flow_matching", "loss": "mse"}:
        raise ValueError("CFM objective config mismatch")
    if flow.get("solver") != {"type": "euler", "num_steps": policy.euler_solver.num_steps}:
        raise ValueError("Euler solver config mismatch")
    sampler = policy.cfm_objective.time_sampler
    declared = flow.get("time_sampler")
    if not isinstance(declared, dict):
        raise ValueError("Time sampler config mismatch")
    if isinstance(sampler, UniformTimeSampler):
        expected_sampler = {
            "type": "uniform", "min_time": sampler.min_time,
            "max_time": sampler.max_time,
        }
    elif isinstance(sampler, BetaTimeSampler):
        expected_sampler = {
            "type": "beta", "alpha": sampler.alpha, "beta": sampler.beta,
            "scale": sampler.scale, "offset": sampler.offset,
        }
        if "complement" in declared or sampler.complement:
            expected_sampler["complement"] = sampler.complement
    else:
        raise ValueError("Unsupported Stage-1 time sampler")
    if not _same(declared, expected_sampler):
        raise ValueError("Time sampler config mismatch")


def _metadata(value: dict | None) -> dict:
    if value is not None and not isinstance(value, dict):
        raise ValueError("metadata must be a mapping")
    metadata = {
        "dataset_fingerprint": None,
        "encoder_provenance": None,
        "source_commit": None,
        "notes": None,
    }
    if value is not None:
        metadata.update(value)
    fingerprint = metadata["dataset_fingerprint"]
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or Path(fingerprint).is_absolute()
    ):
        raise ValueError("dataset_fingerprint must be an identifier, not an absolute path")
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain JSON-friendly finite values") from exc


def save_policy_checkpoint(
    policy: VisuotactileFlowPolicy, path: str | Path, *,
    config_bundle: PolicyConfigBundle, metadata: dict | None = None,
) -> None:
    if not isinstance(policy, VisuotactileFlowPolicy):
        raise ValueError("policy must be VisuotactileFlowPolicy")
    policy._require_normalizers()
    if not isinstance(config_bundle, PolicyConfigBundle):
        raise ValueError("config_bundle must be PolicyConfigBundle")
    _validate_policy_config(policy, config_bundle)
    payload = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model_state_dict": policy.state_dict(),
        "action_normalizer_state": policy.action_normalizer.state_dict(),
        "state_normalizer_state": policy.state_normalizer.state_dict(),
        "config_bundle": config_bundle.to_dict(),
        "metadata": _metadata(metadata),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_policy_checkpoint(
    policy: VisuotactileFlowPolicy, path: str | Path, *,
    expected_config_bundle: PolicyConfigBundle | None = None, strict: bool = True,
) -> LoadedPolicyCheckpoint:
    if not isinstance(policy, VisuotactileFlowPolicy):
        raise ValueError("policy must be VisuotactileFlowPolicy")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_name") != FORMAT_NAME:
        raise ValueError("Unsupported policy checkpoint format")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported policy checkpoint version")
    bundle = PolicyConfigBundle.from_dict(payload.get("config_bundle"))
    if expected_config_bundle is not None:
        if (not isinstance(expected_config_bundle, PolicyConfigBundle)
                or bundle.canonical_json() != expected_config_bundle.canonical_json()):
            raise ValueError("Checkpoint config bundle mismatch")
    _validate_policy_config(policy, bundle)
    for key in ("model_state_dict", "action_normalizer_state", "state_normalizer_state"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"Missing or invalid {key}")
    metadata = _metadata(payload.get("metadata"))
    policy.load_state_dict(payload["model_state_dict"], strict=strict)
    policy.action_normalizer.load_state_dict(payload["action_normalizer_state"])
    policy.state_normalizer.load_state_dict(payload["state_normalizer_state"])
    return LoadedPolicyCheckpoint(config_bundle=bundle, metadata=metadata)
