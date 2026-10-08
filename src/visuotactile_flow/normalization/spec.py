"""Model-side normalization metadata and section configuration."""

from copy import deepcopy

from visuotactile_flow.data.schema import DataConfig


NORMALIZER_VERSION = 2


def contract_metadata(config: DataConfig, feature: str, modes: dict, *, dataset_fingerprint: str | None = None) -> dict:
    if feature not in {"action", "state"}:
        raise ValueError("feature must be action or state")
    result = {
        "normalizer_version": NORMALIZER_VERSION,
        "fit_split": "train",
        "feature": feature,
        "action_label_source": config.action_label_source.value,
        "control_frequency_hz": config.control_frequency_hz,
        "observation_history": config.observation_history,
        "action_horizon": config.action_horizon,
        "target_start_offset_steps": config.target_start_offset_steps,
        "action_representation": "relative_tcp_fixed_anchor_rot6d_rows",
        "tactile_representation_source": config.tactile_representation_source,
        "normalization_modes": deepcopy(modes),
        "dataset_fingerprint": dataset_fingerprint,
    }
    return result


def validate_metadata(metadata: dict, feature: str, modes: dict) -> None:
    if not isinstance(metadata, dict) or metadata.get("normalizer_version") != NORMALIZER_VERSION:
        raise ValueError("Unsupported normalization metadata version")
    source = metadata.get("tactile_representation_source")
    config = DataConfig(tactile_representation_source=source)
    expected = contract_metadata(config, feature, modes, dataset_fingerprint=metadata.get("dataset_fingerprint"))
    if metadata != expected:
        raise ValueError("Normalization metadata does not match TrainingDataContract v1")
