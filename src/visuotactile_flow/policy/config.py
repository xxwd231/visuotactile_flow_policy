"""Small, canonical JSON/YAML-friendly bundle for policy compatibility."""

from dataclasses import dataclass
import json
from pathlib import Path

import yaml


_SECTIONS = (
    "model", "flow", "data", "action_normalization", "state_normalization",
)


def _canonical(value: object) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("Config bundle must contain JSON/YAML-friendly finite values") from exc
    if not isinstance(decoded, dict):
        raise ValueError("Config section must be a mapping")
    return encoded


@dataclass(frozen=True)
class PolicyConfigBundle:
    model: dict
    flow: dict
    data: dict
    action_normalization: dict
    state_normalization: dict

    def __post_init__(self) -> None:
        for name in _SECTIONS:
            value = getattr(self, name)
            if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                raise ValueError(f"{name} must be a string-keyed mapping")
            # Deep-copy to prevent later mutation of the caller's config.
            object.__setattr__(self, name, json.loads(_canonical(value)))

    def to_dict(self) -> dict:
        return {name: json.loads(_canonical(getattr(self, name))) for name in _SECTIONS}

    @classmethod
    def from_dict(cls, value: dict) -> "PolicyConfigBundle":
        if not isinstance(value, dict) or set(value) != set(_SECTIONS):
            raise ValueError("Config bundle requires model, flow, data and both normalization sections")
        return cls(**value)

    def canonical_json(self) -> str:
        return _canonical(self.to_dict())


def load_yaml_mapping(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("YAML config must be a string-keyed mapping")
    return json.loads(_canonical(value))
