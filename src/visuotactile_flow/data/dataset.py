"""Configuration boundary for a future dataset implementation.

No labels are selected or converted here. Acquisition must keep both measured
future pose and commanded target so the choice remains reversible.
"""

from pathlib import Path

import yaml

from .schema import DataConfig


def load_data_config(path: str | Path) -> DataConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Data config must be a YAML mapping")
    return DataConfig(**payload)
