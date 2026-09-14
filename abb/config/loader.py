"""
abb.config.loader — YAML configuration loading with Pydantic validation.

Usage
-----
    from abb.config.loader import load_data_config

    cfg = load_data_config("configs/data/default.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from abb.config.schema import DataConfig


def load_data_config(path: str | Path) -> DataConfig:
    """
    Load and validate a DataConfig from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    DataConfig
        Validated, frozen configuration object.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If YAML content fails Pydantic validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    try:
        return DataConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Config validation failed for {path}:\n{exc}") from exc


def default_data_config() -> DataConfig:
    """Return a DataConfig populated entirely from defaults (no YAML required)."""
    return DataConfig()


def data_config_from_dict(d: dict[str, Any]) -> DataConfig:
    """Parse a DataConfig from a plain dict (useful in tests)."""
    try:
        return DataConfig.model_validate(d)
    except ValidationError as exc:
        raise ValueError(f"Config validation failed:\n{exc}") from exc
