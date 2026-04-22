"""YAML configuration loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_KEYS = ["experiment", "model", "training", "data", "output"]
BASE_CONFIG_FILENAME = "base_config.yaml"


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into ``base``.

    Args:
        base: Base configuration dictionary.
        overrides: Override dictionary.

    Returns:
        Merged configuration dictionary.
    """
    merged = dict(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load, merge, and validate a YAML configuration file.

    Args:
        config_path: Path to the target YAML file.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If the resulting configuration is missing required keys.
    """
    resolved_path = Path(config_path)
    with resolved_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if resolved_path.name != BASE_CONFIG_FILENAME:
        base_config_path = resolved_path.with_name(BASE_CONFIG_FILENAME)
        if base_config_path.exists():
            with base_config_path.open("r", encoding="utf-8") as base_config_file:
                base_config = yaml.safe_load(base_config_file) or {}
            config = _deep_merge(base_config, config)

    missing_keys = [
        key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config
    ]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(
            f"Configuration '{resolved_path}' is missing required keys: {missing}"
        )

    return config
