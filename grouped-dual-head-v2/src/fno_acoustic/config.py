from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _expand(value: Any) -> Any:
    if isinstance(value, str) and (value.startswith("~") or "$" in value):
        return os.path.expandvars(str(Path(value).expanduser()))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if overrides:
        config = _deep_update(config, overrides)
    return _expand(config)


def config_input_features(config: dict[str, Any]) -> list[str]:
    return list(config.get("data", {}).get("input_features", ["time", "source_map", "velocity"]))
