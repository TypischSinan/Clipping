"""Konfiguration laden und mergen."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
WORK_DIR = PROJECT_ROOT / "work"
OUT_DIR = PROJECT_ROOT / "out"
MODELS_DIR = PROJECT_ROOT / "models"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Laedt default.yaml, merged optional eine eigene Datei und dann CLI-Overrides."""
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if path is not None:
        with Path(path).open("r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    return cfg
