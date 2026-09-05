"""Load and merge configuration."""

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

# Output formats the short-form platforms actually take. Width stays at 1080
# throughout: it is what TikTok, Reels and Shorts encode to anyway, so going
# higher only costs render time.
ASPECT_PRESETS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),   # TikTok, Reels, Shorts - full screen
    "4:5": (1080, 1350),    # Instagram feed - the tallest the feed allows
    "1:1": (1080, 1080),    # square, safe everywhere
}


def aspect_override(aspect: str) -> dict:
    """Turn an aspect preset into a config override."""
    if aspect not in ASPECT_PRESETS:
        known = ", ".join(ASPECT_PRESETS)
        raise ValueError(f"Unknown aspect '{aspect}'. Known: {known}")
    width, height = ASPECT_PRESETS[aspect]
    return {"reframe": {"target_width": width, "target_height": height}}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load default.yaml, optionally merge a custom file, then CLI overrides."""
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if path is not None:
        with Path(path).open("r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    return cfg
