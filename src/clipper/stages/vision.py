"""Keyframes for visual understanding.

Why this stage exists: in talking-head podcasts the transcript tells you where
the good moment is. In challenge and reaction content it does not - "oh my god"
says nothing about whether a car just exploded or someone opened a door.
Without images the model is picking blind here.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from ..models import EnergyPeak
from ..utils.ffmpeg import run


def pick_keyframe_times(
    duration: float,
    peaks: list[EnergyPeak],
    n_frames: int,
) -> list[float]:
    """A mix of energy peaks and even coverage.

    Pure peak selection misses quiet setup moments, a pure grid misses the
    payoffs. Hence 60/40.
    """
    if n_frames <= 0 or duration <= 0:
        return []

    n_peak = int(n_frames * 0.6)
    n_grid = n_frames - n_peak

    top_peaks = sorted(peaks, key=lambda p: p.score, reverse=True)[:n_peak]
    times = [p.t for p in top_peaks]

    if n_grid > 0:
        times += list(np.linspace(duration * 0.02, duration * 0.98, n_grid))

    # Drop near-duplicates so we do not waste tokens on them.
    times.sort()
    deduped: list[float] = []
    min_gap = max(duration / (n_frames * 3), 1.0)
    for t in times:
        if not deduped or t - deduped[-1] >= min_gap:
            deduped.append(float(t))
    return deduped


def extract_keyframes(
    video_path: Path,
    times: list[float],
    out_dir: Path,
    width: int = 512,
) -> list[tuple[float, Path]]:
    """Write one JPEG per timestamp. -ss before -i means a fast seek."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[float, Path]] = []

    for i, t in enumerate(times):
        path = out_dir / f"kf_{i:03d}_{t:.2f}.jpg"
        if not path.exists():
            run([
                "ffmpeg", "-v", "error", "-y",
                "-ss", f"{t:.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-vf", f"scale={width}:-2",
                "-q:v", "4",
                str(path),
            ])
        if path.exists():
            frames.append((t, path))
    return frames


def encode_image_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }
