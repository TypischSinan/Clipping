"""Keyframes fuer das visuelle Verstaendnis.

Der Grund fuer diese Stufe: Bei Talking-Head-Podcasts steht im Transkript, wo
der gute Moment ist. Bei Challenge- und Reaction-Content steht er da nicht -
"oh my god" sagt nichts darueber aus, ob gerade ein Auto explodiert oder jemand
eine Tuer oeffnet. Ohne Bilder waehlt das Modell hier blind.
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
    """Mischung aus Energie-Spitzen und gleichmaessiger Abdeckung.

    Reine Peak-Auswahl verpasst ruhige Setup-Momente, reines Raster verpasst
    die Hoehepunkte. Deshalb 60/40.
    """
    if n_frames <= 0 or duration <= 0:
        return []

    n_peak = int(n_frames * 0.6)
    n_grid = n_frames - n_peak

    top_peaks = sorted(peaks, key=lambda p: p.score, reverse=True)[:n_peak]
    times = [p.t for p in top_peaks]

    if n_grid > 0:
        times += list(np.linspace(duration * 0.02, duration * 0.98, n_grid))

    # Doppelte in engem Abstand entfernen, damit keine Tokens verschwendet werden.
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
    """Schreibt je ein JPEG pro Zeitpunkt. -ss vor -i = schneller Seek."""
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
