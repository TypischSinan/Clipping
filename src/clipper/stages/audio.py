"""Stage 4: loudness envelope.

For reaction and challenge content, audio energy is by far the cheapest signal
for viral moments: screams, cheers, explosions and music drops all produce a
clear RMS spike.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import EnergyPeak
from ..utils.ffmpeg import decode_audio_mono, has_audio

SAMPLE_RATE = 16_000


def energy_envelope(video_path: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, normalised_energy), both the same length."""
    window = cfg["audio"]["window"]
    # No audio track is a normal case, not an error: the energy curve is an
    # auxiliary signal. Transcript, shots and keyframes work without it.
    if not has_audio(video_path):
        return np.zeros(0), np.zeros(0)
    samples = decode_audio_mono(video_path, SAMPLE_RATE)

    hop = max(1, int(window * SAMPLE_RATE))
    n_windows = len(samples) // hop
    if n_windows == 0:
        return np.zeros(0), np.zeros(0)

    trimmed = samples[: n_windows * hop].reshape(n_windows, hop)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))

    # dBFS rather than linear: matches human perception considerably better.
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    lo, hi = np.percentile(db, 5), np.percentile(db, 99)
    norm = np.clip((db - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    times = np.arange(n_windows) * window
    return times, norm


def find_peaks(times: np.ndarray, energy: np.ndarray, cfg: dict) -> list[EnergyPeak]:
    if len(energy) == 0:
        return []
    threshold = np.percentile(energy, cfg["audio"]["peak_percentile"])
    idx = np.where(energy >= threshold)[0]

    # Collapse contiguous windows into a single peak.
    peaks: list[EnergyPeak] = []
    for group in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if len(group) == 0:
            continue
        best = group[int(np.argmax(energy[group]))]
        peaks.append(EnergyPeak(t=float(times[best]), score=float(energy[best])))
    return peaks
