"""Stufe 4: Lautstaerke-Huellkurve.

Bei Reaction- und Challenge-Content ist die Audio-Energie das mit Abstand
guenstigste Signal fuer virale Momente: Schreie, Jubel, Explosionen und
Musik-Drops erzeugen alle einen klaren RMS-Ausschlag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import EnergyPeak
from ..utils.ffmpeg import decode_audio_mono

SAMPLE_RATE = 16_000


def energy_envelope(video_path: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Gibt (zeiten, normalisierte_energie) zurueck, beide gleich lang."""
    window = cfg["audio"]["window"]
    samples = decode_audio_mono(video_path, SAMPLE_RATE)

    hop = max(1, int(window * SAMPLE_RATE))
    n_windows = len(samples) // hop
    if n_windows == 0:
        return np.zeros(0), np.zeros(0)

    trimmed = samples[: n_windows * hop].reshape(n_windows, hop)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))

    # dBFS statt linear: entspricht der menschlichen Wahrnehmung deutlich besser.
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

    # Zusammenhaengende Fenster zu einem Peak zusammenfassen.
    peaks: list[EnergyPeak] = []
    for group in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if len(group) == 0:
            continue
        best = group[int(np.argmax(energy[group]))]
        peaks.append(EnergyPeak(t=float(times[best]), score=float(energy[best])))
    return peaks
