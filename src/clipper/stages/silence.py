"""Stage 6b: dead-air removal.

Long-form pacing leaves gaps that short-form cannot afford: a speaker breathes,
walks across the set, waits for a reaction. On a 75-second clip that adds up to
several seconds of nothing, and short-form viewers leave during nothing.

Two signals decide, and both have to agree:

* a gap between two transcribed words, long enough to be a pause rather than a
  breath, and
* audio energy low enough that the gap really is empty.

The second one carries the whole feature. On reaction and challenge content most
word gaps are not silent at all - music runs underneath, something explodes, a
crowd reacts. Cutting on the transcript alone would delete exactly the moments
the clip was picked for. Measured on 14 minutes of source: 160 gaps over 0.6 s,
but only about 40% of them are actually quiet.

Only gaps *between* words are touched, never the clip edges. The start and end
were snapped onto shot boundaries on purpose (see stages/scenes.py); trimming
into them would quietly undo that.
"""

from __future__ import annotations

import numpy as np

from ..models import Segment, TimeMap


def _mean_energy(times: np.ndarray, energy: np.ndarray, a: float, b: float) -> float:
    """Average normalised loudness in [a, b), in source time.

    Returns 1.0 - "loud, keep it" - when there is nothing to go on. Every
    unknown has to fall on the side of keeping material: a wrong cut is visible
    in the clip, a missed cut only costs a second.
    """
    if len(times) == 0 or len(energy) == 0:
        return 1.0
    window = (times >= a) & (times < b)
    if not window.any():
        return 1.0
    return float(energy[window].mean())


def plan_cuts(
    segments: list[Segment],
    times: np.ndarray,
    energy: np.ndarray,
    start: float,
    end: float,
    cfg: dict,
    fps: float,
) -> TimeMap:
    """Work out which parts of [start, end) survive.

    Cut points are snapped onto the output frame grid. That is not cosmetic:
    video keeps whole frames and audio keeps whole sample blocks, and if the two
    round a cut differently the error accumulates over every further cut until
    the picture and the voice drift apart. On the grid both sides cut at exactly
    the same instant and the clip stays locked however many cuts it has.
    """
    duration = max(0.0, end - start)
    identity = TimeMap(keep=[(0.0, duration)], source_duration=duration)

    sc = cfg.get("silence") or {}
    if not sc.get("enabled", False) or duration <= 0:
        return identity

    min_gap = float(sc.get("min_gap", 0.6))
    padding = float(sc.get("padding", 0.12))
    max_energy = float(sc.get("max_energy", 0.35))
    min_removed = float(sc.get("min_removed", 0.25))

    words = sorted(
        (
            w
            for seg in segments
            for w in seg.words
            if w.end > start and w.start < end and w.text
        ),
        key=lambda w: w.start,
    )
    if len(words) < 2:
        # Nothing transcribed means nothing to measure a gap against. Music-only
        # or purely visual stretches stay untouched by design.
        return identity

    frame = 1.0 / fps if fps and fps > 0 else 0.0

    def snap(t: float) -> float:
        return round(t / frame) * frame if frame else t

    # The clip end is a shot boundary and lands wherever it lands, which is
    # almost never on a frame. Every kept interval has to be a whole number of
    # frames or the last one rounds differently for video than for audio, so the
    # tail is trimmed down to the grid - at most one frame.
    grid_end = int(duration / frame) * frame if frame else duration

    removed: list[tuple[float, float]] = []
    for left, right in zip(words, words[1:], strict=False):
        if right.start - left.end < min_gap:
            continue
        if _mean_energy(times, energy, left.end, right.start) > max_energy:
            continue

        lo = snap(left.end - start + padding)
        hi = snap(right.start - start - padding)
        if hi - lo < min_removed:
            continue
        # Never eat into the first or last frame: the clip edges are shot
        # boundaries, and this stage is not allowed to move them.
        if lo <= 0.0 or hi >= grid_end:
            continue
        removed.append((lo, hi))

    if not removed:
        return identity

    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for lo, hi in removed:
        if lo > cursor:
            keep.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < grid_end:
        keep.append((cursor, grid_end))

    # A surviving sliver shorter than a frame renders as nothing or as a single
    # stray frame. Dropping it widens the neighbouring cut, which is the lesser
    # of the two.
    keep = [(a, b) for a, b in keep if b - a > max(frame, 1e-6)]
    if not keep:
        return identity
    return TimeMap(keep=keep, source_duration=duration)
