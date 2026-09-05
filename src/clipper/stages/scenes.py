"""Stage 3: shot detection. A clip must never start mid-shot - a cut on a shot
boundary looks intentional, a cut in the middle looks like a mistake."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from ..models import Shot


def detect_shots(
    video_path: Path,
    cfg: dict,
    on_progress: Callable[[float], None] | None = None,
) -> list[Shot]:
    sc = cfg["scenes"]
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=sc["threshold"],
            min_scene_len=int(sc["min_scene_len"] * video.frame_rate),
        )
    )

    # The detector callback fires on every cut it finds, which on typical
    # material is often enough to drive a progress bar.
    callback = None
    if on_progress is not None:
        rate = video.frame_rate or 1.0

        def callback(_image, frame_num: int) -> None:
            on_progress(frame_num / rate)

    manager.detect_scenes(video, show_progress=False, callback=callback)
    scene_list = manager.get_scene_list()

    if not scene_list:
        return [Shot(index=0, start=0.0, end=float(video.duration.get_seconds()))]

    return [
        Shot(index=i, start=start.get_seconds(), end=end.get_seconds())
        for i, (start, end) in enumerate(scene_list)
    ]


def shot_boundaries(shots: list[Shot]) -> list[float]:
    if not shots:
        return []
    return sorted({s.start for s in shots} | {shots[-1].end})


def snap_to_shots(t: float, shots: list[Shot], *, edge: str, max_drift: float = 1.5) -> float:
    """Snap a timestamp onto a shot boundary.

    The direction is deliberately asymmetric: a clip start prefers the boundary
    before it, a clip end the one after. Always snapping to the nearest boundary
    would clip the first or last syllable of the moment in close cases.
    """
    boundaries = shot_boundaries(shots)
    if not boundaries:
        return t

    if edge == "start":
        preferred = [b for b in boundaries if b <= t and t - b <= max_drift]
        if preferred:
            return max(preferred)
    elif edge == "end":
        preferred = [b for b in boundaries if b >= t and b - t <= max_drift]
        if preferred:
            return min(preferred)

    nearest = min(boundaries, key=lambda b: abs(b - t))
    return nearest if abs(nearest - t) <= max_drift else t


def snap_end_within(
    start: float,
    desired_end: float,
    shots: list[Shot],
    min_duration: float,
    max_duration: float,
    hard_max: float | None = None,
) -> float:
    """Pick a clip end that sits on a shot boundary *and* honours the length range.

    An earlier version snapped first and then clamped to min/max - which put the
    end back in the middle of a shot and made the snapping pointless. Here we
    only search among boundaries that already fall inside the allowed window.
    """
    lo, hi = start + min_duration, start + max_duration
    # `hard_max` is the edge of the free window, e.g. the start of the next
    # already-placed clip. Cutting past it would overlap, no matter what the
    # length range allows.
    if hard_max is not None:
        hi = min(hi, hard_max)
    boundaries = shot_boundaries(shots)
    usable = [b for b in boundaries if lo <= b <= hi]
    if usable:
        return min(usable, key=lambda b: abs(b - desired_end))
    # No shot boundary inside the allowed window: hard cut, as close to the
    # desired end as possible - but never past the end of the material. Without
    # this cap you get clips beyond the end of the video, and ffmpeg then writes
    # a truncated or empty file.
    end = max(lo, min(hi, desired_end))
    return min(end, boundaries[-1]) if boundaries else end
