"""Stufe 3: Shot-Erkennung. Clips duerfen nie mitten in einem Shot anfangen -
ein Schnitt auf der Grenze wirkt gewollt, ein Schnitt mittendrin wie ein Fehler."""

from __future__ import annotations

from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from ..models import Shot


def detect_shots(video_path: Path, cfg: dict) -> list[Shot]:
    sc = cfg["scenes"]
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=sc["threshold"],
            min_scene_len=int(sc["min_scene_len"] * video.frame_rate),
        )
    )
    manager.detect_scenes(video, show_progress=False)
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
    """Zieht einen Zeitpunkt auf eine Shot-Grenze.

    Die Richtung ist nicht symmetrisch: ein Clipanfang wird bevorzugt auf die
    Grenze davor gezogen und ein Clipende auf die danach. Zoege man immer auf
    die absolut naechste Grenze, schnitte man bei knappen Faellen die erste
    beziehungsweise letzte Silbe des Moments ab.
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
    """Waehlt das Clipende so, dass es sowohl auf einer Shot-Grenze liegt als
    auch die Laengenvorgabe einhaelt.

    Fruehere Fassung hat erst gesnappt und danach hart auf min/max korrigiert -
    damit landete das Ende wieder mitten in einer Einstellung und das Snapping
    war wirkungslos. Hier wird nur unter den Grenzen gesucht, die im erlaubten
    Fenster liegen.
    """
    lo, hi = start + min_duration, start + max_duration
    # `hard_max` ist die Kante des freien Fensters, etwa der Anfang des naechsten
    # bereits vergebenen Clips. Darueber hinaus zu schneiden waere eine
    # Ueberlappung, egal was die Laengenvorgabe erlaubt.
    if hard_max is not None:
        hi = min(hi, hard_max)
    boundaries = shot_boundaries(shots)
    usable = [b for b in boundaries if lo <= b <= hi]
    if usable:
        return min(usable, key=lambda b: abs(b - desired_end))
    # Keine Schnittgrenze im erlaubten Fenster: harter Schnitt, so nah wie
    # moeglich am gewuenschten Ende - aber nie ueber das Material hinaus.
    # Ohne diese Deckelung entstehen Clips, die hinter dem Videoende liegen;
    # ffmpeg liefert dann eine zu kurze oder leere Datei.
    end = max(lo, min(hi, desired_end))
    return min(end, boundaries[-1]) if boundaries else end
