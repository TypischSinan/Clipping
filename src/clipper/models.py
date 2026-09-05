"""Datenmodelle der Pipeline. Alle Zeiten in Sekunden (float), relativ zum Quellvideo."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float
    end: float
    text: str


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Shot(BaseModel):
    """Ein Kamera-Shot zwischen zwei harten Schnitten."""

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class SourceVideo(BaseModel):
    video_id: str
    title: str
    channel: str
    duration: float
    width: int
    height: int
    fps: float
    path: str
    url: str


class EnergyPeak(BaseModel):
    """Lautstärke-Spitze im Audio - bei Reaction-Content das stärkste Einzelsignal."""

    t: float
    score: float  # 0..1, normalisiert über das gesamte Video


class Candidate(BaseModel):
    """Ein vorgeschlagener Clip, bevor er gerendert wird."""

    start: float
    end: float
    title: str
    hook: str = ""
    reason: str = ""
    score: float = 0.0  # 0..100
    caption: str = ""  # Vorschlag für die TikTok-Caption

    @property
    def duration(self) -> float:
        return self.end - self.start


class CropKeyframe(BaseModel):
    """Crop-Fenster ab Zeitpunkt t. Konstant bis zum naechsten Keyframe."""

    t: float
    x: int
    y: int
    w: int
    h: int


class ClipPlan(BaseModel):
    """Alles, was der Renderer für einen Clip braucht."""

    index: int
    candidate: Candidate
    crops: list[CropKeyframe]
    ass_path: str | None = None
    out_path: str | None = None
