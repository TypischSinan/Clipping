"""Pipeline data models. All times in seconds (float), relative to the source video."""

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
    """One camera shot between two hard cuts."""

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
    """Loudness peak in the audio - the strongest single signal for reaction content."""

    t: float
    score: float  # 0..1, normalised across the whole video


class Candidate(BaseModel):
    """A proposed clip, before it gets rendered."""

    start: float
    end: float
    title: str
    hook: str = ""
    reason: str = ""
    score: float = 0.0  # 0..100
    caption: str = ""  # suggested TikTok caption

    @property
    def duration(self) -> float:
        return self.end - self.start


class CropKeyframe(BaseModel):
    """Crop window starting at time t. Constant until the next keyframe."""

    t: float
    x: int
    y: int
    w: int
    h: int


class ClipPlan(BaseModel):
    """Everything the renderer needs for one clip."""

    index: int
    candidate: Candidate
    crops: list[CropKeyframe]
    ass_path: str | None = None
    out_path: str | None = None
