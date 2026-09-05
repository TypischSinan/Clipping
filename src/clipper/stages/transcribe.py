"""Stage 2: transcript with word-level timestamps (faster-whisper / CTranslate2)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..models import Segment, Word
from ..utils.cuda import register_cuda_dlls
from ..utils.ffmpeg import has_audio


def _resolve_device(requested: str) -> tuple[str, str]:
    """Return (device, compute_type). Falls back to CPU cleanly."""
    if requested == "cpu":
        return "cpu", "int8"

    register_cuda_dlls()
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def transcribe(
    video_path: Path,
    cfg: dict,
    on_progress: Callable[[float], None] | None = None,
) -> list[Segment]:
    # Nothing to transcribe without an audio track, and Whisper would raise.
    # The rest of the pipeline copes with an empty transcript: no captions, and
    # selection falls back to shots plus keyframes.
    if not has_audio(video_path):
        return []

    # Must run before the import: CTranslate2 resolves the CUDA DLLs on load.
    register_cuda_dlls()
    from faster_whisper import WhisperModel

    tc = cfg["transcribe"]
    device, auto_compute = _resolve_device(tc["device"])
    compute_type = auto_compute if tc["compute_type"] == "auto" else tc["compute_type"]

    model = WhisperModel(tc["model"], device=device, compute_type=compute_type)

    segments_iter, _info = model.transcribe(
        str(video_path),
        language=tc.get("language"),
        word_timestamps=True,
        vad_filter=tc.get("vad_filter", True),
        beam_size=5,
    )

    segments: list[Segment] = []
    for seg in segments_iter:
        words = [
            Word(start=w.start, end=w.end, text=w.word.strip())
            for w in (seg.words or [])
            if w.start is not None and w.end is not None
        ]
        segments.append(
            Segment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
        )
        # faster-whisper yields lazily, so the segment end doubles as a
        # position in the source - the only progress signal available here.
        if on_progress is not None:
            on_progress(seg.end)
    return segments
