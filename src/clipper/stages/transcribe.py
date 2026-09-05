"""Stufe 2: Transkript mit Wort-Zeitstempeln (faster-whisper / CTranslate2)."""

from __future__ import annotations

from pathlib import Path

from ..models import Segment, Word
from ..utils.cuda import register_cuda_dlls


def _resolve_device(requested: str) -> tuple[str, str]:
    """Gibt (device, compute_type) zurueck. Faellt sauber auf CPU zurueck."""
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


def transcribe(video_path: Path, cfg: dict) -> list[Segment]:
    # Muss vor dem Import laufen: CTranslate2 loest die CUDA-DLLs beim Laden auf.
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
    return segments
