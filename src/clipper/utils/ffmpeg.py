"""Thin ffmpeg/ffprobe wrapper. No Python wrapper package needed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


class FFmpegError(RuntimeError):
    pass


def ensure_ffmpeg() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise FFmpegError(
                f"'{binary}' not found on PATH. Install ffmpeg "
                "(e.g. 'winget install Gyan.FFmpeg') and reopen your shell."
            )


def run(args: list[str], *, capture: bool = False, cwd: Path | None = None) -> str:
    """Run ffmpeg/ffprobe; on failure raise with stderr included in the message."""
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd) if cwd else None,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise FFmpegError(f"{args[0]} exit {proc.returncode}:\n{tail}")
    return proc.stdout or ""


def probe(path: Path) -> dict:
    out = run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        capture=True,
    )
    return json.loads(out)


def video_stream(path: Path) -> dict:
    for stream in probe(path).get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise FFmpegError(f"No video stream in {path}")


def has_audio(path: Path) -> bool:
    """Whether the file carries at least one audio stream.

    Screen recordings, silent B-roll and some music videos have none. Every
    stage that touches audio has to check first, otherwise the run dies after
    the download and the transcription - the two expensive steps.
    """
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def parse_fps(rate: str) -> float:
    """'30000/1001' -> 29.97"""
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(rate)


def decode_audio_mono(path: Path, sample_rate: int = 16_000) -> np.ndarray:
    """Decode the audio track as a float32 mono array, for the energy analysis."""
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-map", "0:a:0",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(sample_rate),
            "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-10:]
        raise FFmpegError("Audio decoding failed:\n" + "\n".join(tail))
    return np.frombuffer(proc.stdout, dtype=np.float32)


def has_encoder(name: str) -> bool:
    """Only checks whether ffmpeg was built with this encoder."""
    try:
        out = run(["ffmpeg", "-v", "error", "-hide_banner", "-encoders"], capture=True)
    except FFmpegError:
        return False
    return name in out


def encoder_works(name: str) -> bool:
    """Encode one test frame.

    A compiled-in encoder does not mean a working one: NVENC fails at runtime
    when the driver is older than the NVENC API ffmpeg was built against.
    Otherwise that only surfaces during rendering.
    """
    if not has_encoder(name):
        return False
    try:
        run([
            "ffmpeg", "-v", "error", "-hide_banner",
            "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=1:duration=0.1",
            "-c:v", name, "-frames:v", "1",
            "-f", "null", "-",
        ])
        return True
    except FFmpegError:
        return False


_encoder_cache: dict[str, str] = {}


def pick_encoder(preference: str) -> str:
    """'auto' picks NVENC if it actually works, otherwise libx264."""
    if preference != "auto":
        return preference
    if "auto" in _encoder_cache:
        return _encoder_cache["auto"]

    for candidate in ("h264_nvenc", "libx264"):
        if encoder_works(candidate):
            _encoder_cache["auto"] = candidate
            return candidate

    _encoder_cache["auto"] = "libx264"
    return "libx264"
