"""Stage 1: YouTube link -> local mp4 + metadata."""

from __future__ import annotations

from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func

from ..models import SourceVideo
from ..utils.ffmpeg import parse_fps, probe, video_stream


def _to_seconds(value: str) -> float:
    """Accepts '90', '1:30' and '0:01:30'."""
    parts = value.strip().split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part or 0)
    return seconds


def parse_section(spec: str) -> tuple[float, float]:
    """Split '*120-360' or '2:00-6:00' into (start, end) in seconds."""
    text = spec.strip().lstrip("*")
    start_text, sep, end_text = text.partition("-")
    if not sep:
        raise ValueError(f"Section '{spec}' must have the form START-END")
    return _to_seconds(start_text), _to_seconds(end_text)


def download(url: str, work_dir: Path, cfg: dict, force: bool = False) -> SourceVideo:
    max_height = cfg["ingest"]["max_height"]
    sections = cfg["ingest"].get("download_sections")

    # Fetch metadata first so the work directory can be named after video_id.
    with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    video_id = info["id"]
    target_dir = work_dir / video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(target_dir / "source.%(ext)s")

    existing = sorted(target_dir.glob("source.*"))
    existing = [p for p in existing if p.suffix in {".mp4", ".mkv", ".webm"}]

    if existing and not force:
        path = existing[0]
    else:
        opts = {
            "outtmpl": outtmpl,
            "format": (
                f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
                f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
            ),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        if sections:
            # download_range_func(chapters, ranges) - ranges are (start, end)
            # in seconds. The yt-dlp CLI's asterisk syntax is not understood by
            # the Python API, so it has to be resolved beforehand.
            start_s, end_s = parse_section(sections)
            opts["download_ranges"] = download_range_func(None, [(start_s, end_s)])
            # Without this yt-dlp cuts at the keyframe and the section starts
            # earlier than requested - shifting every timestamp downstream.
            opts["force_keyframes_at_cuts"] = True
        with YoutubeDL(opts) as ydl:
            ydl.download([url])
        candidates = [p for p in sorted(target_dir.glob("source.*"))
                      if p.suffix in {".mp4", ".mkv", ".webm"}]
        if not candidates:
            raise RuntimeError(f"Download produced no video file in {target_dir}")
        path = candidates[0]

    stream = video_stream(path)
    # Read the duration from the file, not from yt-dlp metadata: on a partial
    # download every timestamp in the pipeline refers to the file, while the
    # metadata reports the length of the full video.
    file_duration = float(probe(path).get("format", {}).get("duration") or 0.0)

    return SourceVideo(
        video_id=video_id,
        title=info.get("title", ""),
        channel=info.get("uploader", "") or info.get("channel", ""),
        duration=file_duration or float(info.get("duration") or 0.0),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=parse_fps(stream.get("r_frame_rate", "30/1")),
        path=str(path),
        url=url,
    )
