"""Stage 1: source video -> local mp4 + metadata.

Accepts a URL for yt-dlp or a path to a file that already exists on disk.
"""

from __future__ import annotations

import hashlib
import re
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


def _local_video_id(path: Path) -> str:
    """Stable, readable id for a local file.

    The filename alone would collide between two different videos that happen
    to share a name, so a short hash of the absolute path is appended.
    """
    stem = re.sub(r"[^\w-]+", "_", path.stem)[:40].strip("_") or "local"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


def _probe_source(path: Path, video_id: str, title: str, channel: str, url: str) -> SourceVideo:
    stream = video_stream(path)
    # Read the duration from the file, not from any metadata: on a partial
    # download every timestamp in the pipeline refers to the file.
    duration = float(probe(path).get("format", {}).get("duration") or 0.0)
    return SourceVideo(
        video_id=video_id,
        title=title,
        channel=channel,
        duration=duration,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=parse_fps(stream.get("r_frame_rate", "30/1")),
        path=str(path),
        url=url,
    )


def ingest_local(path: Path, work_dir: Path) -> SourceVideo:
    """Use a file that is already on disk.

    The file is referenced where it lies rather than copied - source videos run
    to gigabytes and a second copy buys nothing. Only the derived artefacts go
    into the work directory.
    """
    video_id = _local_video_id(path)
    (work_dir / video_id).mkdir(parents=True, exist_ok=True)
    return _probe_source(path, video_id, path.stem, "", str(path))


def download(url: str, work_dir: Path, cfg: dict, force: bool = False) -> SourceVideo:
    # A path that exists wins over URL handling - otherwise yt-dlp tries to
    # parse "C:/..." as a URL scheme and fails with an unhelpful message.
    candidate = Path(url).expanduser()
    if candidate.exists() and candidate.is_file():
        return ingest_local(candidate, work_dir)

    max_height = cfg["ingest"]["max_height"]
    sections = cfg["ingest"].get("download_sections")

    # Fetch metadata first so the work directory can be named after video_id.
    # `noplaylist` matters here: a link copied out of a playlist or a mix carries
    # `&list=`, and without the flag yt-dlp resolves the playlist instead of the
    # video - the id becomes the playlist id and every entry gets downloaded over
    # the same `source.%(ext)s`.
    with YoutubeDL(
        {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    ) as ydl:
        info = ydl.extract_info(url, download=False)

    # A playlist URL with no resolvable single video still comes back as a
    # playlist dict. Say so instead of failing later on a missing "id".
    if info.get("_type") in {"playlist", "multi_video"}:
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise RuntimeError(f"'{url}' resolves to an empty playlist.")
        info = entries[0]

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
            "noplaylist": True,
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

    return _probe_source(
        path,
        video_id,
        info.get("title", ""),
        info.get("uploader", "") or info.get("channel", ""),
        url,
    )
