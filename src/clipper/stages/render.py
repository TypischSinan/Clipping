"""Stage 8: rendering.

One single ffmpeg call per clip: trim, crop, scale, burn in captions, normalise
loudness, encode.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..models import ClipPlan, TimeMap
from ..utils.ffmpeg import has_audio, pick_encoder, run

# Config sections that change what a rendered file looks like. Everything else -
# which clips were picked, how the transcript was produced - is upstream of the
# encoder and cannot make an existing file wrong.
RENDER_SECTIONS = ("reframe", "captions", "render", "silence")


def fingerprint(cfg: dict) -> str:
    """Short hash over the config a rendered clip depends on.

    `build` reuses clips that are already on disk, and the filename carries only
    index, score and title - nothing about the format. Without this hash
    `build --aspect 4:5` would report success while handing back the 9:16 files
    from the previous run.
    """
    subset = {name: cfg.get(name) for name in RENDER_SECTIONS}
    canonical = json.dumps(subset, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _crop_x_expr(plan: ClipPlan) -> str:
    """Build a piecewise ffmpeg expression over t from the crop keyframes.

    Result looks like: if(lt(t,2.5),100,if(lt(t,7.0),240,310))
    """
    crops = sorted(plan.crops, key=lambda c: c.t)
    if len(crops) == 1:
        return str(crops[0].x)

    # Nest from the back so the first comparison ends up outermost.
    expr = str(crops[-1].x)
    for i in range(len(crops) - 2, -1, -1):
        expr = f"if(lt(t,{crops[i + 1].t:.3f}),{crops[i].x},{expr})"
    return expr


def _select_expr(timing: TimeMap, fps: float) -> str:
    """Turn the kept intervals into a select/aselect expression.

    Both bounds are pulled back by half a frame. The interval edges sit exactly
    on frame positions, and `between` is inclusive at both ends, so comparing
    against them leaves it to floating-point rounding whether the edge frame is
    kept - independently for video and for audio. That is not a rounding
    curiosity: measured over 13 cuts the two sides disagreed often enough to
    accumulate 100 ms of A/V drift. Half-frame offsets are positions no frame
    and no audio block ever occupies, so each interval keeps exactly
    (b - a) * fps frames, on both sides, whatever the arithmetic does.
    """
    half = 0.5 / fps
    return "+".join(
        f"between(t,{a - half:.4f},{b - half:.4f})" for a, b in timing.keep
    )


# Deliberately no path escaping: the subtitles filter reads the colon of a
# Windows drive letter as an option separator, and the escaping required for
# that differs between ffmpeg versions and shells. Instead ffmpeg runs with
# cwd = the ASS file's directory and receives only the filename - then there is
# no colon to escape in the first place.


def _ffmpeg_args(
    source_path: Path,
    plan: ClipPlan,
    out_path: Path,
    cfg: dict,
    encoder: str,
    with_audio: bool = True,
) -> tuple[list[str], Path | None]:
    """Build the single ffmpeg call for one clip.

    Split out of render_clip so the argument list can be asserted in a test.
    The audio rate in particular is invisible in the picture and shows up only
    in the container metadata, which is exactly the kind of thing that stays
    broken for a long time.

    Returns the args plus the cwd ffmpeg has to run in - see the note above on
    why the subtitles filter needs one.
    """
    rc = cfg["render"]
    rf = cfg["reframe"]

    cand = plan.candidate
    base = plan.crops[0]
    timing = plan.timing
    cutting = timing is not None and not timing.is_identity

    # Order matters. `crop` reads the source timestamps, so it has to run before
    # anything renumbers them. `fps` comes next and forces a fixed grid, which is
    # what lets the select expression below cut on exactly the same instants as
    # the audio does. Scaling last means only the surviving frames get resized.
    filters = [
        f"crop=w={base.w}:h={base.h}:x='{_crop_x_expr(plan)}':y={base.y}",
        f"fps={rc['fps']}",
    ]
    if cutting:
        filters += [
            f"select='{_select_expr(timing, float(rc['fps']))}'",
            # Renumber from zero so the gaps actually close instead of freezing.
            "setpts=N/FRAME_RATE/TB",
        ]
    filters += [
        f"scale={rf['target_width']}:{rf['target_height']}:flags=lanczos",
        "setsar=1",
    ]
    subtitle_cwd: Path | None = None
    if plan.ass_path:
        ass = Path(plan.ass_path)
        subtitle_cwd = ass.parent
        filters.append(f"subtitles={ass.name}")

    args = [
        "ffmpeg", "-v", "error", "-y",
        # -ss before -i: fast seek. No -copyts, so timestamps start at 0.
        "-ss", f"{cand.start:.3f}",
        "-t", f"{cand.duration:.3f}",
        "-i", str(source_path),
        "-vf", ",".join(filters),
    ]

    afilters: list[str] = []
    if with_audio and cutting:
        # One audio block per video frame. Left alone, `aselect` keeps whole
        # decoder frames of ~21 ms while `select` keeps whole video frames of
        # ~33 ms, so the two have no common grid to cut on at all. This is the
        # first half of keeping them together; the second is the half-frame
        # offset in _select_expr, and the drift measured at 100 ms over 13 cuts
        # only disappeared once both were in place.
        samples = max(1, round(rc["audio_rate"] / rc["fps"]))
        afilters += [
            f"aresample={rc['audio_rate']}",
            # Start the audio grid at zero, where the regenerated video frames
            # already start. On the material measured here the first packet was
            # close enough to zero that this changed nothing, but it costs
            # nothing and removes the dependency on that being true.
            "asetpts=PTS-STARTPTS",
            f"asetnsamples=n={samples}:p=0",
            f"aselect='{_select_expr(timing, float(rc['fps']))}'",
            "asetpts=N/SR/TB",
        ]
    if with_audio and rc["loudnorm"]:
        # -14 LUFS is the target TikTok normalises to anyway.
        afilters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    if afilters:
        args += ["-af", ",".join(afilters)]

    if encoder.endswith("_nvenc"):
        args += [
            "-c:v", encoder,
            "-preset", rc["preset"],
            "-rc", "vbr", "-cq", str(rc["crf"]),
            "-b:v", "0",
        ]
    else:
        args += ["-c:v", encoder, "-preset", "medium", "-crf", str(rc["crf"])]

    args += ["-pix_fmt", "yuv420p"]

    if with_audio:
        args += [
            "-c:a", "aac", "-b:a", rc["audio_bitrate"],
            # Pin the audio rate. loudnorm runs its internal chain at 192 kHz,
            # and without an explicit rate the AAC encoder just clamps to its
            # own maximum of 96 kHz - a rate no short-form platform expects, so
            # every upload gets resampled again on their side.
            "-ar", str(rc["audio_rate"]),
        ]
    else:
        # Asking for an audio codec on a source without an audio stream makes
        # ffmpeg fail outright rather than just producing a silent file.
        args += ["-an"]

    args += ["-movflags", "+faststart", str(out_path)]
    return args, subtitle_cwd


def render_clip(
    source_path: Path,
    plan: ClipPlan,
    out_path: Path,
    cfg: dict,
) -> Path:
    encoder = pick_encoder(cfg["render"]["encoder"])
    args, subtitle_cwd = _ffmpeg_args(
        source_path, plan, out_path, cfg, encoder, with_audio=has_audio(source_path)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(args, cwd=subtitle_cwd)
    return out_path


def safe_filename(text: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    cleaned = re.sub(r"[\s_]+", "_", cleaned)
    return cleaned[:max_len] or "clip"
