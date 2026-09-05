"""Stage 8: rendering.

One single ffmpeg call per clip: trim, crop, scale, burn in captions, normalise
loudness, encode.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import ClipPlan
from ..utils.ffmpeg import has_audio, pick_encoder, run


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

    filters = [
        f"crop=w={base.w}:h={base.h}:x='{_crop_x_expr(plan)}':y={base.y}",
        f"scale={rf['target_width']}:{rf['target_height']}:flags=lanczos",
        f"fps={rc['fps']}",
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

    if with_audio and rc["loudnorm"]:
        # -14 LUFS is the target TikTok normalises to anyway.
        args += ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]

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
