#!/usr/bin/env python
"""Build the README demo GIF from a video you have already processed.

The left panel is the 16:9 source with the crop window outlined and everything
it discards dimmed; the right panel is the finished 9:16 clip. Two videos side
by side would leave the actual work invisible - the window is the work.

Needs `clipper analyze` plus a finished `clipper build` for the video, because
it reads the cached shots and transcript from `work/<id>/` and the clip list
from `out/<id>/clips.json`.

The clip is re-rendered rather than taken from `out/`: a file on disk may
predate the current reframe settings, and then the box would sit somewhere the
output never went. Dead-air removal is forced off for the same reason - a cut in
the right panel with no cut in the left one only reads as a glitch.

The GIF currently in the repo was built with:

    python docs/make_demo.py erLbbextvlY --clip 23 --offset 9 --duration 4.6

Use --scan first to find a clip whose crop actually moves; a static one makes a
dull demo.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clipper.config import OUT_DIR, WORK_DIR, load_config  # noqa: E402
from clipper.models import Candidate, ClipPlan, Segment, Shot  # noqa: E402
from clipper.stages import captions, reframe, render  # noqa: E402
from clipper.utils.cache import read_json  # noqa: E402

# --- layout -----------------------------------------------------------------
PAD, GUTTER, TOP = 28, 32, 58
SRC_W, SRC_H = 600, 338           # 16:9
OUT_W, OUT_H = 270, 480           # 9:16
CANVAS_W = PAD + SRC_W + GUTTER + OUT_W + PAD
CANVAS_H = TOP + OUT_H + PAD
OUT_X = PAD + SRC_W + GUTTER

BG, FG, DIM = "0x0F1115", "0xE6EDF3", "0x8B949E"
ACCENT = "0xFFE000"               # the caption highlight colour

FONTS = {
    "bold": [
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    "regular": [
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
}


def find_font(weight: str, override: str | None) -> str:
    if override:
        return override
    for path in FONTS[weight]:
        if Path(path).exists():
            return path
    raise SystemExit(
        f"No {weight} font found on this {platform.system()} machine. "
        f"Pass --font-{weight} with a path to a .ttf."
    )


def escape(value: str) -> str:
    r"""Escape a value for a filtergraph option.

    A colon separates options, so it has to go - in a Windows font path as much
    as in the text "16:9". Same trap the subtitles filter sets with a drive
    letter, solved there by running ffmpeg from the file's directory instead.
    """
    return value.replace("\\", "/").replace(":", r"\:")


def load(video_id: str):
    work, out = WORK_DIR / video_id, OUT_DIR / video_id
    missing = [
        p for p in (work / "shots.json", work / "transcript.json",
                    work / "source.json", out / "clips.json")
        if not p.exists()
    ]
    if missing:
        raise SystemExit(
            f"Missing for '{video_id}':\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nRun 'clipper analyze' and 'clipper build' for it first."
        )
    return (
        read_json(work / "source.json"),
        [Shot(**s) for s in read_json(work / "shots.json")],
        [Segment(**s) for s in read_json(work / "transcript.json")],
        read_json(out / "clips.json")["clips"],
    )


def to_candidate(clip: dict) -> Candidate:
    return Candidate(**{k: clip[k] for k in ("start", "end", "title", "hook", "score")})


def scan(source, shots, clips, cfg) -> None:
    """Report how far each clip's crop travels. Slow - one reframe pass each."""
    src_path = Path(source["path"])
    rows = []
    for clip in clips:
        crops = reframe.analyze(src_path, clip["start"], clip["end"], shots, cfg)
        xs = [k.x for k in crops]
        rows.append((max(xs) - min(xs), len(crops), clip))
    rows.sort(reverse=True, key=lambda r: r[0])
    print(f"{'clip':>5}  {'x-span':>7}  {'shots':>5}  {'score':>5}  hook")
    for span, n, clip in rows:
        print(f"{clip['index']:>5}  {span:>6}px  {n:>5}  {clip['score']:>5.0f}  {clip['hook'][:50]}")


def crop_boxes(crops, cand, scale, box_w, offset, duration, dim) -> str:
    """One fixed box per shot, switched on for the stretch it belongs to.

    An expression in drawbox's `x` is the obvious way to move a box and it
    silently does not work: drawbox evaluates its geometry once when the filter
    is configured, so the box stays on whichever branch that first evaluation
    happened to take, for every frame, with no error. `enable` *is* evaluated
    per frame, so the positions become separate filters instead.
    """
    parts: list[str] = []
    for i, k in enumerate(crops):
        a = k.t - offset
        b = (crops[i + 1].t if i + 1 < len(crops) else cand.duration) - offset
        if b <= 0 or a >= duration:
            continue
        gate = f"enable='between(t,{max(a, -1.0):.3f},{min(b, duration + 1):.3f})'"
        x = round(k.x * scale)
        right = x + box_w
        if x > 0:
            parts.append(f"drawbox=x=0:y=0:w={x}:h=ih:color=black@{dim}:t=fill:{gate}")
        if right < SRC_W:
            parts.append(
                f"drawbox=x={right}:y=0:w={SRC_W - right}:h=ih:"
                f"color=black@{dim}:t=fill:{gate}"
            )
        parts.append(f"drawbox=x={x}:y=0:w={box_w}:h=ih:color={ACCENT}:t=4:{gate}")
    return ",".join(parts)


def label(body: str, x: int, y: int, size: int, colour: str, font: str) -> str:
    return (
        f"drawtext=fontfile='{escape(font)}':text='{escape(body)}'"
        f":x={x}:y={y}:fontsize={size}:fontcolor={colour}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id", help="Video ID, as used by work/ and out/")
    ap.add_argument("--clip", type=int, help="Clip index (default: highest scoring)")
    ap.add_argument("--offset", type=float, default=0.0, help="Seconds into the clip")
    ap.add_argument("--duration", type=float, default=4.6)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--dim", type=float, default=0.78,
                    help="Opacity of the black over the discarded area")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.gif")
    ap.add_argument("--scan", action="store_true",
                    help="List clips by how far the crop travels, then exit")
    ap.add_argument("--font-bold", dest="font_bold")
    ap.add_argument("--font-regular", dest="font_regular")
    args = ap.parse_args()

    cfg = load_config()
    cfg["silence"]["enabled"] = False
    source, shots, segments, clips = load(args.video_id)
    src_path = Path(source["path"])

    if args.scan:
        scan(source, shots, clips, cfg)
        return

    if args.clip is None:
        clip = max(clips, key=lambda c: c["score"])
        print(f"No --clip given, using the highest scoring one ({clip['index']}).")
    else:
        clip = next((c for c in clips if c["index"] == args.clip), None)
        if clip is None:
            raise SystemExit(f"No clip {args.clip} in {args.video_id}.")

    cand = to_candidate(clip)
    if args.offset + args.duration > cand.duration:
        raise SystemExit(
            f"Clip {clip['index']} is only {cand.duration:.1f}s long; "
            f"offset {args.offset} plus duration {args.duration} runs past its end."
        )

    bold = find_font("bold", args.font_bold)
    regular = find_font("regular", args.font_regular)

    with tempfile.TemporaryDirectory(prefix="clipper-demo-") as tmpdir:
        build(Path(tmpdir), args, cfg, clip, cand, source, src_path, shots,
              segments, bold, regular)


def build(tmp, args, cfg, clip, cand, source, src_path, shots, segments,
          bold, regular) -> None:
    """Render the clip and compose the GIF. Everything lands in `tmp`, which is
    a system temp directory - a build must not leave anything in the repo."""
    crops = sorted(reframe.analyze(src_path, cand.start, cand.end, shots, cfg),
                   key=lambda k: k.t)
    ass = captions.build_ass(segments, cand.start, cand.end, tmp / "demo.ass", cfg,
                             hook=cand.hook)
    plan = ClipPlan(index=clip["index"], candidate=cand, crops=crops, ass_path=str(ass))
    clip_mp4 = tmp / "clip.mp4"
    render.render_clip(src_path, plan, clip_mp4, cfg)

    scale = SRC_W / source["width"]
    box_w = round(crops[0].w * scale)
    boxes = crop_boxes(crops, cand, scale, box_w, args.offset, args.duration, args.dim)

    graph = ";".join([
        f"color=c={BG}:s={CANVAS_W}x{CANVAS_H}:d={args.duration}:r={args.fps}[bg]",
        f"[0:v]scale={SRC_W}:{SRC_H},{boxes}[src]",
        f"[1:v]scale={OUT_W}:{OUT_H}[out]",
        f"[bg][src]overlay={PAD}:{TOP}[a]",
        f"[a][out]overlay={OUT_X}:{TOP}[b]",
        "[b]" + ",".join([
            label("SOURCE  16:9", PAD, 22, 20, FG, bold),
            label("OUTPUT  9:16", OUT_X, 22, 20, FG, bold),
            label("The yellow window is the crop.", PAD, TOP + SRC_H + 26, 17, FG, bold),
            label("One position per shot, so the frame never drifts mid-shot.",
                  PAD, TOP + SRC_H + 54, 16, DIM, regular),
            label("Captions are burned in word by word, loudness normalised.",
                  PAD, TOP + SRC_H + 80, 16, DIM, regular),
        ]) + "[v]",
    ])

    inputs = [
        "-ss", f"{cand.start + args.offset:.3f}", "-t", f"{args.duration:.3f}",
        "-i", str(src_path),
        "-ss", f"{args.offset:.3f}", "-t", f"{args.duration:.3f}", "-i", str(clip_mp4),
    ]
    palette = tmp / "palette.png"

    # Two passes: one to find a palette for this specific clip, one to apply it.
    # A generic web palette on this material bands badly in the dimmed area.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex",
         f"{graph};[v]palettegen=max_colors={args.colors}:stats_mode=diff[p]",
         "-map", "[p]", str(palette)],
        check=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inputs, "-i", str(palette), "-filter_complex",
         f"{graph};[v][2:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle[g]",
         "-map", "[g]", "-loop", "0", str(args.out)],
        check=True,
    )

    print(
        f"clip {clip['index']}  {CANVAS_W}x{CANVAS_H}  {args.duration:.1f}s @ "
        f"{args.fps}fps  ->  {args.out}  "
        f"({args.out.stat().st_size / 1_000_000:.2f} MB)"
    )


if __name__ == "__main__":
    main()
