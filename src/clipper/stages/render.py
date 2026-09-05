"""Stufe 8: Rendern.

Ein einziger ffmpeg-Aufruf pro Clip: schneiden, croppen, skalieren, Untertitel
einbrennen, Lautheit normalisieren, encoden.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import ClipPlan
from ..utils.ffmpeg import pick_encoder, run


def _crop_x_expr(plan: ClipPlan) -> str:
    """Baut aus den Crop-Keyframes einen stueckweisen ffmpeg-Ausdruck ueber t.

    Ergebnis der Form: if(lt(t,2.5),100,if(lt(t,7.0),240,310))
    """
    crops = sorted(plan.crops, key=lambda c: c.t)
    if len(crops) == 1:
        return str(crops[0].x)

    # Von hinten nach vorn verschachteln, damit der erste Vergleich aussen steht.
    expr = str(crops[-1].x)
    for i in range(len(crops) - 2, -1, -1):
        expr = f"if(lt(t,{crops[i + 1].t:.3f}),{crops[i].x},{expr})"
    return expr


# Absichtlich kein Pfad-Escaping: der subtitles-Filter interpretiert den
# Doppelpunkt eines Windows-Laufwerksbuchstabens als Options-Trenner, und die
# noetige Maskierung unterscheidet sich je nach ffmpeg-Version und Shell.
# Stattdessen laeuft ffmpeg mit cwd = Verzeichnis der ASS-Datei und bekommt
# nur den Dateinamen - dann gibt es gar keinen Doppelpunkt zu escapen.


def render_clip(
    source_path: Path,
    plan: ClipPlan,
    out_path: Path,
    cfg: dict,
) -> Path:
    rc = cfg["render"]
    rf = cfg["reframe"]
    encoder = pick_encoder(rc["encoder"])

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
        # -ss vor -i: schneller Seek. -copyts entfaellt, Zeiten starten bei 0.
        "-ss", f"{cand.start:.3f}",
        "-t", f"{cand.duration:.3f}",
        "-i", str(source_path),
        "-vf", ",".join(filters),
    ]

    if rc["loudnorm"]:
        # -14 LUFS ist der Zielwert, auf den TikTok ohnehin normalisiert.
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

    args += [
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", rc["audio_bitrate"],
        "-movflags", "+faststart",
        str(out_path),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(args, cwd=subtitle_cwd)
    return out_path


def safe_filename(text: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    cleaned = re.sub(r"[\s_]+", "_", cleaned)
    return cleaned[:max_len] or "clip"
