"""Orchestrierung aller Stufen.

Die Pipeline ist an der Momentauswahl geteilt:

    analyze  -> Stufen 1-4 + Keyframes, schreibt einen Briefing-Text
    select   -> nimmt fertige Clipvorschlaege entgegen und raeumt sie auf
    build    -> Stufen 6-8, rendert aus den Vorschlaegen die Clips

Damit kann die Auswahl entweder ueber die API laufen (`run`, braucht einen
Key) oder von einer Claude-Code-Session uebernommen werden, die das Briefing
liest und die Vorschlaege selbst schreibt - dann faellt kein API-Key an.

Jede Stufe cached in `work/<video_id>/`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console

from .config import OUT_DIR, WORK_DIR
from .models import Candidate, ClipPlan, EnergyPeak, Segment, Shot, SourceVideo
from .stages import audio, captions, ingest, reframe, render, scenes, select, transcribe
from .utils.cache import read_json, write_json
from .utils.ffmpeg import ensure_ffmpeg

console = Console()


def _step(label: str) -> None:
    console.print(f"[bold cyan]>[/bold cyan] {label}")


@dataclass
class Analysis:
    """Alles, was vor der Momentauswahl feststeht."""

    source: SourceVideo
    segments: list[Segment]
    shots: list[Shot]
    times: np.ndarray
    energy: np.ndarray
    peaks: list[EnergyPeak]

    @property
    def work(self) -> Path:
        return WORK_DIR / self.source.video_id


# --------------------------------------------------------------------------
# Stufen 1-4
# --------------------------------------------------------------------------

def analyze(url: str, cfg: dict, *, force: bool = False) -> Analysis:
    ensure_ffmpeg()

    _step("Video wird geladen")
    source = ingest.download(url, WORK_DIR, cfg, force=force)
    work = WORK_DIR / source.video_id
    source_path = Path(source.path)
    write_json(work / "source.json", source.model_dump())
    console.print(
        f"  [dim]{source.title}[/dim]\n"
        f"  [dim]{source.width}x{source.height} @ {source.fps:.2f}fps, "
        f"{source.duration:.0f}s[/dim]"
    )

    transcript_path = work / "transcript.json"
    if transcript_path.exists() and not force:
        _step("Transkript aus Cache")
        segments = [Segment(**s) for s in read_json(transcript_path)]
    else:
        _step("Transkription laeuft (Whisper)")
        segments = transcribe.transcribe(source_path, cfg)
        write_json(transcript_path, [s.model_dump() for s in segments])
    console.print(f"  [dim]{len(segments)} Segmente[/dim]")

    shots_path = work / "shots.json"
    if shots_path.exists() and not force:
        _step("Shots aus Cache")
        shots = [Shot(**s) for s in read_json(shots_path)]
    else:
        _step("Shot-Erkennung")
        shots = scenes.detect_shots(source_path, cfg)
        write_json(shots_path, [s.model_dump() for s in shots])
    avg = source.duration / max(len(shots), 1)
    console.print(f"  [dim]{len(shots)} Shots, im Schnitt {avg:.1f}s[/dim]")

    energy_path = work / "energy.npz"
    if energy_path.exists() and not force:
        _step("Audio-Energie aus Cache")
        loaded = np.load(energy_path)
        times, energy = loaded["times"], loaded["energy"]
    else:
        _step("Audio-Analyse")
        times, energy = audio.energy_envelope(source_path, cfg)
        np.savez_compressed(energy_path, times=times, energy=energy)
    peaks = audio.find_peaks(times, energy, cfg)
    console.print(f"  [dim]{len(peaks)} Energie-Spitzen[/dim]")

    return Analysis(source, segments, shots, times, energy, peaks)


def load_analysis(video_id: str) -> Analysis:
    """Laedt eine frueher berechnete Analyse aus dem Cache."""
    work = WORK_DIR / video_id
    if not (work / "source.json").exists():
        raise FileNotFoundError(
            f"Keine Analyse fuer '{video_id}' in {work}. Erst 'clipper analyze' laufen lassen."
        )
    source = SourceVideo(**read_json(work / "source.json"))
    segments = [Segment(**s) for s in read_json(work / "transcript.json")]
    shots = [Shot(**s) for s in read_json(work / "shots.json")]
    loaded = np.load(work / "energy.npz")
    times, energy = loaded["times"], loaded["energy"]
    peaks = audio.find_peaks(times, energy, {"audio": {"peak_percentile": 90}})
    return Analysis(source, segments, shots, times, energy, peaks)


# --------------------------------------------------------------------------
# Briefing fuer eine externe Auswahl (Claude-Code-Session statt API)
# --------------------------------------------------------------------------

def _assignment_line(sel: dict, duration: float) -> str:
    """Formuliert den Auftrag - Stueckzahl oder 'so viele wie moeglich'."""
    span = f"je {sel['min_duration']:.0f}-{sel['max_duration']:.0f} Sekunden"
    if sel.get("clips"):
        return f"Waehle {sel['clips']} Clips, {span}."
    # Grober Anhaltspunkt, wie viele ueberschneidungsfrei ueberhaupt Platz haben.
    ceiling = int(duration // max(sel["min_duration"], 1))
    return (
        f"Waehle SO VIELE Clips wie das Material hergibt, {span}. "
        f"Rein rechnerisch passen bis zu {ceiling} ueberschneidungsfreie Clips "
        f"in {duration:.0f}s - das ist eine Obergrenze, keine Vorgabe. "
        f"Gehe das Video von vorne bis hinten durch und nimm jeden Moment mit, "
        f"der fuer sich steht. Clips unter Score {sel.get('min_score', 45)} "
        f"werden ohnehin verworfen, also lieber ehrlich bewerten als schoenen."
    )


def write_brief(analysis: Analysis, cfg: dict, *, keyframes: int | None = None) -> Path:
    """Schreibt alles, was fuer die Momentauswahl noetig ist, als Markdown.

    Der Text ist bewusst so knapp gehalten, dass er samt Keyframes in ein
    Kontextfenster passt.
    """
    from .stages import vision

    sel = cfg["select"]
    n_frames = sel["vision_frames"] if keyframes is None else keyframes
    source = analysis.source
    work = analysis.work

    frames: list[tuple[float, Path]] = []
    if n_frames > 0:
        _step(f"Keyframes werden extrahiert ({n_frames})")
        kf_times = vision.pick_keyframe_times(source.duration, analysis.peaks, n_frames)
        frames = vision.extract_keyframes(
            Path(source.path), kf_times, work / "keyframes"
        )

    top_peaks = sorted(analysis.peaks, key=lambda p: p.score, reverse=True)[:40]
    top_peaks.sort(key=lambda p: p.t)

    lines: list[str] = [
        f"# {source.title}",
        "",
        f"- Kanal: {source.channel}",
        f"- Laenge: {source.duration:.0f}s",
        f"- Aufloesung: {source.width}x{source.height} @ {source.fps:.2f}fps",
        f"- Video-ID: `{source.video_id}`",
        "",
        "## Auftrag",
        "",
        _assignment_line(sel, source.duration),
        "",
        select.system_prompt(cfg).strip(),
        "",
        "Ergebnis als JSON in dieser Form:",
        "",
        "```json",
        '{"clips": [{"start": 12.5, "end": 38.0, "title": "...", "hook": "...",',
        '            "caption": "...", "reason": "...", "score": 82}]}',
        "```",
        "",
        "Danach einspielen mit:",
        "",
        "```bash",
        f"clipper select {source.video_id} --from clips.json",
        f"clipper build {source.video_id}",
        "```",
        "",
        "## Shot-Grenzen (Sekunden)",
        "",
        select._shots_block(analysis.shots).split("\n", 1)[1],
        "",
        "## Energie-Spitzen (staerkste 40)",
        "",
        " ".join(f"{p.t:.0f}s:{p.score:.2f}" for p in top_peaks),
        "",
    ]

    if frames:
        lines += [
            f"## Keyframes ({len(frames)})",
            "",
            "Chronologisch. Die Bilder gehoeren zum Briefing - ohne sie ist die",
            "Auswahl bei Action-Content blind.",
            "",
        ]
        lines += [f"- `{p.as_posix()}` — t={t:.1f}s" for t, p in frames]
        lines.append("")

    lines += ["## Transkript", "", select._transcript_block(analysis.segments), ""]

    path = work / "brief.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Auswahl entgegennehmen
# --------------------------------------------------------------------------

def store_candidates(
    analysis: Analysis, raw: list[dict], cfg: dict
) -> list[Candidate]:
    """Nimmt rohe Clipvorschlaege an und raeumt sie auf.

    Bewusst derselbe `_finalize`-Durchlauf wie beim API-Pfad: von Hand
    gewaehlte Clips bekommen genauso Shot-Snapping, Wortgrenzen-Schutz,
    Laengenkorrektur und Ueberlappungspruefung.
    """
    candidates = [Candidate(**item) for item in raw]
    cleaned = select._finalize(candidates, analysis.shots, cfg, analysis.segments)
    write_json(
        analysis.work / "candidates.json", [c.model_dump() for c in cleaned]
    )
    return cleaned


def load_candidates(video_id: str) -> list[Candidate]:
    path = WORK_DIR / video_id / "candidates.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Keine Clipauswahl fuer '{video_id}'. Erst 'clipper select' laufen lassen."
        )
    return [Candidate(**c) for c in read_json(path)]


# --------------------------------------------------------------------------
# Stufen 6-8
# --------------------------------------------------------------------------

def build(analysis: Analysis, candidates: list[Candidate], cfg: dict) -> list[ClipPlan]:
    ensure_ffmpeg()
    source = analysis.source
    source_path = Path(source.path)
    work = analysis.work

    out_dir = OUT_DIR / source.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_clip(i: int, cand: Candidate) -> ClipPlan:
        crops = reframe.analyze(source_path, cand.start, cand.end, analysis.shots, cfg)

        ass_path = None
        if cfg["captions"]["enabled"]:
            ass_path = captions.build_ass(
                analysis.segments,
                cand.start,
                cand.end,
                work / "subs" / f"clip_{i:03d}.ass",
                cfg,
                hook=cand.hook,
            )

        plan = ClipPlan(
            index=i,
            candidate=cand,
            crops=crops,
            ass_path=str(ass_path) if ass_path else None,
        )
        name = f"{i:03d}_{int(cand.score):03d}_{render.safe_filename(cand.title)}.mp4"
        out_path = out_dir / name
        render.render_clip(source_path, plan, out_path, cfg)
        plan.out_path = str(out_path)
        return plan

    # Parallel, weil beide teuren Schritte den GIL freigeben: OpenCV dekodiert
    # in C, ffmpeg laeuft als eigener Prozess. Ohne NVENC ist libx264 auf der
    # CPU der Engpass, und davon gibt es hier 20 Threads.
    workers = max(1, int(cfg["render"].get("workers", 4)))
    plans: list[ClipPlan] = []
    _step(f"{len(candidates)} Clips werden gerendert ({workers} parallel)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(make_clip, i, cand): (i, cand)
            for i, cand in enumerate(candidates, start=1)
        }
        for future in as_completed(futures):
            i, cand = futures[future]
            try:
                plan = future.result()
            except Exception as exc:
                # Ein kaputter Clip darf die restlichen nicht mitreissen.
                console.print(f"  [red]x[/red] Clip {i} ({cand.title}): {exc}")
                continue
            plans.append(plan)
            console.print(
                f"  [green]->[/green] {Path(plan.out_path).name} "
                f"[dim]({cand.start:.0f}-{cand.end:.0f}s, Score {cand.score:.0f})[/dim]"
            )

    plans.sort(key=lambda p: p.index)

    _write_manifest(out_dir, source, plans)
    return plans


# --------------------------------------------------------------------------
# Durchlauf am Stueck (API-Pfad oder Heuristik)
# --------------------------------------------------------------------------

def run_pipeline(
    url: str,
    cfg: dict,
    *,
    force: bool = False,
    use_llm: bool = True,
    reselect: bool = False,
) -> list[ClipPlan]:
    analysis = analyze(url, cfg, force=force)
    work = analysis.work

    candidates_path = work / "candidates.json"
    if candidates_path.exists() and not force and not reselect:
        _step("Clipauswahl aus Cache")
        cands = [Candidate(**c) for c in read_json(candidates_path)]
    else:
        keyframes: list[tuple[float, Path]] = []
        n_frames = cfg["select"].get("vision_frames", 0)
        if use_llm and n_frames > 0:
            _step(f"Keyframes werden extrahiert ({n_frames})")
            from .stages import vision

            kf_times = vision.pick_keyframe_times(
                analysis.source.duration, analysis.peaks, n_frames
            )
            keyframes = vision.extract_keyframes(
                Path(analysis.source.path), kf_times, work / "keyframes"
            )

        if use_llm:
            _step(f"Clipauswahl ({cfg['select']['model']})")
            cands = select.select_with_llm(
                analysis.source, analysis.segments, analysis.shots,
                analysis.times, analysis.energy, keyframes, cfg,
            )
        else:
            _step("Clipauswahl (heuristisch, ohne LLM)")
            cands = select.select_heuristic(
                analysis.segments, analysis.shots, analysis.times,
                analysis.energy, analysis.peaks, cfg,
            )
        write_json(candidates_path, [c.model_dump() for c in cands])

    if not cands:
        console.print("[yellow]Keine Clips gefunden.[/yellow]")
        return []
    console.print(f"  [dim]{len(cands)} Clips ausgewaehlt[/dim]")

    return build(analysis, cands, cfg)


def _write_manifest(out_dir: Path, source: SourceVideo, plans: list[ClipPlan]) -> None:
    """Sidecar mit Hooks und Captions - das braucht man beim Hochladen."""
    write_json(
        out_dir / "clips.json",
        {
            "source": source.model_dump(),
            "clips": [
                {
                    "index": p.index,
                    "file": Path(p.out_path).name if p.out_path else None,
                    "start": p.candidate.start,
                    "end": p.candidate.end,
                    "duration": round(p.candidate.duration, 2),
                    "score": p.candidate.score,
                    "title": p.candidate.title,
                    "hook": p.candidate.hook,
                    "caption": p.candidate.caption,
                    "reason": p.candidate.reason,
                }
                for p in plans
            ],
        },
    )
