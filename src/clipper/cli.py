"""Command line interface."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from . import pipeline
from .config import ASPECT_PRESETS, OUT_DIR, WORK_DIR, aspect_override, load_config
from .models import ClipPlan
from .pipeline import run_pipeline
from .utils.cache import read_json

# The Windows console defaults to cp1252 and raises on any emoji. TikTok
# captions are basically always text plus emoji, so all three streams are forced
# to UTF-8. stdin belongs in here as much as the output ones: `clipper select
# <id> --from -` reads a JSON body full of captions, and on a cp1252 console
# that decode fails before the pipeline sees a single clip.
for _stream in (sys.stdin, sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

app = typer.Typer(add_completion=False, help="Turn long-form YouTube videos into vertical clips.")
console = Console()


def _merge(base: dict, extra: dict) -> dict:
    """Shallow-merge one section dict into another."""
    for section, values in extra.items():
        base.setdefault(section, {}).update(values)
    return base


def _overrides(**kwargs: Any) -> dict:
    """Build the override dict from the CLI flags that were set."""
    mapping = {
        "clips": ("select", "clips"),
        "min_duration": ("select", "min_duration"),
        "max_duration": ("select", "max_duration"),
        "vision_frames": ("select", "vision_frames"),
        "language": ("transcribe", "language"),
        "whisper_model": ("transcribe", "model"),
    }
    out: dict[str, dict] = {}
    for name, value in kwargs.items():
        if value is None or name not in mapping:
            continue
        section, key = mapping[name]
        out.setdefault(section, {})[key] = value
    return out


def _print_table(plans: list[ClipPlan]) -> None:
    # The trimmed column only appears when something was actually cut - on a run
    # with dead-air removal off it would be a column of zeros.
    trimmed = any(p.timing is not None and not p.timing.is_identity for p in plans)

    table = Table(title="Finished clips", show_lines=False)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Length", justify="right")
    if trimmed:
        table.add_column("Cut", justify="right", style="dim")
    table.add_column("Hook")
    table.add_column("File", style="dim")

    for plan in plans:
        cand = plan.candidate
        # The length that matters is the one the viewer sees, not the source
        # window it was taken from.
        length = plan.timing.duration if plan.timing else cand.duration
        row = [str(plan.index), f"{cand.score:.0f}", f"{length:.0f}s"]
        if trimmed:
            cut = plan.timing.removed if plan.timing else 0.0
            row.append(f"-{cut:.1f}s" if cut > 0.05 else "-")
        row += [
            cand.hook[:44] or cand.title[:44],
            Path(plan.out_path).name if plan.out_path else "-",
        ]
        table.add_row(*row)
    console.print(table)
    if plans and plans[0].out_path:
        console.print(f"\nOutput: [bold]{Path(plans[0].out_path).parent}[/bold]")


@app.command()
def run(
    url: str = typer.Argument(..., help="YouTube URL"),
    clips: int | None = typer.Option(None, "--clips", "-n", help="Number of clips"),
    min_duration: float | None = typer.Option(None, "--min", help="Minimum length in seconds"),
    max_duration: float | None = typer.Option(None, "--max", help="Maximum length in seconds"),
    language: str | None = typer.Option(None, "--lang", help="Language, e.g. en or de"),
    whisper_model: str | None = typer.Option(None, "--whisper", help="Whisper model"),
    vision_frames: int | None = typer.Option(
        None, "--vision", help="Keyframes sent to the model (0 = off)"
    ),
    aspect: str | None = typer.Option(
        None, "--aspect", help=f"Output format: {' | '.join(ASPECT_PRESETS)}"
    ),
    no_captions: bool = typer.Option(False, "--no-captions", help="Without captions"),
    no_silence: bool = typer.Option(
        False, "--no-silence", help="Keep speech pauses instead of cutting them out"
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Heuristic selection instead of Claude (no API key needed)"
    ),
    reselect: bool = typer.Option(
        False, "--reselect", help="Recompute only the selection, keep the rest of the cache"
    ),
    force: bool = typer.Option(False, "--force", help="Discard all caches"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Custom YAML config"),
) -> None:
    """Process a video from URL to finished clips."""
    overrides = _overrides(
        clips=clips, min_duration=min_duration, max_duration=max_duration,
        language=language, whisper_model=whisper_model, vision_frames=vision_frames,
    )
    if no_captions:
        _merge(overrides, {"captions": {"enabled": False}})
    if no_silence:
        _merge(overrides, {"silence": {"enabled": False}})
    if aspect:
        _merge(overrides, aspect_override(aspect))
    cfg = load_config(config, overrides)

    use_llm = not no_llm
    if use_llm and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        console.print(
            "[yellow]No ANTHROPIC_API_KEY set - falling back to heuristic "
            "selection.[/yellow]\n"
            "[dim]Better without a key: run 'clipper analyze' and let a Claude "
            "Code session do the selection.[/dim]"
        )
        use_llm = False

    result = run_pipeline(url, cfg, force=force, use_llm=use_llm, reselect=reselect)

    if not result.plans:
        raise typer.Exit(code=1)

    _print_table(result.plans)
    if not result.ok:
        # Exit non-zero even though clips were produced. A partial run that
        # reports success is how you end up uploading 29 of 34 clips and never
        # noticing the other five.
        raise typer.Exit(code=1)


@app.command()
def analyze(
    url: str = typer.Argument(..., help="YouTube URL"),
    clips: int | None = typer.Option(None, "--clips", "-n", help="Number of clips"),
    min_duration: float | None = typer.Option(None, "--min", help="Minimum length in seconds"),
    max_duration: float | None = typer.Option(None, "--max", help="Maximum length in seconds"),
    language: str | None = typer.Option(None, "--lang", help="Language, e.g. en or de"),
    whisper_model: str | None = typer.Option(None, "--whisper", help="Whisper model"),
    vision_frames: int | None = typer.Option(
        None, "--vision", help="Number of keyframes for the briefing"
    ),
    aspect: str | None = typer.Option(
        None, "--aspect", help=f"Output format: {' | '.join(ASPECT_PRESETS)}"
    ),
    force: bool = typer.Option(False, "--force", help="Discard all caches"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Custom YAML config"),
) -> None:
    """Stages 1-4 plus keyframes. Writes a briefing for moment selection.

    Needs no API key: the briefing is meant to be read and answered by a Claude
    Code session.
    """
    overrides = _overrides(
        clips=clips, min_duration=min_duration, max_duration=max_duration,
        language=language, whisper_model=whisper_model, vision_frames=vision_frames,
    )
    if aspect:
        _merge(overrides, aspect_override(aspect))
    cfg = load_config(config, overrides)

    analysis = pipeline.analyze(url, cfg, force=force)
    brief = pipeline.write_brief(analysis, cfg)

    console.print(
        f"\n[bold green]Briefing ready[/bold green]\n"
        f"  Video ID: [bold]{analysis.source.video_id}[/bold]\n"
        f"  Briefing: [bold]{brief}[/bold]\n"
    )
    console.print(
        "Next: read the briefing, write the clips as JSON, then\n"
        f"  [dim]clipper select {analysis.source.video_id} --from clips.json[/dim]\n"
        f"  [dim]clipper build {analysis.source.video_id}[/dim]"
    )


@app.command("select")
def select_cmd(
    video_id: str = typer.Argument(..., help="Video ID from 'clipper analyze'"),
    from_file: str = typer.Option(
        ..., "--from", help="JSON file with the clips, or '-' for stdin"
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Custom YAML config"),
    clips: int | None = typer.Option(None, "--clips", "-n", help="Number of clips"),
    min_duration: float | None = typer.Option(None, "--min"),
    max_duration: float | None = typer.Option(None, "--max"),
) -> None:
    """Take finished clip proposals and clean them up.

    The proposals run through the same post-processing as the API path: cut
    boundaries, word boundaries, length correction, overlaps.
    """
    cfg = load_config(config, _overrides(
        clips=clips, min_duration=min_duration, max_duration=max_duration
    ))

    text = sys.stdin.read() if from_file == "-" else Path(from_file).read_text("utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # The message is already formatted for the user; a traceback would only
        # add noise. `from None` says that explicitly.
        console.print(f"[red]Not valid JSON: {exc}[/red]")
        raise typer.Exit(code=1) from None

    raw = payload.get("clips", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        console.print("[red]Expected {\"clips\": [...]} or a plain list.[/red]")
        raise typer.Exit(code=1)

    analysis = pipeline.load_analysis(video_id, cfg)
    try:
        cleaned = pipeline.store_candidates(analysis, raw, cfg)
    except ValidationError as exc:
        console.print(f"[red]Clip does not match the schema:[/red]\n{exc}")
        raise typer.Exit(code=1) from None

    console.print(f"[green]{len(cleaned)} clips accepted[/green]")
    for cand in cleaned:
        console.print(
            f"  [cyan]{cand.start:7.2f}-{cand.end:7.2f}[/cyan] "
            f"({cand.duration:4.1f}s) [{cand.score:3.0f}] {cand.title}"
        )

    # Name the dropped clips. When maximising yield that is exactly the
    # actionable information: a clip almost always falls out because after
    # snapping it runs into a higher-scored neighbour - with slightly shifted
    # boundaries it fits on the next attempt.
    kept = {c.title for c in cleaned}
    dropped = [c for c in raw if c.get("title") not in kept]
    if dropped:
        min_score = cfg["select"].get("min_score", 0)
        console.print(f"\n[yellow]{len(dropped)} dropped:[/yellow]")
        for item in dropped:
            score = item.get("score", 0)
            why = (f"score below {min_score:g}" if score < min_score
                   else "overlap, or too short after snapping")
            console.print(
                f"  [dim]{item.get('start', 0):7.2f}-{item.get('end', 0):7.2f} "
                f"[{score:3.0f}] {item.get('title', '?')} - {why}[/dim]"
            )
    console.print(f"\n[dim]clipper build {video_id}[/dim]")


@app.command()
def build(
    video_id: str = typer.Argument(..., help="Video ID from 'clipper analyze'"),
    aspect: str | None = typer.Option(
        None, "--aspect", help=f"Output format: {' | '.join(ASPECT_PRESETS)}"
    ),
    no_captions: bool = typer.Option(False, "--no-captions", help="Without captions"),
    no_silence: bool = typer.Option(
        False, "--no-silence", help="Keep speech pauses instead of cutting them out"
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-render clips that already exist"
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Custom YAML config"),
) -> None:
    """Render the clips from the stored selection.

    Clips that already exist on disk are reused - but only the ones that were
    rendered with the same output settings. Change the aspect ratio, the
    captions or the encoder and they are rebuilt without being asked.
    """
    overrides: dict = {}
    if no_captions:
        _merge(overrides, {"captions": {"enabled": False}})
    if no_silence:
        _merge(overrides, {"silence": {"enabled": False}})
    if aspect:
        _merge(overrides, aspect_override(aspect))
    cfg = load_config(config, overrides or None)

    analysis = pipeline.load_analysis(video_id, cfg)
    candidates = pipeline.load_candidates(video_id)
    result = pipeline.build(analysis, candidates, cfg, force=force)

    if not result.plans:
        raise typer.Exit(code=1)
    _print_table(result.plans)
    if not result.ok:
        raise typer.Exit(code=1)


@app.command("list")
def list_clips(
    video_id: str | None = typer.Argument(None, help="Video ID, or all if omitted"),
) -> None:
    """Show already generated clips with their hook and caption."""
    dirs = [OUT_DIR / video_id] if video_id else sorted(
        d for d in OUT_DIR.glob("*") if d.is_dir()
    )

    found = False
    for directory in dirs:
        manifest = directory / "clips.json"
        if not manifest.exists():
            continue
        found = True
        data = read_json(manifest)
        console.print(f"\n[bold]{data['source']['title']}[/bold] [dim]{directory.name}[/dim]")
        for clip in data["clips"]:
            length = clip.get("output_duration", clip["duration"])
            cut = clip["duration"] - length
            note = f" [dim](-{cut:.0f}s)[/dim]" if cut > 0.5 else ""
            console.print(
                f"  [cyan]{clip['index']:>2}[/cyan] "
                f"[{clip['score']:>3.0f}] {length:>4.0f}s{note}  {clip['hook']}"
            )
            if clip.get("caption"):
                console.print(f"      [dim]{clip['caption']}[/dim]")

    if not found:
        console.print("[yellow]No clips generated yet.[/yellow]")


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


@app.command()
def clean(
    video_id: str | None = typer.Argument(None, help="Video ID, or all if omitted"),
    outputs: bool = typer.Option(
        False, "--outputs", help="Also delete the rendered clips in out/"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation"),
) -> None:
    """Delete cached working data.

    By default only work/ goes - the source video, transcript, shots, energy
    curve and keyframes. Those are all reproducible from the source. The
    rendered clips in out/ are kept unless --outputs is given, because they are
    the actual result and re-rendering them is the expensive part.
    """
    targets: list[Path] = []
    roots = [WORK_DIR] + ([OUT_DIR] if outputs else [])
    for root in roots:
        if not root.exists():
            continue
        if video_id:
            candidate = root / video_id
            if candidate.is_dir():
                targets.append(candidate)
        else:
            targets += [d for d in sorted(root.iterdir()) if d.is_dir()]

    if not targets:
        console.print("[yellow]Nothing to clean.[/yellow]")
        return

    total = 0
    for t in targets:
        size = _dir_size(t)
        total += size
        console.print(f"  [dim]{t.parent.name}/{t.name}[/dim]  {_fmt_size(size)}")
    console.print(f"\n[bold]{len(targets)} directories, {_fmt_size(total)}[/bold]")

    if not yes and not typer.confirm("Delete these?"):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(code=1)

    for t in targets:
        shutil.rmtree(t, ignore_errors=True)
    console.print(f"[green]Freed {_fmt_size(total)}.[/green]")


if __name__ == "__main__":
    app()
