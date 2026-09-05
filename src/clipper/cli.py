"""Command line interface."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from . import pipeline
from .config import OUT_DIR, load_config
from .models import ClipPlan
from .pipeline import run_pipeline
from .utils.cache import read_json

# The Windows console defaults to cp1252 and raises UnicodeEncodeError on any
# emoji. TikTok captions are basically always text plus emoji, so output is
# forced to UTF-8. 'replace' means an old console shows replacement characters
# at worst instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

app = typer.Typer(add_completion=False, help="Turn long-form YouTube videos into vertical clips.")
console = Console()


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
    table = Table(title="Finished clips", show_lines=False)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Length", justify="right")
    table.add_column("Hook")
    table.add_column("File", style="dim")

    for plan in plans:
        cand = plan.candidate
        table.add_row(
            str(plan.index),
            f"{cand.score:.0f}",
            f"{cand.duration:.0f}s",
            cand.hook[:44] or cand.title[:44],
            Path(plan.out_path).name if plan.out_path else "-",
        )
    console.print(table)
    if plans and plans[0].out_path:
        console.print(f"\nOutput: [bold]{Path(plans[0].out_path).parent}[/bold]")


@app.command()
def run(
    url: str = typer.Argument(..., help="YouTube URL"),
    clips: Optional[int] = typer.Option(None, "--clips", "-n", help="Number of clips"),
    min_duration: Optional[float] = typer.Option(None, "--min", help="Minimum length in seconds"),
    max_duration: Optional[float] = typer.Option(None, "--max", help="Maximum length in seconds"),
    language: Optional[str] = typer.Option(None, "--lang", help="Language, e.g. en or de"),
    whisper_model: Optional[str] = typer.Option(None, "--whisper", help="Whisper model"),
    vision_frames: Optional[int] = typer.Option(
        None, "--vision", help="Keyframes sent to the model (0 = off)"
    ),
    no_captions: bool = typer.Option(False, "--no-captions", help="Without captions"),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Heuristic selection instead of Claude (no API key needed)"
    ),
    reselect: bool = typer.Option(
        False, "--reselect", help="Recompute only the selection, keep the rest of the cache"
    ),
    force: bool = typer.Option(False, "--force", help="Discard all caches"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom YAML config"),
) -> None:
    """Process a video from URL to finished clips."""
    overrides = _overrides(
        clips=clips, min_duration=min_duration, max_duration=max_duration,
        language=language, whisper_model=whisper_model, vision_frames=vision_frames,
    )
    if no_captions:
        overrides["captions"] = {"enabled": False}
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

    plans = run_pipeline(url, cfg, force=force, use_llm=use_llm, reselect=reselect)

    if not plans:
        raise typer.Exit(code=1)

    _print_table(plans)


@app.command()
def analyze(
    url: str = typer.Argument(..., help="YouTube URL"),
    clips: Optional[int] = typer.Option(None, "--clips", "-n", help="Number of clips"),
    min_duration: Optional[float] = typer.Option(None, "--min", help="Minimum length in seconds"),
    max_duration: Optional[float] = typer.Option(None, "--max", help="Maximum length in seconds"),
    language: Optional[str] = typer.Option(None, "--lang", help="Language, e.g. en or de"),
    whisper_model: Optional[str] = typer.Option(None, "--whisper", help="Whisper model"),
    vision_frames: Optional[int] = typer.Option(
        None, "--vision", help="Number of keyframes for the briefing"
    ),
    force: bool = typer.Option(False, "--force", help="Discard all caches"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom YAML config"),
) -> None:
    """Stages 1-4 plus keyframes. Writes a briefing for moment selection.

    Needs no API key: the briefing is meant to be read and answered by a Claude
    Code session.
    """
    cfg = load_config(config, _overrides(
        clips=clips, min_duration=min_duration, max_duration=max_duration,
        language=language, whisper_model=whisper_model, vision_frames=vision_frames,
    ))

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
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom YAML config"),
    clips: Optional[int] = typer.Option(None, "--clips", "-n", help="Number of clips"),
    min_duration: Optional[float] = typer.Option(None, "--min"),
    max_duration: Optional[float] = typer.Option(None, "--max"),
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
        console.print(f"[red]Not valid JSON: {exc}[/red]")
        raise typer.Exit(code=1)

    raw = payload.get("clips", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        console.print("[red]Expected {\"clips\": [...]} or a plain list.[/red]")
        raise typer.Exit(code=1)

    analysis = pipeline.load_analysis(video_id)
    try:
        cleaned = pipeline.store_candidates(analysis, raw, cfg)
    except ValidationError as exc:
        console.print(f"[red]Clip does not match the schema:[/red]\n{exc}")
        raise typer.Exit(code=1)

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
            why = ("score below %g" % min_score if score < min_score
                   else "overlap, or too short after snapping")
            console.print(
                f"  [dim]{item.get('start', 0):7.2f}-{item.get('end', 0):7.2f} "
                f"[{score:3.0f}] {item.get('title', '?')} - {why}[/dim]"
            )
    console.print(f"\n[dim]clipper build {video_id}[/dim]")


@app.command()
def build(
    video_id: str = typer.Argument(..., help="Video ID from 'clipper analyze'"),
    no_captions: bool = typer.Option(False, "--no-captions", help="Without captions"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom YAML config"),
) -> None:
    """Render the clips from the stored selection."""
    overrides = {"captions": {"enabled": False}} if no_captions else None
    cfg = load_config(config, overrides)

    analysis = pipeline.load_analysis(video_id)
    candidates = pipeline.load_candidates(video_id)
    plans = pipeline.build(analysis, candidates, cfg)

    if not plans:
        raise typer.Exit(code=1)
    _print_table(plans)


@app.command("list")
def list_clips(
    video_id: Optional[str] = typer.Argument(None, help="Video ID, or all if omitted"),
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
            console.print(
                f"  [cyan]{clip['index']:>2}[/cyan] "
                f"[{clip['score']:>3.0f}] {clip['duration']:>4.0f}s  {clip['hook']}"
            )
            if clip.get("caption"):
                console.print(f"      [dim]{clip['caption']}[/dim]")

    if not found:
        console.print("[yellow]No clips generated yet.[/yellow]")


if __name__ == "__main__":
    app()
