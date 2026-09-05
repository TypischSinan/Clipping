"""Kommandozeile."""

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

# Die Windows-Konsole laeuft per Default auf cp1252 und wirft bei jedem Emoji
# einen UnicodeEncodeError. TikTok-Captions bestehen praktisch immer aus Text
# plus Emoji, also wird die Ausgabe hart auf UTF-8 gestellt. 'replace' sorgt
# dafuer, dass eine alte Konsole hoechstens Ersatzzeichen zeigt statt
# abzustuerzen.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

app = typer.Typer(add_completion=False, help="YouTube-Longform zu vertikalen Clips.")
console = Console()


def _overrides(**kwargs: Any) -> dict:
    """Baut den Override-Dict aus den gesetzten CLI-Flags."""
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
    table = Table(title="Fertige Clips", show_lines=False)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Laenge", justify="right")
    table.add_column("Hook")
    table.add_column("Datei", style="dim")

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
        console.print(f"\nAusgabe: [bold]{Path(plans[0].out_path).parent}[/bold]")


@app.command()
def run(
    url: str = typer.Argument(..., help="YouTube-URL"),
    clips: Optional[int] = typer.Option(None, "--clips", "-n", help="Anzahl Clips"),
    min_duration: Optional[float] = typer.Option(None, "--min", help="Mindestlaenge in s"),
    max_duration: Optional[float] = typer.Option(None, "--max", help="Maximallaenge in s"),
    language: Optional[str] = typer.Option(None, "--lang", help="Sprache, z.B. en oder de"),
    whisper_model: Optional[str] = typer.Option(None, "--whisper", help="Whisper-Modell"),
    vision_frames: Optional[int] = typer.Option(
        None, "--vision", help="Keyframes fuer das Modell (0 = aus)"
    ),
    no_captions: bool = typer.Option(False, "--no-captions", help="Ohne Untertitel"),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Heuristische Auswahl statt Claude (kein API-Key noetig)"
    ),
    reselect: bool = typer.Option(
        False, "--reselect", help="Nur die Auswahl neu rechnen, Cache sonst behalten"
    ),
    force: bool = typer.Option(False, "--force", help="Alle Caches verwerfen"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Eigene YAML"),
) -> None:
    """Verarbeitet ein Video von der URL bis zu fertigen Clips."""
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
            "[yellow]Kein ANTHROPIC_API_KEY gesetzt - es laeuft die heuristische "
            "Auswahl. Fuer die deutlich bessere LLM-Auswahl den Key setzen.[/yellow]"
        )
        use_llm = False

    plans = run_pipeline(url, cfg, force=force, use_llm=use_llm, reselect=reselect)

    if not plans:
        raise typer.Exit(code=1)

    _print_table(plans)


@app.command()
def analyze(
    url: str = typer.Argument(..., help="YouTube-URL"),
    clips: Optional[int] = typer.Option(None, "--clips", "-n", help="Anzahl Clips"),
    min_duration: Optional[float] = typer.Option(None, "--min", help="Mindestlaenge in s"),
    max_duration: Optional[float] = typer.Option(None, "--max", help="Maximallaenge in s"),
    language: Optional[str] = typer.Option(None, "--lang", help="Sprache, z.B. en oder de"),
    whisper_model: Optional[str] = typer.Option(None, "--whisper", help="Whisper-Modell"),
    vision_frames: Optional[int] = typer.Option(
        None, "--vision", help="Anzahl Keyframes fuer das Briefing"
    ),
    force: bool = typer.Option(False, "--force", help="Alle Caches verwerfen"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Eigene YAML"),
) -> None:
    """Stufen 1-4 plus Keyframes. Schreibt ein Briefing zur Momentauswahl.

    Braucht keinen API-Key: das Briefing ist dafuer gedacht, von einer
    Claude-Code-Session gelesen und beantwortet zu werden.
    """
    cfg = load_config(config, _overrides(
        clips=clips, min_duration=min_duration, max_duration=max_duration,
        language=language, whisper_model=whisper_model, vision_frames=vision_frames,
    ))

    analysis = pipeline.analyze(url, cfg, force=force)
    brief = pipeline.write_brief(analysis, cfg)

    console.print(
        f"\n[bold green]Briefing fertig[/bold green]\n"
        f"  Video-ID: [bold]{analysis.source.video_id}[/bold]\n"
        f"  Briefing: [bold]{brief}[/bold]\n"
    )
    console.print(
        "Naechster Schritt: Briefing lesen, Clips als JSON schreiben, dann\n"
        f"  [dim]clipper select {analysis.source.video_id} --from clips.json[/dim]\n"
        f"  [dim]clipper build {analysis.source.video_id}[/dim]"
    )


@app.command("select")
def select_cmd(
    video_id: str = typer.Argument(..., help="Video-ID aus 'clipper analyze'"),
    from_file: str = typer.Option(
        ..., "--from", help="JSON-Datei mit den Clips, oder '-' fuer stdin"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Eigene YAML"),
    clips: Optional[int] = typer.Option(None, "--clips", "-n", help="Anzahl Clips"),
    min_duration: Optional[float] = typer.Option(None, "--min"),
    max_duration: Optional[float] = typer.Option(None, "--max"),
) -> None:
    """Nimmt fertige Clipvorschlaege entgegen und raeumt sie auf.

    Die Vorschlaege durchlaufen dieselbe Nachbearbeitung wie beim API-Pfad:
    Schnittgrenzen, Wortgrenzen, Laengenkorrektur, Ueberlappungen.
    """
    cfg = load_config(config, _overrides(
        clips=clips, min_duration=min_duration, max_duration=max_duration
    ))

    text = sys.stdin.read() if from_file == "-" else Path(from_file).read_text("utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Kein gueltiges JSON: {exc}[/red]")
        raise typer.Exit(code=1)

    raw = payload.get("clips", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        console.print("[red]Erwartet wird {\"clips\": [...]} oder eine Liste.[/red]")
        raise typer.Exit(code=1)

    analysis = pipeline.load_analysis(video_id)
    try:
        cleaned = pipeline.store_candidates(analysis, raw, cfg)
    except ValidationError as exc:
        console.print(f"[red]Clip passt nicht ins Schema:[/red]\n{exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]{len(cleaned)} Clips uebernommen[/green]")
    for cand in cleaned:
        console.print(
            f"  [cyan]{cand.start:7.2f}-{cand.end:7.2f}[/cyan] "
            f"({cand.duration:4.1f}s) [{cand.score:3.0f}] {cand.title}"
        )

    # Verworfene namentlich nennen. Beim Maximieren der Ausbeute ist genau das
    # die handlungsrelevante Information: ein Clip faellt fast immer weg, weil
    # er nach dem Snappen in einen hoeher bewerteten Nachbarn hineinragt - mit
    # etwas verschobenen Grenzen passt er beim naechsten Versuch.
    kept = {c.title for c in cleaned}
    dropped = [c for c in raw if c.get("title") not in kept]
    if dropped:
        min_score = cfg["select"].get("min_score", 0)
        console.print(f"\n[yellow]{len(dropped)} verworfen:[/yellow]")
        for item in dropped:
            score = item.get("score", 0)
            why = ("Score unter %g" % min_score if score < min_score
                   else "Ueberlappung oder zu kurz nach dem Snappen")
            console.print(
                f"  [dim]{item.get('start', 0):7.2f}-{item.get('end', 0):7.2f} "
                f"[{score:3.0f}] {item.get('title', '?')} - {why}[/dim]"
            )
    console.print(f"\n[dim]clipper build {video_id}[/dim]")


@app.command()
def build(
    video_id: str = typer.Argument(..., help="Video-ID aus 'clipper analyze'"),
    no_captions: bool = typer.Option(False, "--no-captions", help="Ohne Untertitel"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Eigene YAML"),
) -> None:
    """Rendert die Clips aus der gespeicherten Auswahl."""
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
    video_id: Optional[str] = typer.Argument(None, help="Video-ID, sonst alle"),
) -> None:
    """Zeigt bereits erzeugte Clips samt Hook und Caption."""
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
        console.print("[yellow]Noch keine Clips erzeugt.[/yellow]")


if __name__ == "__main__":
    app()
