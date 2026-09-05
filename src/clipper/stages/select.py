"""Stufe 5: Momentauswahl.

Baut ein kompaktes "Timeline-Dokument" aus Transkript, Shot-Grenzen und
Audio-Energie, haengt Keyframes als Bilder an und laesst Claude die Clips
waehlen. Ohne API-Key greift eine rein heuristische Auswahl.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ..models import Candidate, EnergyPeak, Segment, Shot, SourceVideo, Word
from .scenes import snap_end_within, snap_to_shots
from .vision import encode_image_block

SYSTEM_PROMPT = """\
Du bist Cutter fuer virale Short-Form-Clips (TikTok, Reels, Shorts).

Du bekommst die Timeline eines Long-Form-Videos: Transkript mit Zeitstempeln,
Shot-Grenzen und eine normalisierte Audio-Energiekurve. Zusaetzlich Keyframes
als Bilder, jeder mit seinem Zeitstempel beschriftet.

Deine Aufgabe: die Momente finden, die als eigenstaendiger Clip funktionieren.

Kriterien, nach Wichtigkeit:
1. Der Clip muss ohne jeden Kontext verstaendlich sein. Wer das Quellvideo nicht
   kennt, muss in den ersten 2 Sekunden begreifen, worum es geht.
2. Es braucht eine Aufloesung - eine Frage die beantwortet wird, ein Versuch der
   gelingt oder scheitert, eine Reaktion die kommt. Ein Clip der aufbaut und
   dann abbricht, verliert den Zuschauer.
3. Der Anfang muss sofort greifen. Kein Vorlauf, kein Anmoderieren.
4. Hohe Audio-Energie korreliert stark mit Reaktionen und Hoehepunkten, ist aber
   allein kein Grund - laute Musik ohne Ereignis ist kein Clip.

Setze start und end IMMER auf die naechstgelegene Shot-Grenze aus der Liste.
Ueberlappende Clips sind nicht erlaubt.

Fuer jeden Clip:
- title: interner Arbeitstitel, kurz
- hook: der Text, der die ersten Sekunden ueber dem Bild steht. Maximal 8 Woerter,
  neugierig machend, kein Clickbait der nicht eingeloest wird.
- caption: fertige TikTok-Caption inkl. 3-5 Hashtags
- reason: eine Zeile, warum dieser Moment funktioniert
- score: 0-100, deine ehrliche Einschaetzung des viralen Potenzials.
  Nutze die Skala wirklich aus - wenn ein Video nur mittelmaessige Momente
  hergibt, vergib auch mittelmaessige Scores.

SPRACHE: title, hook und caption muessen auf {output_language} sein - das ist
der Text, der im Video und unter dem Post steht. Er muss zur Sprache des
Quellmaterials und der Zielgruppe passen, nicht zur Sprache dieser Anweisung.
`reason` ist nur eine interne Notiz und darf deutsch bleiben.
"""


LANGUAGE_NAMES = {
    "en": "Englisch",
    "de": "Deutsch",
    "es": "Spanisch",
    "fr": "Franzoesisch",
    "pt": "Portugiesisch",
}


def system_prompt(cfg: dict) -> str:
    code = cfg["select"].get("output_language", "en")
    return SYSTEM_PROMPT.format(output_language=LANGUAGE_NAMES.get(code, code))


class _LLMClip(BaseModel):
    start: float
    end: float
    title: str
    hook: str = ""
    caption: str = ""
    reason: str = ""
    score: float = 0.0


class _LLMResponse(BaseModel):
    clips: list[_LLMClip] = Field(default_factory=list)


def _assignment(sel: dict, duration: float) -> str:
    """Auftragszeile - feste Stueckzahl oder 'so viele wie moeglich'."""
    span = f"je {sel['min_duration']:.0f}-{sel['max_duration']:.0f} Sekunden"
    if sel.get("clips"):
        return f"Gewuenscht: {sel['clips']} Clips, {span}."
    ceiling = int(duration // max(sel["min_duration"], 1))
    return (
        f"Gewuenscht: SO VIELE Clips wie das Material hergibt, {span}. "
        f"Bis zu {ceiling} passen ueberschneidungsfrei hinein - Obergrenze, "
        f"keine Vorgabe. Gehe das Video von vorne bis hinten durch. Clips "
        f"unter Score {sel.get('min_score', 45)} werden verworfen, also lieber "
        f"ehrlich bewerten als schoenen."
    )


def _transcript_block(segments: list[Segment], max_chars: int = 60_000) -> str:
    lines = [f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments if s.text]
    text = "\n".join(lines)
    if len(text) > max_chars:
        # Nicht still abschneiden - dem Modell sagen, dass etwas fehlt.
        text = text[:max_chars] + "\n[... Transkript gekuerzt ...]"
    return text


def _shots_block(shots: list[Shot], limit: int = 400) -> str:
    if len(shots) <= limit:
        chosen = shots
        note = ""
    else:
        # Bei sehr schnittintensiven Videos nur die laengeren Shots zeigen -
        # Ein-Sekunden-Shots sind ohnehin keine sinnvollen Clipgrenzen.
        chosen = sorted(shots, key=lambda s: s.duration, reverse=True)[:limit]
        chosen = sorted(chosen, key=lambda s: s.start)
        note = f" (nur die {limit} laengsten von {len(shots)} Shots)"
    return f"Shot-Grenzen in Sekunden{note}:\n" + ", ".join(
        f"{s.start:.2f}" for s in chosen
    )


def _energy_block(times: np.ndarray, energy: np.ndarray, step: float = 2.0) -> str:
    if len(times) == 0:
        return "Keine Audiodaten."
    if len(times) > 1:
        stride = max(1, int(step / max(times[1] - times[0], 1e-6)))
    else:
        stride = 1
    pairs = [f"{times[i]:.0f}:{energy[i]:.2f}" for i in range(0, len(times), stride)]
    return "Audio-Energie (Sekunde:Wert 0-1):\n" + " ".join(pairs)


def select_with_llm(
    source: SourceVideo,
    segments: list[Segment],
    shots: list[Shot],
    times: np.ndarray,
    energy: np.ndarray,
    keyframes: list[tuple[float, Path]],
    cfg: dict,
) -> list[Candidate]:
    import anthropic

    sel = cfg["select"]
    client = anthropic.Anthropic()

    content: list[dict] = [{
        "type": "text",
        "text": (
            f"Video: {source.title}\n"
            f"Kanal: {source.channel}\n"
            f"Laenge: {source.duration:.0f}s\n\n"
            f"{_assignment(sel, source.duration)}\n\n"
            f"--- TRANSKRIPT ---\n{_transcript_block(segments)}\n\n"
            f"--- {_shots_block(shots)}\n\n"
            f"--- {_energy_block(times, energy)}\n"
        ),
    }]

    if keyframes:
        content.append({
            "type": "text",
            "text": f"\n--- {len(keyframes)} KEYFRAMES (chronologisch) ---",
        })
        for t, path in keyframes:
            content.append({"type": "text", "text": f"t={t:.1f}s"})
            content.append(encode_image_block(path))

    response = client.messages.parse(
        model=sel["model"],
        max_tokens=16000,
        system=system_prompt(cfg),
        thinking={"type": "adaptive"},
        output_config={"effort": sel.get("effort", "high")},
        messages=[{"role": "user", "content": content}],
        output_format=_LLMResponse,
    )

    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", "unbekannt")
        raise RuntimeError(
            f"Das Modell hat die Anfrage abgelehnt ({category}). "
            "Mit --no-llm laeuft die heuristische Auswahl."
        )

    parsed = response.parsed_output
    clips = parsed.clips if parsed else []
    return _finalize(
        [
            Candidate(
                start=c.start, end=c.end, title=c.title, hook=c.hook,
                caption=c.caption, reason=c.reason, score=c.score,
            )
            for c in clips
        ],
        shots,
        cfg,
        segments,
    )


def select_heuristic(
    segments: list[Segment],
    shots: list[Shot],
    times: np.ndarray,
    energy: np.ndarray,
    peaks: list[EnergyPeak],
    cfg: dict,
) -> list[Candidate]:
    """Fallback ohne API-Key: Fenster um die staerksten Energie-Spitzen.

    Deutlich schlechter als die LLM-Auswahl, weil rein akustisch - aber
    brauchbar zum Testen der restlichen Pipeline.
    """
    sel = cfg["select"]
    target = (sel["min_duration"] + sel["max_duration"]) / 2
    candidates: list[Candidate] = []

    for peak in sorted(peaks, key=lambda p: p.score, reverse=True):
        # Peak leicht nach hinten setzen: die Reaktion ist interessanter als der Aufbau.
        start = max(0.0, peak.t - target * 0.65)
        end = start + target
        if any(not (end <= c.start or start >= c.end) for c in candidates):
            continue
        text = " ".join(
            s.text for s in segments if s.start < end and s.end > start
        ).strip()
        # Bewusst kein Hook: die Heuristik kennt nur Lautstaerke und koennte
        # hoechstens den Transkriptanfang wiederholen - der laeuft ohnehin
        # schon als Untertitel. Ein leerer Hook unterdrueckt das Overlay.
        preview = " ".join(text.split()[:8])
        candidates.append(
            Candidate(
                start=start, end=end,
                title=f"Peak @ {peak.t:.0f}s",
                caption=preview,
                hook="",
                reason=f"Audio-Energie {peak.score:.2f}",
                score=peak.score * 100,
            )
        )
        if sel.get("clips") and len(candidates) >= sel["clips"]:
            break

    return _finalize(candidates, shots, cfg, segments)


def _avoid_word_split(
    t: float, words: list[Word], *, edge: str, max_drift: float = 0.4
) -> float:
    """Verschiebt einen Schnittpunkt aus einem laufenden Wort heraus.

    Ein Schnitt mitten in einer Silbe faellt sofort als Fehler auf, auch wenn
    er sauber auf einer Bildgrenze sitzt. Die Verschiebung bleibt klein, damit
    die Shot-Ausrichtung nicht nennenswert verloren geht.
    """
    for word in words:
        if word.start < t < word.end:
            target = word.start if edge == "start" else word.end
            if abs(target - t) <= max_drift:
                return target
            break
    return t


def _finalize(
    candidates: list[Candidate],
    shots: list[Shot],
    cfg: dict,
    segments: list[Segment] | None = None,
) -> list[Candidate]:
    """Auf Shot- und Wortgrenzen ziehen, Laenge einhalten, Ueberlappungen entfernen."""
    sel = cfg["select"]
    min_d, max_d = sel["min_duration"], sel["max_duration"]
    min_score = sel.get("min_score", 0)
    words = [w for s in (segments or []) for w in s.words]

    def place(
        cand: Candidate, edge: str, taken: list[Candidate] | None = None
    ) -> tuple[float, float]:
        start = snap_to_shots(cand.start, shots, edge=edge)
        hard_max = None
        if taken:
            # In die freie Luecke zwischen den schon vergebenen Clips einpassen.
            start = max(start, max((c.end for c in taken if c.end <= cand.end), default=0.0))
            hard_max = min((c.start for c in taken if c.start >= start), default=None)
        # Laenge und Schnittgrenze gemeinsam loesen statt nacheinander.
        end = snap_end_within(start, cand.end, shots, min_d, max_d, hard_max=hard_max)
        if words:
            start = _avoid_word_split(start, words, edge="start")
            end = _avoid_word_split(end, words, edge="end")
        return start, end

    cleaned: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        # Schwache Clips fruehzeitig aussortieren. Bei unbegrenzter Clipzahl
        # ist der Score die einzige Qualitaetsbremse - ohne sie wuerde jedes
        # belanglose Fenster gerendert.
        if cand.score < min_score:
            continue

        def usable(bounds: tuple[float, float]) -> bool:
            start, end = bounds
            if start < 0 or end - start < min_d * 0.8:
                return False
            return all(end <= c.start or start >= c.end for c in cleaned)

        placed = place(cand, "start")
        if not usable(placed):
            # Zweiter Versuch mit dem Start auf der naechsten Grenze NACH dem
            # Wunschzeitpunkt. Das Rueckwaerts-Snappen laesst direkt
            # aufeinanderfolgende Momente sonst ineinanderlaufen, und der
            # spaetere verliert - obwohl beide Platz haetten.
            placed = place(cand, "end")
            if not usable(placed):
                # Dritter Versuch: in die tatsaechlich freie Luecke einpassen.
                # Nachbarn dehnen sich beim Snappen auf Schnittgrenzen aus und
                # quetschen einen Moment sonst heraus, fuer den noch Platz ist.
                placed = place(cand, "end", taken=cleaned)
                if not usable(placed):
                    continue

        cand.start, cand.end = placed
        cleaned.append(cand)

    # clips = 0 heisst unbegrenzt: alles nehmen, was Score, Laenge und
    # Ueberlappung ueberlebt hat. Die Auswahl lief nach Score absteigend, damit
    # bei Ueberlappung der staerkere Clip gewinnt; ausgegeben wird chronologisch.
    limit = sel.get("clips") or 0
    cleaned.sort(key=lambda c: c.score, reverse=True)
    if limit > 0:
        cleaned = cleaned[:limit]
    cleaned.sort(key=lambda c: c.start)
    return cleaned
