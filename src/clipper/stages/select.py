"""Stage 5: moment selection.

Builds a compact "timeline document" from the transcript, shot boundaries and
audio energy, attaches keyframes as images, and lets Claude pick the clips.
Without an API key a purely heuristic selection takes over.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ..models import Candidate, EnergyPeak, Segment, Shot, SourceVideo, Word
from .scenes import snap_end_within, snap_to_shots
from .vision import encode_image_block

SYSTEM_PROMPT = """\
You are an editor cutting viral short-form clips (TikTok, Reels, Shorts).

You receive the timeline of a long-form video: a timestamped transcript, shot
boundaries and a normalised audio energy curve. Plus keyframes as images, each
labelled with its timestamp.

Your task: find the moments that work as a standalone clip.

Criteria, in order of importance:
1. The clip must make sense with zero context. Someone who has never seen the
   source video must grasp what is going on within the first 2 seconds.
2. It needs a resolution - a question that gets answered, an attempt that
   succeeds or fails, a reaction that lands. A clip that builds up and then
   cuts away loses the viewer.
3. The opening must grab immediately. No run-up, no introduction.
4. High audio energy correlates strongly with reactions and payoffs, but is
   not a reason on its own - loud music without an event is not a clip.

ALWAYS place start and end on the nearest shot boundary from the list.
Overlapping clips are not allowed.

For each clip:
- title: short internal working title
- hook: the text shown over the frame for the first few seconds. At most 8
  words, makes the viewer curious, no clickbait you do not pay off.
- caption: a finished TikTok caption including 3-5 hashtags
- reason: one line on why this moment works
- score: 0-100, your honest estimate of viral potential. Actually use the
  range - if a video only yields mediocre moments, hand out mediocre scores.

LANGUAGE: title, hook and caption must be in {output_language} - that is the
text shown in the video and under the post. It has to match the language of the
source material and the target audience, not the language of these
instructions. `reason` is an internal note and may be in any language.
"""


LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "pt": "Portuguese",
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
    """The assignment line - a fixed count, or as many clips as possible."""
    span = f"{sel['min_duration']:.0f}-{sel['max_duration']:.0f} seconds each"
    if sel.get("clips"):
        return f"Wanted: {sel['clips']} clips, {span}."
    ceiling = int(duration // max(sel["min_duration"], 1))
    return (
        f"Wanted: AS MANY clips as the material supports, {span}. "
        f"Up to {ceiling} fit without overlapping - that is a ceiling, not a "
        f"target. Work through the video from start to finish. Clips below "
        f"score {sel.get('min_score', 45)} are discarded, so score honestly "
        f"rather than generously."
    )


def _transcript_block(segments: list[Segment], max_chars: int = 60_000) -> str:
    lines = [f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments if s.text]
    text = "\n".join(lines)
    if len(text) > max_chars:
        # Do not truncate silently - tell the model something is missing.
        text = text[:max_chars] + "\n[... transcript truncated ...]"
    return text


def _shots_block(shots: list[Shot], limit: int = 400) -> str:
    if len(shots) <= limit:
        chosen = shots
        note = ""
    else:
        # On very cut-heavy videos show only the longer shots - one-second
        # shots are not sensible clip boundaries anyway.
        chosen = sorted(shots, key=lambda s: s.duration, reverse=True)[:limit]
        chosen = sorted(chosen, key=lambda s: s.start)
        note = f" (only the {limit} longest of {len(shots)} shots)"
    return f"Shot boundaries in seconds{note}:\n" + ", ".join(
        f"{s.start:.2f}" for s in chosen
    )


def _energy_block(times: np.ndarray, energy: np.ndarray, step: float = 2.0) -> str:
    if len(times) == 0:
        return "No audio data."
    if len(times) > 1:
        stride = max(1, int(step / max(times[1] - times[0], 1e-6)))
    else:
        stride = 1
    pairs = [f"{times[i]:.0f}:{energy[i]:.2f}" for i in range(0, len(times), stride)]
    return "Audio energy (second:value 0-1):\n" + " ".join(pairs)


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
            f"Channel: {source.channel}\n"
            f"Length: {source.duration:.0f}s\n\n"
            f"{_assignment(sel, source.duration)}\n\n"
            f"--- TRANSCRIPT ---\n{_transcript_block(segments)}\n\n"
            f"--- {_shots_block(shots)}\n\n"
            f"--- {_energy_block(times, energy)}\n"
        ),
    }]

    if keyframes:
        content.append({
            "type": "text",
            "text": f"\n--- {len(keyframes)} KEYFRAMES (chronological) ---",
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
        category = getattr(response.stop_details, "category", "unknown")
        raise RuntimeError(
            f"The model refused the request ({category}). "
            "Use --no-llm to fall back to heuristic selection."
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
    peaks: list[EnergyPeak],
    cfg: dict,
) -> list[Candidate]:
    """Fallback without an API key: windows around the strongest energy peaks.

    Considerably worse than the LLM selection because it is purely acoustic -
    but good enough to exercise the rest of the pipeline.
    """
    sel = cfg["select"]
    target = (sel["min_duration"] + sel["max_duration"]) / 2
    candidates: list[Candidate] = []

    for peak in sorted(peaks, key=lambda p: p.score, reverse=True):
        # Shift the peak slightly back: the reaction beats the build-up.
        start = max(0.0, peak.t - target * 0.65)
        end = start + target
        if any(not (end <= c.start or start >= c.end) for c in candidates):
            continue
        text = " ".join(
            s.text for s in segments if s.start < end and s.end > start
        ).strip()
        # Deliberately no hook: the heuristic only knows loudness and could at
        # best repeat the start of the transcript - which already runs as a
        # caption anyway. An empty hook suppresses the overlay.
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
    """Move a cut point out of a word that is still being spoken.

    A cut in the middle of a syllable reads as a mistake immediately, even when
    it sits cleanly on a frame boundary. The shift stays small so the shot
    alignment is not meaningfully lost.
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
    """Snap to shot and word boundaries, honour length, resolve overlaps."""
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
            # Fit into the free gap between the clips already placed.
            start = max(start, max((c.end for c in taken if c.end <= cand.end), default=0.0))
            hard_max = min((c.start for c in taken if c.start >= start), default=None)
        # Solve length and cut boundary together rather than one after the other.
        end = snap_end_within(start, cand.end, shots, min_d, max_d, hard_max=hard_max)
        if words:
            # The drift only ever grows a clip - the start moves back to the
            # word's beginning, the end forward to its end. Apply each shift
            # only while it keeps the clip inside max_duration: running long is
            # worse than a cut landing a fraction of a second inside a word.
            shifted = _avoid_word_split(start, words, edge="start")
            if end - shifted <= max_d:
                start = shifted
            shifted = _avoid_word_split(end, words, edge="end")
            if shifted - start <= max_d:
                end = shifted
        return start, end

    cleaned: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        # Drop weak clips early. With an unlimited clip count the score is the
        # only quality brake - without it every unremarkable window would get
        # rendered.
        if cand.score < min_score:
            continue

        def usable(bounds: tuple[float, float]) -> bool:
            start, end = bounds
            # Both ends are hard. The floor used to carry a 0.8 slack factor,
            # which quietly let clips ship up to 20% under min_duration - at
            # the tail of the video, where no boundary sits far enough out,
            # that is exactly where it bit.
            if start < 0 or not min_d <= end - start <= max_d:
                return False
            return all(end <= c.start or start >= c.end for c in cleaned)

        placed = place(cand, "start")
        if not usable(placed):
            # Second attempt with the start on the next boundary AFTER the
            # desired time. Snapping backwards otherwise makes directly
            # consecutive moments run into each other, and the later one loses -
            # even though there is room for both.
            placed = place(cand, "end")
            if not usable(placed):
                # Third attempt: fit into the actually free gap. Neighbours
                # expand when snapping to cut boundaries and would otherwise
                # squeeze out a moment there is still room for.
                placed = place(cand, "end", taken=cleaned)
                if not usable(placed):
                    continue

        cand.start, cand.end = placed
        cleaned.append(cand)

    # clips = 0 means unlimited: take everything that survived score, length
    # and overlap. Selection ran by descending score so the stronger clip wins a
    # collision; output is chronological.
    limit = sel.get("clips") or 0
    cleaned.sort(key=lambda c: c.score, reverse=True)
    if limit > 0:
        cleaned = cleaned[:limit]
    cleaned.sort(key=lambda c: c.start)
    return cleaned
