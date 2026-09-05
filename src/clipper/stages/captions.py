"""Stage 7: captions and hook overlay as an ASS file.

Karaoke style (active word coloured and briefly scaled up) rather than block
text: the word-by-word emphasis pulls the eye along with the audio.

The hook deliberately lives in the same ASS file instead of a drawtext filter -
that avoids escaping special characters in the filtergraph and keeps the
typography of both elements in one place.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Segment, Word

# Punctuation stripped from the end of a word. A mid-sentence comma is kept on
# purpose; only what stops being a reading aid at the end of a line is removed.
_TRAILING_PUNCT = re.compile(r"[.,!?;:]+$")


def _fmt_time(seconds: float) -> str:
    """ASS time format: H:MM:SS.cc"""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _header(cfg: dict) -> str:
    cp = cfg["captions"]
    hk = cp.get("hook", {})
    rf = cfg["reframe"]
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {rf['target_width']}
PlayResY: {rf['target_height']}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{cp['font']},{cp['font_size']},{cp['primary_color']},{cp['highlight_color']},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{cp['outline']},{cp['shadow']},2,60,60,{cp['margin_v']},1
Style: Hook,{hk.get('font', cp['font'])},{hk.get('font_size', 110)},{hk.get('color', '&H00FFFFFF')},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{hk.get('outline', 7)},{cp['shadow']},8,70,70,{hk.get('margin_v', 300)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _escape(text: str) -> str:
    """Strip characters ASS would read as an override block."""
    return text.replace("\\", "").replace("{", "").replace("}", "")


def _wrap(text: str, max_chars: int, max_lines: int | None = None) -> str:
    """Wrap text with \\N without breaking words apart.

    Surplus lines are dropped and the last one gets an ellipsis - truncation
    always happens on a word boundary, never mid-word.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        if len(last) + 1 > max_chars and " " in last:
            last = last.rsplit(" ", 1)[0]
        lines[-1] = last + "…"

    return "\\N".join(lines)


def _blocks(words: list[Word], cfg: dict) -> list[list[list[Word]]]:
    """Group words into blocks, each block a list of lines.

    Two reasons to break: the block has max_lines full lines, or a speech pause
    precedes the next word. Without the pause check a line would sit dead on
    screen through a long silence.
    """
    cp = cfg["captions"]
    per_line = cp["max_chars_per_line"]
    max_lines = cp.get("max_lines", 2)
    max_words = cp.get("max_words", 4)
    max_gap = cp.get("max_gap", 0.6)

    # Wrapping happens while grouping, not afterwards. Going by a plain
    # character capacity (per_line * max_lines) would be wrong: greedily wrapped
    # lines leave slack at the end of each line, so a block would then need more
    # lines than allowed.
    blocks: list[list[list[Word]]] = []
    lines: list[list[Word]] = []
    line: list[Word] = []

    def line_width(candidate: list[Word]) -> int:
        return len(" ".join(w.text for w in candidate))

    for i, word in enumerate(words):
        # Word count is the leading constraint: it decides how much is on screen
        # at once. Character width stays purely as an overflow guard.
        if max_words and sum(len(ln) for ln in lines) + len(line) >= max_words:
            if line:
                lines.append(line)
                line = []
            blocks.append(lines)
            lines = []

        if line and line_width(line + [word]) > per_line:
            lines.append(line)
            line = []
            if len(lines) >= max_lines:
                blocks.append(lines)
                lines = []
        line.append(word)

        gap_follows = i + 1 < len(words) and words[i + 1].start - word.end > max_gap
        if gap_follows:
            lines.append(line)
            line = []
            blocks.append(lines)
            lines = []

    if line:
        lines.append(line)
    if lines:
        blocks.append(lines)

    return blocks


def _render_block(lines: list[list[Word]], active: Word, cfg: dict) -> str:
    """Build one event's text: the whole block, active word highlighted."""
    cp = cfg["captions"]
    highlight = cp["highlight_color"]
    pop = cp.get("pop", True)

    rendered_lines: list[str] = []
    for line in lines:
        parts: list[str] = []
        for word in line:
            text = _escape(word.text)
            if word is active:
                if pop:
                    # Enter slightly larger and scale back to normal. The timing
                    # is relative to the event start, i.e. to the word onset.
                    tags = (
                        f"\\c{highlight}&\\fscx112\\fscy112"
                        f"\\t(0,110,\\fscx100\\fscy100)"
                    )
                else:
                    tags = f"\\c{highlight}&"
                parts.append(f"{{{tags}}}{text}{{\\r}}")
            else:
                parts.append(text)
        rendered_lines.append(" ".join(parts))
    return "\\N".join(rendered_lines)


def build_ass(
    segments: list[Segment],
    start: float,
    end: float,
    out_path: Path,
    cfg: dict,
    hook: str = "",
) -> Path:
    """Build an ASS file for the window [start, end), with times starting at 0."""
    cp = cfg["captions"]
    strip = cp.get("strip_punctuation", True)

    words: list[Word] = []
    for seg in segments:
        for word in seg.words:
            if word.end <= start or word.start >= end or not word.text:
                continue
            text = word.text.upper() if cp["uppercase"] else word.text
            if strip:
                text = _TRAILING_PUNCT.sub("", text)
            if not text:
                continue
            words.append(
                Word(
                    start=max(word.start, start) - start,
                    end=min(word.end, end) - start,
                    text=text,
                )
            )

    out: list[str] = [_header(cfg)]

    # --- Hook overlay ------------------------------------------------------
    hk = cp.get("hook", {})
    if hook and hk.get("enabled", True):
        duration = min(float(hk.get("duration", 2.0)), end - start)
        body = _wrap(
            _escape(hook.upper() if hk.get("uppercase", True) else hook),
            hk.get("max_chars_per_line", 18),
            hk.get("max_lines", 3),
        )
        out.append(
            f"Dialogue: 1,{_fmt_time(0.0)},{_fmt_time(duration)},Hook,,0,0,0,,"
            f"{{\\fad(120,220)}}{body}"
        )

    # --- Captions ----------------------------------------------------------
    for lines in _blocks(words, cfg):
        flat = [w for line in lines for w in line]
        if not flat:
            continue
        block_end = max(flat[-1].end, flat[0].start + 0.3)

        for i, word in enumerate(flat):
            w_start = word.start
            w_end = flat[i + 1].start if i + 1 < len(flat) else block_end
            if w_end <= w_start:
                continue
            out.append(
                f"Dialogue: 0,{_fmt_time(w_start)},{_fmt_time(w_end)},Main,,0,0,0,,"
                f"{_render_block(lines, word, cfg)}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return out_path
