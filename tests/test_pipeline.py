"""Tests fuer die Stellen, an denen die Pipeline still falsch laufen kann.

Schwerpunkt liegt auf Schnittgrenzen und Untertitel-Layout - dort faellt ein
Fehler im Ergebnis erst beim Ansehen auf, nicht als Exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clipper.config import load_config
from clipper.models import Candidate, ClipPlan, CropKeyframe, Segment, Shot, Word
from clipper.stages.captions import _blocks, _render_block, _wrap
from clipper.stages.render import _crop_x_expr
from clipper.stages.scenes import snap_end_within, snap_to_shots
from clipper.stages.select import _avoid_word_split, _finalize


@pytest.fixture
def cfg():
    return load_config()


def shots(*bounds: float) -> list[Shot]:
    return [
        Shot(index=i, start=a, end=b)
        for i, (a, b) in enumerate(zip(bounds, bounds[1:]))
    ]


# --- Schnittgrenzen ---------------------------------------------------------

def test_snap_start_prefers_boundary_before():
    """Ein Clipanfang darf nicht nach hinten rutschen - sonst fehlt die erste Silbe."""
    s = shots(0.0, 10.0, 20.0, 30.0)
    assert snap_to_shots(10.4, s, edge="start") == 10.0


def test_snap_end_prefers_boundary_after():
    s = shots(0.0, 10.0, 20.0, 30.0)
    assert snap_to_shots(19.6, s, edge="end") == 20.0


def test_snap_falls_back_to_nearest_when_out_of_drift():
    s = shots(0.0, 10.0, 20.0)
    # 15.0 ist von beiden Grenzen 5s entfernt, also ausserhalb max_drift.
    assert snap_to_shots(15.0, s, edge="start", max_drift=1.5) == 15.0


def test_snap_end_within_stays_on_boundary_and_in_range():
    """Der Kernfall des frueheren Bugs: Laengenkorrektur darf das Snapping
    nicht zerstoeren."""
    s = shots(0.0, 5.0, 12.0, 18.0, 40.0)
    end = snap_end_within(0.0, 6.0, s, min_duration=15.0, max_duration=30.0)
    assert end == 18.0          # einzige Grenze in [15, 30]
    assert 15.0 <= end <= 30.0


def test_snap_end_within_without_usable_boundary():
    """Ohne Schnittgrenze im erlaubten Fenster bleibt nur der harte Schnitt."""
    s = shots(0.0, 2.0, 100.0)
    end = snap_end_within(0.0, 20.0, s, min_duration=15.0, max_duration=30.0)
    assert 15.0 <= end <= 30.0


def test_finalize_never_leaves_the_length_window(cfg):
    cfg["select"].update(clips=5, min_duration=15.0, max_duration=30.0)
    s = shots(0.0, 5.0, 12.0, 18.0, 26.0, 60.0)
    # Zu kurzer Vorschlag - frueher wurde blind auf min_duration verlaengert.
    out = _finalize([Candidate(start=0.0, end=6.0, title="x", score=90)], s, cfg)
    assert len(out) == 1
    assert 15.0 <= out[0].duration <= 30.0


def test_finalize_output_is_never_overlapping(cfg):
    """Die Zusicherung ist ueberlappungsfreie Ausgabe - nicht, dass ein
    kollidierender Clip wegfaellt. Passt er verschoben in die Luecke, wird er
    behalten; der hoeher bewertete behaelt seine Wunschposition."""
    cfg["select"].update(clips=5, min_duration=5.0, max_duration=30.0)
    s = shots(0.0, 10.0, 20.0, 30.0)
    out = _finalize(
        [
            Candidate(start=0.0, end=20.0, title="a", score=90),
            Candidate(start=10.0, end=30.0, title="b", score=80),
        ],
        s,
        cfg,
    )
    assert out[0].title == "a" and out[0].start == 0.0
    for first, second in zip(out, out[1:]):
        assert first.end <= second.start


def test_finalize_drops_a_clip_with_no_room_left(cfg):
    """Ohne ausreichende Luecke faellt der schwaechere Clip weiterhin weg."""
    cfg["select"].update(clips=5, min_score=0, min_duration=20.0, max_duration=30.0)
    s = shots(0.0, 10.0, 20.0, 30.0)
    out = _finalize(
        [
            Candidate(start=0.0, end=30.0, title="a", score=90),
            Candidate(start=10.0, end=30.0, title="b", score=80),
        ],
        s,
        cfg,
    )
    assert [c.title for c in out] == ["a"]


# --- Wortgrenzen ------------------------------------------------------------

def test_avoid_word_split_pulls_start_back():
    words = [Word(start=10.0, end=10.6, text="HELLO")]
    assert _avoid_word_split(10.3, words, edge="start") == 10.0


def test_avoid_word_split_pushes_end_forward():
    words = [Word(start=10.0, end=10.6, text="HELLO")]
    assert _avoid_word_split(10.3, words, edge="end") == 10.6


def test_avoid_word_split_respects_max_drift():
    """Ein sehr langes Wort darf den Schnitt nicht beliebig weit verschieben."""
    words = [Word(start=10.0, end=14.0, text="AAAA")]
    assert _avoid_word_split(12.0, words, edge="start", max_drift=0.4) == 12.0


def test_avoid_word_split_ignores_gaps():
    words = [Word(start=10.0, end=10.5, text="A"), Word(start=11.0, end=11.5, text="B")]
    assert _avoid_word_split(10.75, words, edge="start") == 10.75


# --- Untertitel -------------------------------------------------------------

def words_from(text: str, start: float = 0.0, step: float = 0.4) -> list[Word]:
    return [
        Word(start=start + i * step, end=start + i * step + step * 0.8, text=w)
        for i, w in enumerate(text.split())
    ]


def test_block_breaks_on_speech_pause(cfg):
    cfg["captions"].update(max_chars_per_line=40, max_lines=2, max_gap=0.6)
    ws = [
        Word(start=0.0, end=0.4, text="EINS"),
        Word(start=0.4, end=0.8, text="ZWEI"),
        Word(start=3.0, end=3.4, text="DREI"),   # 2.2s Pause davor
    ]
    blocks = _blocks(ws, cfg)
    assert len(blocks) == 2
    assert [w.text for w in blocks[0][0]] == ["EINS", "ZWEI"]


def test_block_respects_line_and_block_capacity(cfg):
    cfg["captions"].update(max_chars_per_line=10, max_lines=2, max_gap=99.0)
    blocks = _blocks(words_from("AAA BBB CCC DDD EEE FFF"), cfg)
    for block in blocks:
        assert len(block) <= 2
        for line in block:
            assert len(" ".join(w.text for w in line)) <= 10


def test_render_block_highlights_only_the_active_word(cfg):
    cfg["captions"].update(pop=False)
    ws = words_from("DAS IST DAS")   # "DAS" kommt doppelt vor
    lines = [ws]
    out = _render_block(lines, ws[2], cfg)
    # Nur eine Hervorhebung, obwohl der Text identisch ist -> Identitaet, nicht Gleichheit
    assert out.count(cfg["captions"]["highlight_color"]) == 1


def test_render_block_pop_is_relative_to_event_start(cfg):
    cfg["captions"].update(pop=True)
    ws = words_from("EINS ZWEI")
    out = _render_block([ws], ws[0], cfg)
    assert "\\t(0,110," in out


def test_wrap_does_not_break_words():
    out = _wrap("EIN ZWEI DREI VIER", 9)
    for line in out.split("\\N"):
        assert len(line) <= 9


# --- Crop-Ausdruck ----------------------------------------------------------

def test_crop_expr_single_keyframe():
    plan = ClipPlan(
        index=1,
        candidate=Candidate(start=0, end=10, title="x"),
        crops=[CropKeyframe(t=0.0, x=100, y=0, w=608, h=1080)],
    )
    assert _crop_x_expr(plan) == "100"


def test_crop_expr_nests_in_chronological_order():
    """Gleiche x-Werte duerfen die Reihenfolge nicht durcheinanderbringen."""
    plan = ClipPlan(
        index=1,
        candidate=Candidate(start=0, end=10, title="x"),
        crops=[
            CropKeyframe(t=0.0, x=100, y=0, w=608, h=1080),
            CropKeyframe(t=2.0, x=200, y=0, w=608, h=1080),
            CropKeyframe(t=4.0, x=100, y=0, w=608, h=1080),
        ],
    )
    expr = _crop_x_expr(plan)
    assert expr == "if(lt(t,2.000),100,if(lt(t,4.000),200,100))"


# --- Abschnitts-Download ----------------------------------------------------

def test_parse_section_plain_seconds():
    from clipper.stages.ingest import parse_section
    assert parse_section("*120-360") == (120.0, 360.0)


def test_parse_section_timecodes():
    from clipper.stages.ingest import parse_section
    assert parse_section("2:00-6:00") == (120.0, 360.0)
    assert parse_section("0:01:30-0:02:00") == (90.0, 120.0)


def test_parse_section_requires_range():
    from clipper.stages.ingest import parse_section
    with pytest.raises(ValueError):
        parse_section("120")


# --- Wortzahl pro Block -----------------------------------------------------

def test_block_respects_max_words(cfg):
    cfg["captions"].update(max_words=4, max_chars_per_line=99, max_lines=2, max_gap=99.0)
    blocks = _blocks(words_from("A B C D E F G H I J"), cfg)
    for block in blocks:
        assert sum(len(line) for line in block) <= 4


def test_max_words_wins_over_char_capacity(cfg):
    """Kurze Woerter duerfen sich nicht ueber die Wortgrenze hinaus stapeln."""
    cfg["captions"].update(max_words=3, max_chars_per_line=99, max_lines=2, max_gap=99.0)
    blocks = _blocks(words_from("A B C D E F"), cfg)
    assert len(blocks) == 2
    assert all(sum(len(l) for l in b) == 3 for b in blocks)


def test_max_words_zero_disables_the_limit(cfg):
    cfg["captions"].update(max_words=0, max_chars_per_line=99, max_lines=2, max_gap=99.0)
    blocks = _blocks(words_from("A B C D E F G H"), cfg)
    assert len(blocks) == 1


def test_language_prompt_names_the_target(cfg):
    from clipper.stages.select import system_prompt
    cfg["select"]["output_language"] = "en"
    assert "Englisch" in system_prompt(cfg)
    cfg["select"]["output_language"] = "es"
    assert "Spanisch" in system_prompt(cfg)


# --- Unbegrenzte Clipzahl ---------------------------------------------------

def test_finalize_unlimited_returns_all_surviving(cfg):
    cfg["select"].update(clips=0, min_score=0, min_duration=5.0, max_duration=15.0)
    s = shots(*[float(x) for x in range(0, 130, 10)])
    cands = [Candidate(start=float(a), end=float(a + 10), title=f"c{a}", score=50)
             for a in range(0, 120, 20)]
    out = _finalize(cands, s, cfg)
    assert len(out) == 6          # keine Deckelung auf 8 oder aehnliches


def test_finalize_limit_still_caps_when_set(cfg):
    cfg["select"].update(clips=2, min_score=0, min_duration=5.0, max_duration=15.0)
    s = shots(*[float(x) for x in range(0, 130, 10)])
    cands = [Candidate(start=float(a), end=float(a + 10), title=f"c{a}", score=float(a))
             for a in range(0, 120, 20)]
    out = _finalize(cands, s, cfg)
    assert len(out) == 2


def test_finalize_drops_below_min_score(cfg):
    cfg["select"].update(clips=0, min_score=50, min_duration=5.0, max_duration=15.0)
    s = shots(0.0, 10.0, 20.0, 30.0, 40.0)
    out = _finalize(
        [
            Candidate(start=0.0, end=10.0, title="gut", score=80),
            Candidate(start=20.0, end=30.0, title="schwach", score=30),
        ],
        s,
        cfg,
    )
    assert [c.title for c in out] == ["gut"]


def test_finalize_returns_chronological_order(cfg):
    """Nach Score sortiert wird nur intern - ausgegeben wird nach Zeit."""
    cfg["select"].update(clips=0, min_score=0, min_duration=5.0, max_duration=15.0)
    s = shots(0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
    out = _finalize(
        [
            Candidate(start=30.0, end=40.0, title="spaet", score=90),
            Candidate(start=0.0, end=10.0, title="frueh", score=60),
        ],
        s,
        cfg,
    )
    assert [c.title for c in out] == ["frueh", "spaet"]


def test_finalize_retries_forward_when_backward_snap_collides(cfg):
    """Zwei direkt aufeinanderfolgende Momente duerfen sich nicht gegenseitig
    ausschliessen, nur weil der Start rueckwaerts auf den Vorgaenger snappt."""
    cfg["select"].update(clips=0, min_score=0, min_duration=10.0, max_duration=25.0)
    s = shots(0.0, 12.0, 24.0, 36.0, 48.0, 60.0)
    out = _finalize(
        [
            Candidate(start=0.0, end=24.0, title="erster", score=90),
            # Startet knapp nach dem Ende des ersten - rueckwaerts gesnappt
            # landet er bei 24.0 und ragt hinein.
            Candidate(start=24.5, end=48.0, title="zweiter", score=80),
        ],
        s,
        cfg,
    )
    assert [c.title for c in out] == ["erster", "zweiter"]
    assert out[0].end <= out[1].start


def test_snap_end_within_respects_hard_max():
    s = shots(0.0, 10.0, 20.0, 30.0, 40.0)
    end = snap_end_within(0.0, 35.0, s, min_duration=10.0, max_duration=40.0, hard_max=22.0)
    assert end <= 22.0


def test_finalize_fits_a_clip_into_the_remaining_gap(cfg):
    """Ein Moment zwischen zwei vergebenen Clips darf nicht wegfallen,
    solange die Luecke noch min_duration hergibt."""
    cfg["select"].update(clips=0, min_score=0, min_duration=10.0, max_duration=40.0)
    s = shots(0.0, 15.0, 30.0, 45.0, 60.0, 75.0)
    out = _finalize(
        [
            Candidate(start=0.0, end=15.0, title="links", score=95),
            Candidate(start=45.0, end=75.0, title="rechts", score=90),
            Candidate(start=16.0, end=44.0, title="mitte", score=60),
        ],
        s,
        cfg,
    )
    assert "mitte" in [c.title for c in out]
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start


def test_snap_end_never_exceeds_the_material():
    """Der Fallback darf kein Ende hinter der letzten Shot-Grenze liefern -
    sonst entstehen Clips, die ueber das Videoende hinausragen."""
    s = shots(0.0, 10.0, 20.0, 30.0)
    end = snap_end_within(28.0, 60.0, s, min_duration=20.0, max_duration=30.0)
    assert end <= 30.0
