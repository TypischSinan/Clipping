"""Tests for the places where the pipeline can go quietly wrong.

The focus is on cut boundaries and caption layout - that is where a bug only
shows up when you watch the result, not as an exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from clipper.config import load_config
from clipper.models import (
    Candidate,
    ClipPlan,
    CropKeyframe,
    Segment,
    Shot,
    TimeMap,
    Word,
)
from clipper.stages.captions import _blocks, _render_block, _wrap, build_ass
from clipper.stages.render import _crop_x_expr, _ffmpeg_args, _select_expr, fingerprint
from clipper.stages.scenes import snap_end_within, snap_to_shots
from clipper.stages.select import _avoid_word_split, _finalize
from clipper.stages.silence import plan_cuts


@pytest.fixture
def cfg():
    return load_config()


def shots(*bounds: float) -> list[Shot]:
    return [
        Shot(index=i, start=a, end=b)
        for i, (a, b) in enumerate(zip(bounds, bounds[1:], strict=False))
    ]


# --- Cut boundaries ---------------------------------------------------------

def test_snap_start_prefers_boundary_before():
    """A clip start must not drift later - otherwise the first syllable is gone."""
    s = shots(0.0, 10.0, 20.0, 30.0)
    assert snap_to_shots(10.4, s, edge="start") == 10.0


def test_snap_end_prefers_boundary_after():
    s = shots(0.0, 10.0, 20.0, 30.0)
    assert snap_to_shots(19.6, s, edge="end") == 20.0


def test_snap_falls_back_to_nearest_when_out_of_drift():
    s = shots(0.0, 10.0, 20.0)
    # 15.0 is 5s from either boundary, i.e. outside max_drift.
    assert snap_to_shots(15.0, s, edge="start", max_drift=1.5) == 15.0


def test_snap_end_within_stays_on_boundary_and_in_range():
    """The core case of the earlier bug: length correction must not destroy
    the snapping."""
    s = shots(0.0, 5.0, 12.0, 18.0, 40.0)
    end = snap_end_within(0.0, 6.0, s, min_duration=15.0, max_duration=30.0)
    assert end == 18.0          # the only boundary in [15, 30]
    assert 15.0 <= end <= 30.0


def test_snap_end_within_without_usable_boundary():
    """With no cut boundary in the allowed window only a hard cut remains."""
    s = shots(0.0, 2.0, 100.0)
    end = snap_end_within(0.0, 20.0, s, min_duration=15.0, max_duration=30.0)
    assert 15.0 <= end <= 30.0


def test_finalize_never_leaves_the_length_window(cfg):
    cfg["select"].update(clips=5, min_duration=15.0, max_duration=30.0)
    s = shots(0.0, 5.0, 12.0, 18.0, 26.0, 60.0)
    # Proposal too short - this used to be blindly extended to min_duration.
    out = _finalize([Candidate(start=0.0, end=6.0, title="x", score=90)], s, cfg)
    assert len(out) == 1
    assert 15.0 <= out[0].duration <= 30.0


def test_finalize_output_is_never_overlapping(cfg):
    """The guarantee is non-overlapping output - not that a colliding clip is
    dropped. If it fits into the gap when shifted, it is kept; the higher-scored
    clip keeps its preferred position."""
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
    for first, second in zip(out, out[1:], strict=False):
        assert first.end <= second.start


def test_finalize_drops_a_clip_with_no_room_left(cfg):
    """Without a large enough gap the weaker clip is still dropped."""
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


# --- Word boundaries --------------------------------------------------------

def test_avoid_word_split_pulls_start_back():
    words = [Word(start=10.0, end=10.6, text="HELLO")]
    assert _avoid_word_split(10.3, words, edge="start") == 10.0


def test_avoid_word_split_pushes_end_forward():
    words = [Word(start=10.0, end=10.6, text="HELLO")]
    assert _avoid_word_split(10.3, words, edge="end") == 10.6


def test_avoid_word_split_respects_max_drift():
    """A very long word must not shift the cut arbitrarily far."""
    words = [Word(start=10.0, end=14.0, text="AAAA")]
    assert _avoid_word_split(12.0, words, edge="start", max_drift=0.4) == 12.0


def test_avoid_word_split_ignores_gaps():
    words = [Word(start=10.0, end=10.5, text="A"), Word(start=11.0, end=11.5, text="B")]
    assert _avoid_word_split(10.75, words, edge="start") == 10.75


# --- Captions ---------------------------------------------------------------

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
        Word(start=3.0, end=3.4, text="DREI"),   # 2.2s pause before it
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
    ws = words_from("DAS IST DAS")   # "DAS" appears twice
    lines = [ws]
    out = _render_block(lines, ws[2], cfg)
    # Only one highlight although the text is identical -> identity, not equality
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


# --- Crop expression --------------------------------------------------------

def test_crop_expr_single_keyframe():
    plan = ClipPlan(
        index=1,
        candidate=Candidate(start=0, end=10, title="x"),
        crops=[CropKeyframe(t=0.0, x=100, y=0, w=608, h=1080)],
    )
    assert _crop_x_expr(plan) == "100"


def test_crop_expr_nests_in_chronological_order():
    """Equal x values must not scramble the ordering."""
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


# --- Section download -------------------------------------------------------

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


# --- Words per block --------------------------------------------------------

def test_block_respects_max_words(cfg):
    cfg["captions"].update(max_words=4, max_chars_per_line=99, max_lines=2, max_gap=99.0)
    blocks = _blocks(words_from("A B C D E F G H I J"), cfg)
    for block in blocks:
        assert sum(len(line) for line in block) <= 4


def test_max_words_wins_over_char_capacity(cfg):
    """Short words must not stack up beyond the word limit."""
    cfg["captions"].update(max_words=3, max_chars_per_line=99, max_lines=2, max_gap=99.0)
    blocks = _blocks(words_from("A B C D E F"), cfg)
    assert len(blocks) == 2
    assert all(sum(len(line) for line in b) == 3 for b in blocks)


def test_max_words_zero_disables_the_limit(cfg):
    cfg["captions"].update(max_words=0, max_chars_per_line=99, max_lines=2, max_gap=99.0)
    blocks = _blocks(words_from("A B C D E F G H"), cfg)
    assert len(blocks) == 1


def test_language_prompt_names_the_target(cfg):
    from clipper.stages.select import system_prompt
    cfg["select"]["output_language"] = "en"
    assert "English" in system_prompt(cfg)
    cfg["select"]["output_language"] = "es"
    assert "Spanish" in system_prompt(cfg)


# --- Unlimited clip count ---------------------------------------------------

def test_finalize_unlimited_returns_all_surviving(cfg):
    cfg["select"].update(clips=0, min_score=0, min_duration=5.0, max_duration=15.0)
    s = shots(*[float(x) for x in range(0, 130, 10)])
    cands = [Candidate(start=float(a), end=float(a + 10), title=f"c{a}", score=50)
             for a in range(0, 120, 20)]
    out = _finalize(cands, s, cfg)
    assert len(out) == 6          # no cap at 8 or similar


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
    """Sorting by score is internal only - output is ordered by time."""
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
    """Two directly consecutive moments must not exclude each other just
    because the start snaps backwards onto the previous clip."""
    cfg["select"].update(clips=0, min_score=0, min_duration=10.0, max_duration=25.0)
    s = shots(0.0, 12.0, 24.0, 36.0, 48.0, 60.0)
    out = _finalize(
        [
            Candidate(start=0.0, end=24.0, title="erster", score=90),
            # Starts just after the first one ends - snapped backwards it
            # lands at 24.0 and runs into it.
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
    """A moment between two placed clips must not be dropped while the gap
    still allows min_duration."""
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
    for a, b in zip(out, out[1:], strict=False):
        assert a.end <= b.start


def test_snap_end_never_exceeds_the_material():
    """The fallback must not return an end past the last shot boundary -
    otherwise clips extend beyond the end of the video."""
    s = shots(0.0, 10.0, 20.0, 30.0)
    end = snap_end_within(28.0, 60.0, s, min_duration=20.0, max_duration=30.0)
    assert end <= 30.0


# --- Render arguments -------------------------------------------------------

def a_plan() -> ClipPlan:
    return ClipPlan(
        index=0,
        candidate=Candidate(start=10.0, end=30.0, title="clip"),
        crops=[CropKeyframe(t=0.0, x=100, y=0, w=608, h=1080)],
    )


def test_render_pins_the_audio_rate(cfg):
    """loudnorm runs its internal chain at 192 kHz. Without an explicit -ar the
    AAC encoder clamps to its own maximum of 96 kHz - a rate no short-form
    platform expects, so the upload gets resampled again on their side."""
    cfg["render"]["loudnorm"] = True
    args, _ = _ffmpeg_args(Path("src.mp4"), a_plan(), Path("out.mp4"), cfg, "libx264")

    assert "loudnorm" in " ".join(args)
    assert "-ar" in args
    assert args[args.index("-ar") + 1] == str(cfg["render"]["audio_rate"])


def test_render_audio_rate_does_not_depend_on_loudnorm(cfg):
    """Turning normalisation off must not silently change the container."""
    cfg["render"]["loudnorm"] = False
    args, _ = _ffmpeg_args(Path("src.mp4"), a_plan(), Path("out.mp4"), cfg, "libx264")

    assert "loudnorm" not in " ".join(args)
    assert args[args.index("-ar") + 1] == str(cfg["render"]["audio_rate"])


# --- Duration bounds --------------------------------------------------------

def test_finalize_never_ships_a_clip_outside_the_range(cfg):
    """min_duration/max_duration are bounds, not suggestions. The floor used to
    carry an 0.8 slack factor that let clips ship up to 20% short."""
    cfg["select"].update(clips=0, min_score=0, min_duration=60.0, max_duration=90.0)
    s = shots(0.0, 30.0, 45.0, 70.0, 95.0, 130.0, 200.0)
    out = _finalize(
        [
            Candidate(start=0.0, end=30.0, title="asked short", score=90),
            Candidate(start=95.0, end=200.0, title="asked long", score=80),
        ],
        s,
        cfg,
    )
    for c in out:
        assert 60.0 <= c.end - c.start <= 90.0, (c.title, c.end - c.start)


def test_finalize_drops_a_clip_that_cannot_reach_min_duration(cfg):
    """At the tail of a video no boundary sits far enough out. Dropping is
    correct - shipping a 30s clip when 60s was asked for is not."""
    cfg["select"].update(clips=0, min_score=0, min_duration=60.0, max_duration=90.0)
    s = shots(0.0, 10.0, 20.0, 30.0)  # material ends at 30s
    out = _finalize([Candidate(start=0.0, end=25.0, title="tail", score=99)], s, cfg)
    assert out == []


def test_word_drift_does_not_push_a_clip_past_max_duration(cfg):
    """_avoid_word_split only ever grows a clip; it must not breach the ceiling."""
    cfg["select"].update(clips=0, min_score=0, min_duration=60.0, max_duration=90.0)
    s = shots(0.0, 30.0, 60.0, 90.0, 150.0)
    segs = [Segment(start=0.0, end=150.0, text="x",
                    words=[Word(start=89.8, end=90.4, text="over")])]
    out = _finalize(
        [Candidate(start=0.0, end=90.0, title="edge", score=90)], s, cfg, segs
    )
    for c in out:
        assert c.end - c.start <= 90.0, c.end - c.start


# --- Input handling ---------------------------------------------------------

def test_local_video_id_is_stable_and_collision_resistant(tmp_path):
    from clipper.stages.ingest import _local_video_id
    a = tmp_path / "sub_a" / "clip.mp4"
    b = tmp_path / "sub_b" / "clip.mp4"
    a.parent.mkdir()
    b.parent.mkdir()
    a.touch()
    b.touch()
    # Same filename, different paths - the ids must not collide, but each has
    # to be stable across calls or the cache would be rebuilt every run.
    assert _local_video_id(a) == _local_video_id(a)
    assert _local_video_id(a) != _local_video_id(b)
    assert _local_video_id(a).startswith("clip-")


def test_local_video_id_survives_awkward_filenames(tmp_path):
    from clipper.stages.ingest import _local_video_id
    f = tmp_path / "Mr Beast: 7 Days!! (final).mp4"
    f.touch()
    vid = _local_video_id(f)
    assert "/" not in vid and ":" not in vid and " " not in vid


def test_aspect_presets_resolve_and_reject():
    from clipper.config import ASPECT_PRESETS, aspect_override
    assert aspect_override("9:16")["reframe"]["target_height"] == 1920
    assert aspect_override("1:1")["reframe"]["target_height"] == 1080
    assert set(ASPECT_PRESETS) == {"9:16", "4:5", "1:1"}
    with pytest.raises(ValueError):
        aspect_override("16:9")


# --- Sources without an audio track -----------------------------------------

def test_ffmpeg_args_drop_audio_when_the_source_has_none(cfg):
    """Asking for an audio codec on a silent source makes ffmpeg fail outright,
    so the arguments have to change - not just the result."""
    plan = ClipPlan(
        index=1,
        candidate=Candidate(start=0.0, end=10.0, title="x"),
        crops=[CropKeyframe(t=0.0, x=0, y=0, w=608, h=1080)],
    )
    args, _ = _ffmpeg_args(Path("in.mp4"), plan, Path("out.mp4"), cfg, "libx264",
                           with_audio=False)
    assert "-an" in args
    assert "-c:a" not in args and "-ar" not in args

    args, _ = _ffmpeg_args(Path("in.mp4"), plan, Path("out.mp4"), cfg, "libx264",
                           with_audio=True)
    assert "-an" not in args
    assert args[args.index("-ar") + 1] == str(cfg["render"]["audio_rate"])


def test_find_peaks_handles_an_empty_envelope(cfg):
    """A silent source yields no envelope; the peak search must not raise."""
    import numpy as np

    from clipper.stages.audio import find_peaks
    assert find_peaks(np.zeros(0), np.zeros(0), cfg) == []


def test_fmt_size_rounds_per_unit():
    from clipper.cli import _fmt_size
    assert _fmt_size(512) == "512 B"
    assert _fmt_size(4096) == "4.0 KB"
    assert _fmt_size(3_000_000_000).endswith("GB")


# --- Dead-air removal -------------------------------------------------------

def quiet(level: float = 0.1, span: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, span, 0.25)
    return times, np.full(len(times), level)


def spoken(*pairs: tuple[float, float, str]) -> list[Segment]:
    ws = [Word(start=a, end=b, text=t) for a, b, t in pairs]
    return [
        Segment(
            start=ws[0].start,
            end=ws[-1].end,
            text=" ".join(w.text for w in ws),
            words=ws,
        )
    ]


def a_gap_clip() -> list[Segment]:
    """Two words with a 1.5 s hole between them, inside a clip from 8 s to 16 s."""
    return spoken((10.0, 10.5, "one"), (12.0, 12.5, "two"))


def test_silence_cuts_a_quiet_gap(cfg):
    times, energy = quiet(0.1)
    tm = plan_cuts(a_gap_clip(), times, energy, 8.0, 16.0, cfg, 30.0)
    assert tm.cuts == 1
    # 1.5 s gap minus 0.12 s of padding on each side.
    assert tm.removed == pytest.approx(1.26, abs=0.05)


def test_silence_keeps_a_gap_that_is_not_actually_silent(cfg):
    """The check that carries the whole feature. A pause with music, a crowd or
    an explosion under it is not dead air - and on reaction content that is most
    of them. Cutting on the transcript alone would delete the payoff."""
    times, energy = quiet(0.9)
    tm = plan_cuts(a_gap_clip(), times, energy, 8.0, 16.0, cfg, 30.0)
    assert tm.is_identity


def test_silence_respects_the_off_switch(cfg):
    cfg["silence"]["enabled"] = False
    times, energy = quiet(0.1)
    assert plan_cuts(a_gap_clip(), times, energy, 8.0, 16.0, cfg, 30.0).is_identity


def test_silence_ignores_a_gap_that_is_too_short(cfg):
    times, energy = quiet(0.1)
    segs = spoken((10.0, 10.5, "one"), (11.0, 11.5, "two"))   # 0.5 s < min_gap
    assert plan_cuts(segs, times, energy, 8.0, 16.0, cfg, 30.0).is_identity


def test_silence_never_moves_the_clip_edges(cfg):
    """Start and end were snapped onto shot boundaries on purpose. This stage is
    not allowed to undo that, so it only ever cuts between words."""
    times, energy = quiet(0.1)
    tm = plan_cuts(a_gap_clip(), times, energy, 8.0, 16.0, cfg, 30.0)
    assert tm.keep[0][0] == 0.0
    assert tm.keep[-1][1] == pytest.approx(8.0, abs=1 / 30)


def test_silence_snaps_every_interval_onto_the_frame_grid(cfg):
    """Video keeps whole frames and audio keeps whole sample blocks. If an
    interval is not a whole number of frames the two round it differently, and
    the error accumulates over every further cut into audible A/V drift."""
    times, energy = quiet(0.1)
    segs = spoken((10.0, 10.5, "a"), (12.0, 12.5, "b"), (14.3, 14.9, "c"))
    tm = plan_cuts(segs, times, energy, 8.0, 17.0, cfg, 30.0)
    assert tm.cuts >= 2
    for a, b in tm.keep:
        assert (b - a) * 30 == pytest.approx(round((b - a) * 30), abs=1e-6)


def test_silence_with_no_transcript_changes_nothing(cfg):
    """Music-only or purely visual stretches have no word gaps to measure."""
    times, energy = quiet(0.1)
    assert plan_cuts([], times, energy, 8.0, 16.0, cfg, 30.0).is_identity


def test_silence_without_an_energy_curve_changes_nothing(cfg):
    """A source with no audio track leaves nothing to judge a gap by, and an
    unknown has to fall on the side of keeping material."""
    tm = plan_cuts(a_gap_clip(), np.zeros(0), np.zeros(0), 8.0, 16.0, cfg, 30.0)
    assert tm.is_identity


# --- Time mapping -----------------------------------------------------------

def test_timemap_maps_across_a_removed_gap():
    tm = TimeMap(keep=[(0.0, 2.0), (4.0, 6.0)], source_duration=6.0)
    assert tm.duration == 4.0
    assert tm.removed == 2.0
    assert tm.to_output(1.0) == 1.0        # before the cut, unchanged
    assert tm.to_output(4.5) == 2.5        # after it, pulled forward
    assert tm.to_output(3.0) == 2.0        # inside it, collapsed onto the seam


def test_timemap_identity_reports_itself_as_one():
    assert TimeMap(keep=[(0.0, 9.0)], source_duration=9.0).is_identity


def test_select_expr_never_compares_against_a_frame_position():
    """The bounds sit on frame positions and `between` is inclusive at both
    ends, so comparing against them leaves it to floating point whether the edge
    frame survives - and video and audio decided differently often enough to
    accumulate 100 ms of drift over 13 cuts. Half-frame offsets are positions no
    frame ever occupies."""
    tm = TimeMap(keep=[(0.0, 2.0), (4.0, 6.0)], source_duration=6.0)
    expr = _select_expr(tm, 30.0)
    half = 0.5 / 30.0
    assert expr == (
        f"between(t,{-half:.4f},{2.0 - half:.4f})"
        f"+between(t,{4.0 - half:.4f},{6.0 - half:.4f})"
    )


# --- Render arguments -------------------------------------------------------

def a_plan_with(timing: TimeMap | None) -> ClipPlan:
    return ClipPlan(
        index=1,
        candidate=Candidate(start=10.0, end=30.0, title="clip"),
        crops=[CropKeyframe(t=0.0, x=100, y=0, w=608, h=1080)],
        timing=timing,
    )


def test_render_leaves_the_chain_alone_without_cuts(cfg):
    plan = a_plan_with(TimeMap(keep=[(0.0, 20.0)], source_duration=20.0))
    joined = " ".join(_ffmpeg_args(Path("s.mp4"), plan, Path("o.mp4"), cfg, "libx264")[0])
    assert "select=" not in joined


def test_render_cuts_video_and_audio_on_the_same_grid(cfg):
    """Both sides have to be pinned to the same frame grid, or the cuts land on
    different instants and the clip drifts apart."""
    tm = TimeMap(keep=[(0.0, 5.0), (7.0, 20.0)], source_duration=20.0)
    args, _ = _ffmpeg_args(Path("s.mp4"), a_plan_with(tm), Path("o.mp4"), cfg, "libx264")
    video = args[args.index("-vf") + 1]
    audio = args[args.index("-af") + 1]

    expr = _select_expr(tm, float(cfg["render"]["fps"]))
    assert f"select='{expr}'" in video
    assert f"aselect='{expr}'" in audio
    # fps before select: the grid has to exist before anything is selected off it.
    assert video.index("fps=") < video.index("select=")
    samples = round(cfg["render"]["audio_rate"] / cfg["render"]["fps"])
    assert f"asetnsamples=n={samples}" in audio
    assert "loudnorm" in audio


# --- Render fingerprint -----------------------------------------------------

def test_fingerprint_is_stable_for_the_same_config(cfg):
    assert fingerprint(cfg) == fingerprint(load_config())


def test_fingerprint_notices_a_different_output_format(cfg):
    """The filename carries index, score and title - nothing about the format.
    Without this hash `build --aspect 4:5` would report success and hand back
    the 9:16 files from the previous run."""
    other = load_config()
    other["reframe"].update(target_width=1080, target_height=1350)
    assert fingerprint(cfg) != fingerprint(other)


@pytest.mark.parametrize(
    ("section", "change"),
    [
        ("captions", {"enabled": False}),
        ("silence", {"enabled": False}),
        ("render", {"crf": 28}),
    ],
)
def test_fingerprint_notices_render_relevant_changes(cfg, section, change):
    other = load_config()
    other[section].update(change)
    assert fingerprint(cfg) != fingerprint(other)


def test_fingerprint_ignores_what_cannot_change_a_rendered_file(cfg):
    """Selection and transcription happen upstream of the encoder. They decide
    which clips exist, not what an existing file looks like."""
    other = load_config()
    other["select"]["min_score"] = 99
    other["transcribe"]["model"] = "tiny"
    assert fingerprint(cfg) == fingerprint(other)


# --- Captions on a compressed timeline --------------------------------------

def test_captions_follow_the_cuts(cfg, tmp_path):
    """Caption times are mapped before anything else runs, so a word after a cut
    appears when it is actually heard - not where it sat in the source."""
    cfg["captions"]["hook"]["enabled"] = False
    segs = spoken((104.5, 104.9, "later"))
    tm = TimeMap(keep=[(0.0, 2.0), (4.0, 6.0)], source_duration=6.0)

    with_cut = build_ass(segs, 100.0, 106.0, tmp_path / "a.ass", cfg, timing=tm)
    without = build_ass(segs, 100.0, 106.0, tmp_path / "b.ass", cfg)

    assert "0:00:02.50" in with_cut.read_text(encoding="utf-8")
    assert "0:00:04.50" in without.read_text(encoding="utf-8")
