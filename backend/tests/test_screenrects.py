"""Rect-space line screen: the merge, the stripe lattice, and what they cost."""
from __future__ import annotations

import numpy as np
import pytest

from app.patterns.bitmap.screenrects import (
    MAX_RECTS,
    rect_area_um2,
    screen_bands,
    split_by_colour,
    stripe_plan,
    stripe_rects,
)

P, STEPS = 44.0, 22
EXT = 4400.0          # exactly 100 lines


def _flat(n: int = 64, level: float = 0.5) -> np.ndarray:
    return np.full((n, n), level, dtype=np.float32)


# --- the screen -------------------------------------------------------------


def test_band_height_is_the_quantised_tone():
    """Band height is floor(tone*steps)/steps * period — the screen's real tone
    ladder, not a rounding artefact."""
    r, _, _ = screen_bands(_flat(level=0.5), extent_um=EXT, line_period_um=P,
                           tone_steps=STEPS)
    h = r[:, 3] - r[:, 2]
    assert np.allclose(h, (np.floor(0.5 * STEPS) / STEPS) * P)


def test_a_flat_tone_is_one_rectangle_per_line():
    """The whole point of the merge: constant tone across a line costs ONE
    rectangle, not one per raster row and not one per column."""
    r, _, rep = screen_bands(_flat(level=0.6), extent_um=EXT, line_period_um=P,
                             tone_steps=STEPS)
    assert rep["n_lines"] == 100
    assert len(r) == 100
    assert rep["rects_per_line"] == pytest.approx(1.0)


def test_zero_tone_emits_nothing():
    r, _, _ = screen_bands(np.zeros((32, 32), np.float32), extent_um=EXT,
                           line_period_um=P, tone_steps=STEPS)
    assert len(r) == 0


def test_bands_stay_inside_their_line_and_inside_the_extent():
    n = 64
    ramp = np.tile(np.linspace(0.0, 1.0, n, dtype=np.float32), (n, 1))
    r, _, _ = screen_bands(ramp, extent_um=EXT, line_period_um=P, tone_steps=STEPS)
    half = EXT / 2.0
    assert r[:, 0].min() >= -half - 1e-9 and r[:, 1].max() <= half + 1e-9
    assert r[:, 2].min() >= -half - 1e-9 and r[:, 3].max() <= half + 1e-9
    assert ((r[:, 3] - r[:, 2]) <= P + 1e-9).all()


def test_origin_translates_the_whole_cell():
    a, _, _ = screen_bands(_flat(), extent_um=EXT, line_period_um=P, tone_steps=STEPS)
    b, _, _ = screen_bands(_flat(), extent_um=EXT, line_period_um=P, tone_steps=STEPS,
                           origin=(1000.0, -250.0))
    assert np.allclose(b[:, :2], a[:, :2] + 1000.0)
    assert np.allclose(b[:, 2:], a[:, 2:] - 250.0)


def test_a_run_breaks_where_the_PERIOD_changes_not_only_the_tone():
    """Joint run-merge. A rectangle must carry exactly one period or the writer
    cannot array it, so a colour boundary has to split a run of equal tone."""
    n = 64
    tone = _flat(n, 0.6)
    pid = np.zeros((n, n), dtype=np.int32)
    pid[:, n // 2:] = 3
    plain, _, _ = screen_bands(tone, extent_um=EXT, line_period_um=P, tone_steps=STEPS)
    split, ids, _ = screen_bands(tone, extent_um=EXT, line_period_um=P,
                                 tone_steps=STEPS, period_id=pid)
    assert len(split) == 2 * len(plain)
    assert set(np.unique(ids)) == {0, 3}


def test_shape_and_argument_guards():
    with pytest.raises(ValueError):
        screen_bands(np.zeros((4, 4, 3), np.float32), extent_um=EXT,
                     line_period_um=P, tone_steps=STEPS)
    with pytest.raises(ValueError):
        screen_bands(_flat(), extent_um=0.0, line_period_um=P, tone_steps=STEPS)
    with pytest.raises(ValueError):
        screen_bands(_flat(), extent_um=EXT, line_period_um=P, tone_steps=STEPS,
                     period_id=np.zeros((9, 9), np.int32))


# --- the sub-grating --------------------------------------------------------


def test_stripes_are_whole_and_land_on_the_global_lattice():
    """Every stripe sits at phase + k*period exactly. That is what lets a band
    become one array reference instead of N polygons."""
    band = np.array([[0.0, 100.0, 0.0, 10.0]])
    d = 5.0
    s = stripe_rects(band, d, 0.5)
    assert np.allclose((s[:, 0] / d) % 1.0, 0.0)
    assert np.allclose(s[:, 1] - s[:, 0], 2.5)


def test_stripes_never_escape_their_band():
    band = np.array([[3.3, 97.7, 0.0, 10.0]])
    s = stripe_rects(band, 5.0, 0.5)
    assert s[:, 0].min() >= 3.3 - 1e-9
    assert s[:, 1].max() <= 97.7 + 1e-9


def test_snapping_to_the_lattice_is_UNBIASED():
    """Whole stripes are selected by whether their CENTRE is in the band. That
    moves an edge by up to half a period, so it has to be unbiased or the image
    would gain or lose tone everywhere. Over many random bands the realized gold
    must match the duty to well under one stripe."""
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-500, 0, 4000)
    w = rng.uniform(20.0, 400.0, 4000)
    bands = np.stack([x0, x0 + w, np.zeros(4000), np.full(4000, 10.0)], axis=1)
    got = rect_area_um2(stripe_rects(bands, 5.0, 0.5))
    want = rect_area_um2(bands) * 0.5
    assert got == pytest.approx(want, rel=2e-3)


def test_a_band_narrower_than_one_period_may_hold_no_stripe():
    """Physically right: a band with no room for a stripe centre gets none, and
    that is the same shadow region colourzone.min_tone_for_colour describes."""
    band = np.array([[0.1, 1.2, 0.0, 10.0]])
    assert len(stripe_rects(band, 5.0, 0.5)) == 0


def test_stripe_plan_and_stripe_rects_agree():
    """The writer uses the plan and previews use the rects; if they disagreed
    the fab mask and the preview would show different geometry."""
    rng = np.random.default_rng(3)
    x0 = rng.uniform(-200, 0, 200)
    bands = np.stack([x0, x0 + rng.uniform(10, 300, 200),
                      np.zeros(200), np.full(200, 8.0)], axis=1)
    assert stripe_plan(bands, 5.0, 0.5)["total"] == len(stripe_rects(bands, 5.0, 0.5))


def test_per_band_periods_are_honoured():
    bands = np.array([[0.0, 100.0, 0.0, 10.0], [0.0, 100.0, 20.0, 30.0]])
    per = np.array([4.0, 8.0])
    plan = stripe_plan(bands, per, 0.5)
    assert plan["n"][0] > plan["n"][1]
    assert plan["line_um"].tolist() == [2.0, 4.0]


def test_the_rectangle_budget_refuses_rather_than_swapping():
    """The machine bugchecks under memory load, so an over-budget request must
    fail loudly at the plan stage instead of being attempted."""
    band = np.array([[0.0, 1e6, 0.0, 10.0]])
    with pytest.raises(ValueError, match="cap"):
        stripe_rects(band, 2.0, 0.5, max_rects=1000)
    assert stripe_plan(band, 2.0, 0.5)["total"] <= MAX_RECTS * 500


def test_bad_period_or_duty_is_refused():
    band = np.array([[0.0, 10.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        stripe_plan(band, 0.0, 0.5)
    with pytest.raises(ValueError):
        stripe_plan(band, 5.0, 1.0)
    with pytest.raises(ValueError):
        stripe_plan(np.zeros((3, 3)), 5.0, 0.5)


# --- partition --------------------------------------------------------------


def test_split_by_colour_partitions_without_loss():
    r = np.array([[0.0, 10.0, 0.0, 1.0]] * 6)
    pid = np.array([0, 1, 2, 0, 1, 5])
    plain, col, per = split_by_colour(r, pid, {1: 4.0, 2: 6.0})
    assert len(plain) == 3          # ids 0, 0 and the unmapped 5
    assert len(col) == 3
    assert sorted(per.tolist()) == [4.0, 4.0, 6.0]
    assert len(plain) + len(col) == len(r)


def test_plain_mode_is_the_same_path_with_an_empty_field():
    """The control cell must not be a different code path, or a difference on
    the plate could be the code rather than the colour."""
    tone = _flat(level=0.6)
    r, pid, _ = screen_bands(tone, extent_um=EXT, line_period_um=P, tone_steps=STEPS)
    plain, col, _ = split_by_colour(r, pid, {})
    assert len(col) == 0
    assert np.array_equal(plain, r)
