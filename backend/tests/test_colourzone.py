"""Colour-coded zones over a halftone — the knobs that work, and their costs."""
from __future__ import annotations

import numpy as np
import pytest

from app.patterns.bitmap.colourzone import (
    MIN_FEATURE_UM,
    ColourZone,
    hue_ladder,
    min_base_period_um,
    source_width_for_saturation_deg,
    min_tone_for_colour,
    screen_with_colour,
    tone_cap,
    zone_period_um,
)

SCREEN, STEPS, CELL = 44.0, 22, 0.5


def _tone(n: int = 200, level: float = 0.6) -> np.ndarray:
    return np.full((n, n), level, dtype=np.float32)


def _half(n: int = 200) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[:, n // 2 :] = True
    return m


# --- what controls hue ------------------------------------------------------


def test_period_scale_is_the_hue_offset():
    assert zone_period_um(4.4, ColourZone(period_scale=1.0)) == pytest.approx(4.4)
    assert zone_period_um(4.4, ColourZone(period_scale=1.15)) == pytest.approx(5.06)


def test_a_period_ratio_is_view_independent():
    """The reason zones are specified as a scale rather than a colour.

    lambda = d * k, where k is the whole view/light geometry. Two zones at a
    fixed period ratio keep that ratio at EVERY k, so they sweep together and
    the separation between them holds as the piece tilts.
    """
    d1, d2 = 4.4, 4.4 * 1.15
    for k in (0.09, 0.11, 0.14, 0.18):
        assert (d2 * k) / (d1 * k) == pytest.approx(1.15, rel=1e-9)


def test_hue_ladder_spans_the_visible_and_is_ordered():
    lad = hue_ladder(5)
    assert len(lad) == 5
    assert lad == sorted(lad)
    assert lad[-1] / lad[0] == pytest.approx(1.45, rel=1e-3)
    assert hue_ladder(1) == [1.0]
    with pytest.raises(ValueError):
        hue_ladder(0)


def test_a_ladder_centred_on_the_accent_period_does_not_fit():
    """The obvious choice fails: a 3-step ladder on the 4.4 um accent period
    puts its blue end at 1.83 um lines, under the floor. It must be centred
    higher, and min_base_period_um says how much."""
    lo = min(hue_ladder(3))
    assert 4.4 * lo * 0.5 < MIN_FEATURE_UM              # 1.83 um lines
    base = min_base_period_um(3)
    assert base == pytest.approx(4.82, abs=0.02)
    assert base * lo * 0.5 >= MIN_FEATURE_UM - 1e-9     # and 5.0 clears it
    assert 5.0 * lo * 0.5 >= MIN_FEATURE_UM


def test_it_is_the_SPREAD_that_drives_the_base_not_the_step_count():
    """A ladder spans a fixed red/blue ratio however many rungs it has, so more
    steps subdivides it rather than widening it — the base is unchanged. Asking
    for a wider spectrum is what forces a coarser base."""
    assert min_base_period_um(5) == pytest.approx(min_base_period_um(3))
    assert min_base_period_um(3, spread=1.9) > min_base_period_um(3, spread=1.45)


# --- the costs --------------------------------------------------------------


def test_holding_tone_caps_the_renderable_brightness():
    """A gratinged band reflects `duty` of a solid one, so holding tone means
    widening by 1/duty — and that runs out of period at tone == duty."""
    assert tone_cap(ColourZone(duty=0.5)) == pytest.approx(0.5)
    assert tone_cap(ColourZone(duty=0.35)) == pytest.approx(0.35)
    assert tone_cap(ColourZone(duty=0.5, hold_tone=False)) == pytest.approx(1.0)


def test_dark_tones_cannot_carry_colour():
    """A band narrower than a couple of grating periods is a pair of slits, not
    a grating. Shadows staying plain metal is a consequence, not a bug."""
    t = min_tone_for_colour(4.4, SCREEN, duty=0.5, hold_tone=True)
    assert 0.0 < t < 1.0
    # a coarser colour grating needs a wider band, so it needs a brighter tone
    assert min_tone_for_colour(8.8, SCREEN) > min_tone_for_colour(4.4, SCREEN)


def test_the_report_flags_a_zone_that_breaks_the_litho_floor():
    tone = _tone()
    fine = ColourZone(period_scale=0.7, label="too fine")   # 3.08 um -> 1.54 um lines
    ok = ColourZone(period_scale=1.0, label="ok")           # 4.4 um -> 2.2 um lines
    _, rep = screen_with_colour(
        tone, [(_half(), fine)], cell_um=CELL, screen_period_um=SCREEN,
        tone_steps=STEPS, base_period_um=4.4,
    )
    assert rep["zones"][0]["clears_litho_floor"] is False
    assert rep["zones"][0]["line_um"] < MIN_FEATURE_UM
    assert rep["all_zones_printable"] is False

    _, rep2 = screen_with_colour(
        tone, [(_half(), ok)], cell_um=CELL, screen_period_um=SCREEN,
        tone_steps=STEPS, base_period_um=4.4,
    )
    assert rep2["zones"][0]["clears_litho_floor"] is True
    assert rep2["all_zones_printable"] is True


def test_the_report_measures_how_much_of_a_zone_is_too_dark_for_colour():
    n = 200
    ramp = np.tile(np.linspace(0.02, 1.0, n, dtype=np.float32), (n, 1))
    _, rep = screen_with_colour(
        ramp, [(np.ones((n, n), bool), ColourZone())],
        cell_um=CELL, screen_period_um=SCREEN, tone_steps=STEPS,
    )
    frac = rep["zones"][0]["frac_band_too_narrow"]
    assert 0.0 < frac < 1.0, "a full tonal ramp must be partly too dark for colour"


def test_a_coarse_grating_needs_a_narrow_source_to_stay_saturated():
    """Dispersion is d*cos(theta), so a COARSE grating packs the visible band
    into few degrees and any wide source mixes it toward white. At the 4-5 um
    the litho floor allows, that means a lamp rather than a window."""
    assert source_width_for_saturation_deg(4.4) == pytest.approx(2.5, abs=0.3)
    # finer holds its colour under softer light
    assert source_width_for_saturation_deg(2.0) > source_width_for_saturation_deg(4.4)
    with pytest.raises(ValueError):
        source_width_for_saturation_deg(0.0)


# --- the mask it builds -----------------------------------------------------


def test_a_zone_actually_grates_its_bands():
    tone = _tone(level=0.8)
    plain, _ = screen_with_colour(tone, [], cell_um=CELL, screen_period_um=SCREEN,
                                  tone_steps=STEPS)
    col, _ = screen_with_colour(tone, [(_half(), ColourZone(hold_tone=False))],
                                cell_um=CELL, screen_period_um=SCREEN, tone_steps=STEPS)
    left, right = slice(None, 100), slice(100, None)
    # untouched outside the zone
    assert np.array_equal(plain[:, left], col[:, left])
    # and chopped up inside it
    assert col[:, right].mean() < plain[:, right].mean() * 0.75


def test_holding_tone_restores_most_of_the_lost_coverage():
    tone = _tone(level=0.4)
    dark, _ = screen_with_colour(tone, [(_half(), ColourZone(hold_tone=False))],
                                 cell_um=CELL, screen_period_um=SCREEN, tone_steps=STEPS)
    held, _ = screen_with_colour(tone, [(_half(), ColourZone(hold_tone=True))],
                                 cell_um=CELL, screen_period_um=SCREEN, tone_steps=STEPS)
    z = slice(100, None)
    assert held[:, z].mean() > dark[:, z].mean() * 1.5


def test_later_zones_win_where_they_overlap():
    tone = _tone(level=0.8)
    n = tone.shape[0]
    all_of_it = np.ones((n, n), bool)
    a = ColourZone(period_scale=1.0, label="a")
    b = ColourZone(period_scale=1.4, label="b")
    m1, _ = screen_with_colour(tone, [(all_of_it, a), (all_of_it, b)],
                               cell_um=CELL, screen_period_um=SCREEN, tone_steps=STEPS)
    m2, _ = screen_with_colour(tone, [(all_of_it, b)],
                               cell_um=CELL, screen_period_um=SCREEN, tone_steps=STEPS)
    assert np.array_equal(m1, m2)


def test_the_sub_grating_defaults_to_crossing_the_screen():
    """Running it along the screen would just merge with the screen's own lines."""
    tone = _tone()
    _, rep = screen_with_colour(tone, [(_half(), ColourZone())], cell_um=CELL,
                                screen_period_um=SCREEN, tone_steps=STEPS,
                                screen_angle_deg=17.0)
    assert rep["zones"][0]["angle_deg"] == pytest.approx(107.0)


def test_an_explicit_angle_is_honoured():
    tone = _tone()
    _, rep = screen_with_colour(tone, [(_half(), ColourZone(angle_deg=45.0))],
                                cell_um=CELL, screen_period_um=SCREEN, tone_steps=STEPS)
    assert rep["zones"][0]["angle_deg"] == pytest.approx(45.0)


# --- guards -----------------------------------------------------------------


def test_bad_zone_is_refused():
    with pytest.raises(ValueError):
        ColourZone(period_scale=0.0)
    with pytest.raises(ValueError):
        ColourZone(duty=0.99)


def test_shape_and_scale_mismatches_are_refused():
    with pytest.raises(ValueError):
        screen_with_colour(np.zeros((4, 4, 3), np.float32), [], cell_um=CELL,
                           screen_period_um=SCREEN, tone_steps=STEPS)
    with pytest.raises(ValueError):
        screen_with_colour(_tone(), [(np.ones((9, 9), bool), ColourZone())],
                           cell_um=CELL, screen_period_um=SCREEN, tone_steps=STEPS)
    with pytest.raises(ValueError):
        screen_with_colour(_tone(), [], cell_um=0.0, screen_period_um=SCREEN,
                           tone_steps=STEPS)


def test_no_zones_is_a_plain_halftone():
    # A WHOLE number of screen periods: 44 um at 0.5 um cells is 88 px, so 264
    # px is exactly 3. A partial period at the edge biases the mean (200 px is
    # 2.27 periods and reads 0.56), which is the raster, not the screen.
    n = int(round(SCREEN / CELL)) * 3
    tone = np.full((n, n), 0.5, dtype=np.float32)
    mask, rep = screen_with_colour(tone, [], cell_um=CELL, screen_period_um=SCREEN,
                                   tone_steps=STEPS)
    assert rep["zones"] == []
    assert mask.mean() == pytest.approx(0.5, abs=0.01)
