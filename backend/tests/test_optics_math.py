"""The optical identities the box is designed on, as FORMULAS.

These numbers used to be pinned indirectly, through witness-plate cells that
computed them and stuffed them in a ``stats`` dict: a beat ladder, a rotation
ladder, a harmonic pair, a near-field pair, a parallax ruler. Those cells went
with the two-ply design (2026-09-16) — the box is six single plies, and half of
them measure a gap it no longer has — but the identities did not: they still set
the garland's angle fan, the lid's shading moire, the carrier pitch and the
barrier comb.

So they are pinned HERE, against the functions that compute them
(``witness_geom``, ``patterns.effects``), with no geometry built at all. That
is also why this file is cheap: no cell, no raster, no GEOS — milliseconds on a
13.7 GB host.
"""
from __future__ import annotations

import math

import pytest

from app.patterns.effects import moire as em
from app.patterns.effects.gratings import beat_delta_um
from app.witness_geom import (
    BOX_BEAT_UM,
    BOX_CARRIER_UM,
    BOX_COMB_UM,
    BOX_MONO_UM,
    GLASS_N,
    PARALLAX_UM_PER_DEG,
    PLY_UM,
    P_MIN_UM,
    beat_delta,
    combined_beat_um,
    fresnel_number,
    harmonic_beats,
    parallax_um_per_deg,
    swap_deg,
)


# --- the beat is solved FROM the beat ----------------------------------------


@pytest.mark.parametrize("beat_um", [500.0, 1000.0, 1635.0, 3000.0, 6000.0])
def test_beat_delta_inverts_the_beat_identity(beat_um):
    """``p (p + d) / d == beat``. The design picks the BEAT — it is the thing
    the eye reads — and solves the pitch difference out of it; pinning delta
    instead is what once made a cell read 0.075 where it should have read 0.47.
    """
    p = BOX_CARRIER_UM
    d = beat_delta(p, beat_um)
    assert p * (p + d) / d == pytest.approx(beat_um, rel=1e-9)


def test_the_two_beat_delta_solvers_are_one_formula():
    """``witness_geom.beat_delta`` (the plate side) and
    ``effects.gratings.beat_delta_um`` (the compositor side) must agree: the
    lid's monogram pitch comes from one and its witness numbers from the other.
    """
    for p in (22.0, 63.5, BOX_CARRIER_UM):
        for beat in (800.0, 1635.0, 4000.0):
            assert beat_delta(p, beat) == pytest.approx(beat_delta_um(p, beat), rel=1e-12)


def test_a_beat_finer_than_its_own_pitch_is_refused():
    with pytest.raises(ValueError):
        beat_delta(63.5, 50.0)


def test_the_lids_monogram_pitch_beats_the_carrier_at_the_design_beat():
    """``BOX_MONO_UM`` is not a chosen pitch — it is ``BOX_CARRIER_UM`` plus
    whatever delta puts the beat at ``BOX_BEAT_UM``, which is the shading-moire
    exemplar's whole construction."""
    assert BOX_MONO_UM > BOX_CARRIER_UM
    assert combined_beat_um(BOX_CARRIER_UM, BOX_MONO_UM, 0.0) == pytest.approx(
        BOX_BEAT_UM, rel=1e-9
    )
    # And the beat AMPLIFIES a pitch error by p/delta — ~25x at the shipping
    # numbers, which is what made the beat a metrology read and not decoration.
    assert BOX_CARRIER_UM / beat_delta(BOX_CARRIER_UM, BOX_BEAT_UM) > 20.0


# --- rotation vs. vector: one formula, two special cases ---------------------


@pytest.mark.parametrize("angle_deg", [0.5, 1.0, 2.0, 3.0, 6.0])
def test_the_rotation_beat_is_the_equal_pitch_case_of_the_vector_beat(angle_deg):
    """``p / (2 sin(a/2))`` is ``|k1 - k2|`` at p1 == p2. If these disagreed the
    perimeter garland's angle fan — which is designed on the rotation form —
    would be numbered from the wrong equation."""
    p = 63.5
    rot = p / (2.0 * math.sin(math.radians(angle_deg) / 2.0))
    assert combined_beat_um(p, p, angle_deg) == pytest.approx(rot, rel=1e-9)
    assert em.beat_period_rotated(p, angle_deg) == pytest.approx(rot, rel=1e-9)


def test_the_pitch_beat_is_the_zero_angle_case_of_the_vector_beat():
    """And at a == 0 the same formula collapses to ``p1 p2 / |p1 - p2|``, which
    is the shading moire's."""
    p1, p2 = 63.5, 66.07
    assert combined_beat_um(p1, p2, 0.0) == pytest.approx(p1 * p2 / (p2 - p1), rel=1e-12)
    assert em.beat_period_parallel(p1, p2) == pytest.approx(p1 * p2 / (p2 - p1), rel=1e-12)


def test_two_identical_gratings_head_on_have_no_beat():
    assert combined_beat_um(63.5, 63.5, 0.0) == float("inf")


# --- the even-harmonic moire exists ONLY under duty bias ---------------------


def test_the_even_harmonic_moire_exists_only_under_duty_bias():
    """The best argument for the plate's duty ladder. The (2,3) beat between the
    44 um halftone screen and the 63.5 um carrier is 559 um — 6.4 arcmin,
    plainly visible banding across a photograph — and its amplitude is
    identically ZERO at 50 % duty, because the even harmonics of a square wave
    vanish there. A process error conjures a moire the nominal design has not
    got."""
    at50 = {(t["m"], t["n"]): t for t in harmonic_beats(44.0, 63.5, 0.50)}
    at42 = {(t["m"], t["n"]): t for t in harmonic_beats(44.0, 63.5, 0.42)}
    assert at50[(2, 3)]["arcmin"] == pytest.approx(6.42, abs=0.05)
    assert at50[(2, 3)]["amplitude"] == 0.0
    assert at42[(2, 3)]["amplitude"] > 0.0
    # while the (1,1) beat is there at both duties, and is the one the design
    # accounts for: 1.65 arcmin, under the eye's ~2 arcmin limit at 300 mm.
    assert at50[(1, 1)]["arcmin"] == pytest.approx(1.65, abs=0.03)
    assert at50[(1, 1)]["amplitude"] > 0.1
    assert at42[(1, 1)]["amplitude"] > 0.09
    # Biasing the duty can only ADD beats, never remove them.
    vis = lambda tab: {k for k, t in tab.items() if t["arcmin"] > 1.5 and t["amplitude"] > 2e-3}
    assert vis(at50) < vis(at42)


# --- the near field: what a gap does to a fringe ------------------------------


def test_p_min_is_the_fresnel_half_null_of_this_glass():
    """A p/2 slit spreads by ``lambda z / (n p)`` across the gap, so a fringe
    pattern survives one ply only above ``p_min = sqrt(2 lambda z / n)``. On the
    2.25 mm quartz this box is cut from that is 41 um — which is why the
    carrier is 65.5 um and not the 22 um design pitch, and why a 15 um
    scanimation slot could never have worked on this stock."""
    assert P_MIN_UM == pytest.approx(math.sqrt(2.0 * 0.55 * PLY_UM / GLASS_N), rel=1e-12)
    assert P_MIN_UM == pytest.approx(41.2, abs=0.2), "2.25 mm quartz"
    assert fresnel_number(P_MIN_UM) == pytest.approx(0.5, rel=1e-9)
    # the box carrier sits above the boundary, a colour grating far below it
    assert fresnel_number(BOX_CARRIER_UM) > 1.0
    assert fresnel_number(5.0) < 0.05
    # and the boundary is bracketed by the ladder the plate used to carry
    assert 20.0 < P_MIN_UM < 64.0


# --- parallax and the swap angle ---------------------------------------------


def test_the_parallax_rate_is_the_refracted_walk_across_one_ply():
    assert PARALLAX_UM_PER_DEG == pytest.approx(26.93, abs=0.05), "2.25 mm quartz"
    assert parallax_um_per_deg(PLY_UM, GLASS_N) == pytest.approx(PARALLAX_UM_PER_DEG)
    # thinner glass walks slower, a higher index walks slower
    assert parallax_um_per_deg(PLY_UM / 2, GLASS_N) < PARALLAX_UM_PER_DEG
    assert parallax_um_per_deg(PLY_UM, 2.0) < PARALLAX_UM_PER_DEG


def test_the_barrier_swaps_within_a_comfortable_hand_tilt():
    """A straddle-registered barrier peaks after the back layer walks p/4 (the
    rule CLAUDE.md states for the switch exemplar), and the comb pitch is chosen
    so that lands in the 2-8 deg hand sweep on THIS glass."""
    a = swap_deg(BOX_COMB_UM)
    assert 2.0 <= a <= 8.0, a
    # monotone in pitch, and the p/4 walk is what it inverts
    assert swap_deg(BOX_COMB_UM * 2) > a > swap_deg(BOX_COMB_UM / 2)
    walk = PLY_UM * math.tan(math.asin(math.sin(math.radians(a)) / GLASS_N))
    assert walk == pytest.approx(BOX_COMB_UM / 4.0, rel=1e-9)
