"""Shading moire: the beat, and its robustness.

The SHADING MOIRE is the hidden exemplar the lid used to be — one figure on the
outer ply, the uniform carrier on the inner one, and a beat between them from a
pitch MISMATCH rather than from a crossing angle. No production face carries it
since the box became six single plies (``monogram-jp`` composed with
``single_ply=False`` is the only spec that still enters the path), but the
construction is kept built and this is what pins it: the solver, the geometry
``shimmer_moire_layers`` emits, and the two robustness arguments that chose a
pitch beat over an angle beat in the first place.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app import plates
from app.patterns.effects.gratings import beat_delta_um, shimmer_moire_layers
from app.witness_geom import combined_beat_um

ACUITY_ARCMIN = 2.0


def subtense_arcmin(feature_um: float, distance_mm: float = 300.0) -> float:
    """Angular size of a feature (um) at a viewing distance (mm), in arcmin."""
    return math.degrees((feature_um / 1000.0) / distance_mm) * 60.0


# --- the algebra ------------------------------------------------------------


def test_delta_solves_the_beat_equation():
    """beat = p(p+d)/d, so the solver must round-trip."""
    for period, beat in ((22.0, 1635.0), (63.5, 1635.0), (10.0, 500.0)):
        d = beat_delta_um(period, beat)
        assert period * (period + d) / d == pytest.approx(beat, rel=1e-9)


def test_a_beat_finer_than_its_carrier_is_refused():
    with pytest.raises(ValueError):
        beat_delta_um(22.0, 22.0)
    with pytest.raises(ValueError):
        beat_delta_um(22.0, 10.0)


def test_the_beat_is_invariant_to_carrier_pitch():
    """The bug this guards: the carrier is GAP-SCALED with the glass, so it is
    22 um at the 500 um baseline and 63.5 um on 1.5 mm stock. Pinning the pitch
    MISMATCH instead of the beat lets the beat follow the carrier up by that
    same 2.9x — a dozen bands across the lid become one and a half.
    """
    fixed_delta = 0.3
    assert 22.0 * (22.0 + fixed_delta) / fixed_delta == pytest.approx(1635, rel=0.01)
    stretched = 63.5 * (63.5 + fixed_delta) / fixed_delta
    assert stretched > 13000  # 13.5 mm — 1.5 bands across a 20 mm lid

    # Solving delta from the beat holds the look on both.
    for p in (22.0, 63.5):
        d = beat_delta_um(p, 1635.0)
        assert p * (p + d) / d == pytest.approx(1635.0, rel=1e-9)


# --- the geometry actually built --------------------------------------------


def test_both_layers_are_real_gratings():
    """The back used to be all-glass, which is what made the second plate
    pointless on these faces."""
    art = np.ones((400, 400), dtype=bool)
    front, back = shimmer_moire_layers(
        art, back_period_um=22.0, delta_um=0.3, cell_um=2.0
    )
    assert 0.4 < back.mean() < 0.6  # ~50% duty, full field
    assert 0.4 < front.mean() < 0.6


def test_the_front_grating_is_clipped_to_the_figure():
    art = np.zeros((300, 300), dtype=bool)
    art[100:200, 100:200] = True
    front, back = shimmer_moire_layers(
        art, back_period_um=22.0, delta_um=0.3, cell_um=2.0
    )
    assert not front[:100].any()  # nothing outside the figure
    assert back[:100].any()       # ...but the carrier spans everything


def test_a_delta_that_cancels_the_carrier_is_refused():
    art = np.ones((50, 50), dtype=bool)
    with pytest.raises(ValueError):
        shimmer_moire_layers(art, back_period_um=22.0, delta_um=-22.0, cell_um=2.0)


def test_the_lattice_stays_invisible_while_the_beat_reads():
    """The whole construction: gratings below acuity, beat above it."""
    assert subtense_arcmin(22.0, 300.0) < 0.5
    assert subtense_arcmin(1635.0, 300.0) > ACUITY_ARCMIN


# --- robustness to the one thing this build cannot hold ---------------------


def test_a_pitch_derived_beat_survives_misregistration():
    """No backside alignment means front-to-back ROTATION is uncontrolled. A
    pitch-derived beat only drifts; it can never vanish."""
    d = beat_delta_um(22.0, 1635.0)
    for err in (0.0, 0.5, 1.0, 2.0):
        beat = combined_beat_um(22.0 + d, 22.0, err)
        assert math.isfinite(beat)
        assert subtense_arcmin(beat, 300.0) > ACUITY_ARCMIN, f"{err} deg killed it"
    # Half a degree costs about a fifth, not a factor.
    near = combined_beat_um(22.0 + d, 22.0, 0.5)
    assert 0.75 < near / 1635.0 < 1.0


def test_an_angle_derived_beat_does_not():
    """Why the beat is NOT taken from a crossing angle. Designed at 0.45 deg it
    reads 2801 um, but a degree of flip error more than halves it — and landing
    square gives no fringes at all."""
    assert combined_beat_um(22.0, 22.0, 0.45) == pytest.approx(2801, rel=0.05)
    assert combined_beat_um(22.0, 22.0, 1.45) < 900
    assert math.isinf(combined_beat_um(22.0, 22.0, 0.0))


# --- what the plate publishes -----------------------------------------------


def _spec(slug: str):
    return plates.PlateSpec(
        pattern_slug=slug,
        frame=plates.FrameSpec(seed=1),
        glass=plates.GlassSpec(thickness_um=1500.0, n=1.52),
        width_um=28700.0,
        height_um=28700.0,
    )


@pytest.mark.parametrize("slug", sorted(plates.SHIMMER_MOIRE_SLUGS))
def test_shimmer_faces_get_a_readable_beat_on_the_real_glass(slug):
    rd = plates._carrier_recipe_data(_spec(slug))
    beat = combined_beat_um(
        rd["fab_center_period_um"],
        rd["carrier_period_um"],
        rd["switch_axis_deg"] - rd["carrier_angle_deg"],
    )
    assert rd["switch_axis_deg"] == pytest.approx(rd["carrier_angle_deg"])  # parallel
    assert beat == pytest.approx(plates.CENTERPIECE_BEAT_UM, rel=0.01)
    assert subtense_arcmin(beat, 300.0) > ACUITY_ARCMIN


@pytest.mark.parametrize("slug", ["globe-duo-phase"])
def test_switch_and_comb_faces_are_untouched(slug):
    """On these the centerpiece pitch and axis set a SWITCH ANGLE, not a beat,
    and must not be retuned for fringe aesthetics."""
    rd = plates._carrier_recipe_data(_spec(slug))
    assert rd["switch_axis_deg"] == pytest.approx(plates.CENTER_SWITCH_AXIS_DEG)
    assert rd["fab_center_period_um"] == pytest.approx(
        plates.fab_center_period_um(_spec(slug))
    )


def test_the_fill_tracks_the_per_face_carrier_angle():
    """The original defect: a fixed 0 deg fill against a per-face carrier angle
    ((seed*17) mod 180). Two seeds must give two different fill angles, each
    still parallel to its own carrier."""
    seen = set()
    for seed in (1, 2, 3):
        spec = _spec("monogram-jp")
        spec.frame.seed = seed
        rd = plates._carrier_recipe_data(spec)
        assert rd["switch_axis_deg"] == pytest.approx(rd["carrier_angle_deg"])
        seen.add(round(rd["switch_axis_deg"], 3))
    assert len(seen) > 1
