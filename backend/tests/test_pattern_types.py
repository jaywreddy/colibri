"""Pattern-type evaluation harness (app.sim2d metrics) — synthetic masks only.

Every mask is built inline with numpy (tiny 96x96 grids, pixel pitch 1 um,
carrier period 16 um = 16 px); NO pattern generation, file I/O, or registry
access — milliseconds-fast. Each test encodes one row of the parallax
taxonomy:

  * barrier switch (T1)    — front slit + back two-channel interlace: the
    channel-visibility separation detects a clean switch in the first zone,
    for both the slit-centered and the straddle (quarter-offset) slit
    registration.
  * borked phase pair (TX) — image A on the FRONT face + image B on an
    anti-phase carrier in the back (the old jp-monogram-phase construction).
    Regression for the user-reported bug: the front image never moves under
    parallax, so this can never switch — separation must degenerate to ~1
    and both tilt signs must show the same muddled double exposure.
  * carrier-phase reveal (T5/T6) — anti-phase carriers: strong transmission
    modulation with the maximum exactly at the half-period shift.
  * moire fringes (T3)     — detuned gratings decorrelate per quarter-period
    step; identical gratings do not.
  * zone aliasing          — a full-period shift (zone 2) repeats zone 0.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.sim2d import (
    fringe_metrics,
    parallax_shift_um,
    reveal_metrics,
    shift_for_zone,
    sweep_transmission,
    switch_metrics,
    tilt_for_shift_um,
)

SIZE = 96
PERIOD = 16.0  # um; == 16 px at PITCH
PITCH = 1.0
PERIOD_PX = int(PERIOD / PITCH)


# ---------------------------------------------------------------------------
# synthetic building blocks
# ---------------------------------------------------------------------------


def _disk(cx: int, cy: int, r: int, size: int = SIZE) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _cols(offset_px: int = 0, period_px: int = PERIOD_PX, size: int = SIZE) -> np.ndarray:
    """Boolean column mask: True where ((x - offset) mod p) < p/2."""
    row = ((np.arange(size) - offset_px) % period_px) < period_px // 2
    return np.tile(row, (size, 1))


def _grating(period_px: int, phase_px: int = 0, size: int = SIZE) -> np.ndarray:
    """Float 0/1 vertical stripe grating: gold where ((x - phase) mod p) < p/2."""
    row = ((np.arange(size) - phase_px) % period_px) < period_px / 2
    return np.tile(row, (size, 1)).astype(np.float32)


def _to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")


# Two disjoint image silhouettes, vertically separated so x shifts up to a
# full period (16 px) never push their gold out of the frame.
IMAGE_A = _disk(48, 30, 13)
IMAGE_B = _disk(48, 66, 13)


def _barrier_pair(
    image_a: np.ndarray, image_b: np.ndarray, slit_offset_px: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """T1 barrier: BACK = image A interlaced in channel-A columns
    ((x mod p) < p/2) with image B in channel-B columns; FRONT = slit
    grating open where ((x - offset) mod p) < p/2.

    slit_offset_px=0: slit centered over the A channel (the colibri
    registration — the hidden channel appears at BOTH signs of a p/2 shift).
    slit_offset_px=p/4: straddle registration — symmetric A <-> B switch at
    +/- p/4 of shift.
    """
    chan_a_cols = _cols()
    back = (image_a & chan_a_cols) | (image_b & ~chan_a_cols)
    front = ~_cols(offset_px=slit_offset_px)  # gold outside the open slit
    return front.astype(np.float32), back.astype(np.float32)


def _borked_phase_pair(
    image_a: np.ndarray, image_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """TX anti-pattern (old jp-monogram-phase): image A AND carrier(phase 0)
    on the FRONT face; image B AND carrier(phase pi) in the back. The front
    image is static under parallax, so no shift can ever make it vanish."""
    front = image_a & _cols()
    back = image_b & ~_cols()
    return front.astype(np.float32), back.astype(np.float32)


# ---------------------------------------------------------------------------
# zone math and Snell inversion
# ---------------------------------------------------------------------------


def test_shift_for_zone_half_period_multiples() -> None:
    assert shift_for_zone(40.0, 0) == 0.0
    assert shift_for_zone(40.0, 1) == 20.0
    assert shift_for_zone(40.0, 2) == 40.0
    assert shift_for_zone(60.0, 3) == 90.0


def test_tilt_for_shift_anchor_and_roundtrip() -> None:
    # Anchor from the investigation table: 20 um at t=500 um, n=1.46 -> 3.35 deg.
    assert tilt_for_shift_um(20.0, 500.0, 1.46) == pytest.approx(3.345, abs=0.01)
    # Exact inverse of the forward Snell shift.
    dx, _ = parallax_shift_um(14.0, 0.0, 500.0, 1.46)
    assert tilt_for_shift_um(dx, 500.0, 1.46) == pytest.approx(14.0, rel=1e-9)
    # Odd symmetry.
    assert tilt_for_shift_um(-20.0, 500.0, 1.46) == pytest.approx(
        -tilt_for_shift_um(20.0, 500.0, 1.46), rel=1e-12
    )


def test_tilt_for_shift_beyond_grazing_raises() -> None:
    # n*sin(atan(s/t)) >= 1 has no exterior solution (|s| >= ~470 um here).
    with pytest.raises(ValueError):
        tilt_for_shift_um(1000.0, 500.0, 1.46)


# ---------------------------------------------------------------------------
# sweep_transmission
# ---------------------------------------------------------------------------


def test_sweep_transmission_identical_gratings_flat_at_whole_periods() -> None:
    g = _grating(PERIOD_PX)
    rows = sweep_transmission(g, g, PITCH, [0.0, 16.0, 32.0], axis="x")
    assert [row["shift_um"] for row in rows] == [0.0, 16.0, 32.0]
    assert set(rows[0]) == {"shift_um", "transmission"}
    values = [row["transmission"] for row in rows]
    assert max(values) - min(values) < 1e-6


def test_sweep_transmission_bad_axis_raises() -> None:
    g = _grating(PERIOD_PX)
    with pytest.raises(ValueError):
        sweep_transmission(g, g, PITCH, [0.0], axis="z")


def test_sweep_transmission_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        sweep_transmission(
            _grating(PERIOD_PX, size=96), _grating(PERIOD_PX, size=64), PITCH, [0.0]
        )


# ---------------------------------------------------------------------------
# T1 barrier switch — GOOD constructions
# ---------------------------------------------------------------------------


def test_barrier_switch_slit_centered_first_zone() -> None:
    """Slit centered over channel A: head-on shows A; a half-period shift
    (zone 1) gates channel B through the slit with total extinction of A."""
    front, back = _barrier_pair(IMAGE_A, IMAGE_B, slit_offset_px=0)
    m = switch_metrics(front, back, PITCH, PERIOD)  # default shift = p/2
    # Hidden channel fully exposed, visible channel fully extinguished.
    assert m["vis_b_plus"] > 0.95
    assert m["vis_a_plus"] < 0.05
    assert m["vis_b_minus"] > 0.95
    assert m["vis_a_minus"] < 0.05
    assert m["separation"] > 10.0
    # Head-on (zone 0) it is channel A that shows: the channels flip.
    m0 = switch_metrics(front, back, PITCH, PERIOD, shift_um=0.0)
    assert m0["vis_a_plus"] > 0.95
    assert m0["vis_b_plus"] < 0.05


def test_barrier_switch_straddle_symmetric() -> None:
    """Straddle registration (slit offset p/4): +p/4 shows A, -p/4 shows B —
    a symmetric left/right switch whose two views genuinely differ (low
    darkness correlation through the apertures)."""
    front, back = _barrier_pair(IMAGE_A, IMAGE_B, slit_offset_px=PERIOD_PX // 4)
    m = switch_metrics(front, back, PITCH, PERIOD, shift_um=PERIOD / 4.0)
    assert m["vis_a_plus"] > 0.95
    assert m["vis_b_plus"] < 0.05
    assert m["vis_b_minus"] > 0.95
    assert m["vis_a_minus"] < 0.05
    assert m["separation"] > 10.0
    # The two tilt-sign views show different images.
    assert m["corr"] < 0.3


def test_switch_metrics_axis_y_matches_transposed_x() -> None:
    front, back = _barrier_pair(IMAGE_A, IMAGE_B, slit_offset_px=0)
    mx = switch_metrics(front, back, PITCH, PERIOD)
    my = switch_metrics(front.T, back.T, PITCH, PERIOD, axis="y")
    assert my["separation"] == pytest.approx(mx["separation"], rel=1e-6)
    assert my["vis_b_plus"] == pytest.approx(mx["vis_b_plus"], rel=1e-6)
    assert my["vis_a_plus"] == pytest.approx(mx["vis_a_plus"], abs=1e-6)


# ---------------------------------------------------------------------------
# TX borked phase pair — regression for the user-reported bug
# ---------------------------------------------------------------------------


def test_borked_front_image_phase_pair_fails_switch() -> None:
    """The old monogram construction (image A front / image B back on an
    anti-phase carrier) must be flagged as a non-switch: all back gold sits
    in a single column phase, so separation degenerates to ~1, image B leaks
    through at BOTH tilt signs, and the two views are the same muddled
    double exposure (high lit-pixel overlap)."""
    # Overlapping footprints, like the real monogram + heart-globe pair.
    image_a = _disk(48, 48, 16)
    image_b = np.zeros((SIZE, SIZE), dtype=bool)
    image_b[28:68, 28:68] = True
    front, back = _borked_phase_pair(image_a, image_b)

    borked = switch_metrics(front, back, PITCH, PERIOD)  # +/- p/2, zone 1
    # Not interlaced -> structurally cannot switch.
    assert borked["separation"] < 1.5
    # Image B partially visible at BOTH signs — no gating, just mud.
    assert 0.2 < borked["vis_b_plus"] < 0.9
    assert 0.2 < borked["vis_b_minus"] < 0.9
    # Both tilt directions show essentially the same view.
    assert borked["overlap_frac"] > 0.8

    # Direct regression comparison against a correct barrier build of the
    # same idea: the good construction separates >100x better.
    good_front, good_back = _barrier_pair(IMAGE_A, IMAGE_B, slit_offset_px=0)
    good = switch_metrics(good_front, good_back, PITCH, PERIOD)
    assert good["separation"] > 100.0 * borked["separation"]


# ---------------------------------------------------------------------------
# zone aliasing — the period repeat
# ---------------------------------------------------------------------------


def test_switch_zone_aliasing_full_period_repeats_zone_zero() -> None:
    """A full-period shift (zone 2 = shift_for_zone(p, 2) = p) re-registers
    the interlace: the switch metric must match zone 0 (head-on), which is
    exactly the aliasing that made the +/-14 deg demo look muddled."""
    front, back = _barrier_pair(IMAGE_A, IMAGE_B, slit_offset_px=0)
    m0 = switch_metrics(front, back, PITCH, PERIOD, shift_um=shift_for_zone(PERIOD, 0))
    m2 = switch_metrics(front, back, PITCH, PERIOD, shift_um=shift_for_zone(PERIOD, 2))
    assert abs(m2["vis_a_plus"] - m0["vis_a_plus"]) < 0.05
    assert abs(m2["vis_b_plus"] - m0["vis_b_plus"]) < 0.05
    assert m0["separation"] > 10.0
    assert m2["separation"] > 10.0
    # Zone 1 (the actual switch) is a genuinely different view.
    m1 = switch_metrics(front, back, PITCH, PERIOD, shift_um=shift_for_zone(PERIOD, 1))
    assert m1["vis_a_plus"] < 0.05  # A hidden in zone 1, visible in zones 0/2


# ---------------------------------------------------------------------------
# T5/T6 carrier-phase reveal
# ---------------------------------------------------------------------------


def test_reveal_antiphase_carrier_max_at_half_period() -> None:
    """Anti-phase 50%-duty carriers: extinction at registration, full 0.5
    transmission exactly at s = p/2 (ideal T(s) = s/p). Exercises the PIL
    input path."""
    front = _to_image(_grating(PERIOD_PX, phase_px=0))
    back = _to_image(_grating(PERIOD_PX, phase_px=PERIOD_PX // 2))
    r = reveal_metrics(front, back, PITCH, PERIOD)
    assert r["modulation_ratio"] > 5.0
    assert r["t_min"] < 1e-6
    assert r["t_max"] == pytest.approx(0.5, abs=0.02)
    assert r["shift_at_max_um"] == pytest.approx(PERIOD / 2.0, abs=PITCH)
    assert r["shift_at_min_um"] == pytest.approx(0.0, abs=PITCH)
    # Curve covers 0 .. p/2 inclusive at pixel steps and rises monotonically.
    shifts = [row["shift_um"] for row in r["curve"]]
    assert shifts[0] == 0.0 and shifts[-1] == PERIOD / 2.0
    values = [row["transmission"] for row in r["curve"]]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_reveal_in_phase_pair_extinguishes_at_half_period() -> None:
    """In-phase identical carriers are the mirror image: brightest at
    registration, extinct at s = p/2 (the 'blink')."""
    g = _grating(PERIOD_PX)
    r = reveal_metrics(g, g, PITCH, PERIOD)
    assert r["modulation_ratio"] > 5.0
    assert r["shift_at_max_um"] == pytest.approx(0.0, abs=PITCH)
    assert r["shift_at_min_um"] == pytest.approx(PERIOD / 2.0, abs=PITCH)


def test_reveal_static_pair_is_flat() -> None:
    """No back layer -> no modulation: ratio ~1 rejects non-reveals."""
    front = _grating(PERIOD_PX)
    back = np.zeros((SIZE, SIZE), dtype=np.float32)
    r = reveal_metrics(front, back, PITCH, PERIOD)
    assert r["modulation_ratio"] < 1.2


# ---------------------------------------------------------------------------
# T3 moire fringe flow
# ---------------------------------------------------------------------------


def test_fringe_detuned_gratings_decorrelate() -> None:
    """p_f=8 vs p_b=9 px: the 72 px beat envelope glides at magnification
    p_f/(p_f - p_b) = -8x per unit shift, so quarter-period steps strongly
    decorrelate the low-passed field."""
    fm = fringe_metrics(_grating(8), _grating(9), PITCH, 8.0)
    assert fm["decorrelation_per_quarter"] > 0.3
    assert fm["fringe_std"] > 0.03


def test_fringe_identical_gratings_static() -> None:
    """Identical gratings carry no beat: the low-passed field is constant at
    every shift (guarded corr = 1), so decorrelation and visibility ~0."""
    front = _grating(8)
    back_u8 = (_grating(8) * 255).astype(np.uint8)  # exercise integer input
    fm = fringe_metrics(front, back_u8, PITCH, 8.0)
    assert fm["decorrelation_per_quarter"] < 0.05
    assert fm["fringe_std"] < 0.01
    assert fm["corr_full"] == pytest.approx(1.0, abs=1e-6)


def test_fringe_frame_too_small_raises() -> None:
    g = _grating(8, size=32)
    with pytest.raises(ValueError):
        fringe_metrics(g, g, PITCH, 8.0)
