"""Perceptual readability budget — pinned to closed forms and known anchors."""
from __future__ import annotations

import math

import pytest

from app.readability import (
    DEFAULT_DISTANCE_MM,
    beat_period_angle_um,
    beat_period_um,
    moire_readability,
    subtense_arcmin,
    switch_readability,
    tilt_for_shift_deg,
    verdict,
)


# --- primitives -------------------------------------------------------------


def test_subtense_matches_small_angle_formula():
    # 87.3 um at 300 mm is ~1 arcmin — the classic acuity anchor.
    assert subtense_arcmin(87.3, 300.0) == pytest.approx(1.0, abs=0.02)
    # Linear in size, inverse in distance.
    assert subtense_arcmin(200, 300) == pytest.approx(2 * subtense_arcmin(100, 300))
    assert subtense_arcmin(100, 600) == pytest.approx(subtense_arcmin(100, 300) / 2)


def test_tilt_for_shift_matches_sim2d_anchors():
    """Same Snell relation the simulator uses: at t=500 um, n=1.46,
    20 um -> 3.345 deg and 40 um -> 6.69 deg (sim2d docstring anchors)."""
    assert tilt_for_shift_deg(20.0, 500.0, 1.46) == pytest.approx(3.345, abs=0.01)
    assert tilt_for_shift_deg(40.0, 500.0, 1.46) == pytest.approx(6.69, abs=0.02)


def test_tilt_beyond_grazing_is_nan():
    assert math.isnan(tilt_for_shift_deg(5000.0, 500.0, 1.46))


def test_beat_periods_are_the_classic_forms():
    assert beat_period_um(22.0, 24.0) == pytest.approx(22 * 24 / 2)
    assert math.isinf(beat_period_um(22.0, 22.0))
    # Crossed equal gratings: p / (2 sin(a/2)).
    assert beat_period_angle_um(22.0, 5.0) == pytest.approx(
        22.0 / (2 * math.sin(math.radians(5.0) / 2)), rel=1e-9
    )
    assert math.isinf(beat_period_angle_um(22.0, 0.0))


# --- switch faces -----------------------------------------------------------


def test_original_fused_silica_switch_passes_every_gate():
    """60 um barrier on 500 um fused silica — the design the box was tuned for."""
    r = switch_readability(
        face="front", barrier_pitch_um=60.0, thickness_um=500.0, n=1.46
    )
    assert r.passes, [c.to_dict() for c in r.checks if not c.passes]
    assert r.summary["swap_deg"] == pytest.approx(2.5, abs=0.1)
    assert r.summary["lane_arcmin"] < 0.4


def test_thick_bonded_build_flags_the_lane_visibility():
    """1.5 mm soda lime keeps the SAME swap angle (pitch scaled with the gap)
    but pushes the lanes to ~1 arcmin, where the eye starts to resolve them —
    the real consequence of choosing thick stock."""
    r = switch_readability(
        face="front", barrier_pitch_um=173.0, thickness_um=1500.0, n=1.52
    )
    assert r.summary["swap_deg"] == pytest.approx(2.5, abs=0.15)  # angle preserved
    lane = next(c for c in r.checks if c.name == "lane invisible")
    assert lane.value == pytest.approx(0.99, abs=0.05)
    assert not lane.passes                      # this is the gate that catches it
    # ...and it recovers at arm's length.
    far = switch_readability(
        face="front", barrier_pitch_um=173.0, thickness_um=1500.0, n=1.52,
        distance_mm=500.0,
    )
    assert next(c for c in far.checks if c.name == "lane invisible").passes


def test_scaling_pitch_with_gap_preserves_the_swap_angle():
    """The invariant the whole glass-scaling design rests on."""
    a = switch_readability(face="f", barrier_pitch_um=60.0, thickness_um=500.0, n=1.46)
    b = switch_readability(face="f", barrier_pitch_um=173.0, thickness_um=1500.0, n=1.52)
    assert a.summary["swap_deg"] == pytest.approx(b.summary["swap_deg"], abs=0.1)


def test_too_coarse_a_barrier_fails_on_lanes_not_on_angle():
    r = switch_readability(
        face="f", barrier_pitch_um=400.0, thickness_um=1500.0, n=1.52
    )
    assert not next(c for c in r.checks if c.name == "lane invisible").passes


# --- moiré faces ------------------------------------------------------------


def test_frame_moire_at_the_shipping_pitches_reads():
    """22 um carrier vs 23.98 um louvre: gratings invisible, beat visible."""
    r = moire_readability(
        face="top", back_pitch_um=22.0, front_pitch_um=23.98,
        angle_offset_deg=3.0, thickness_um=500.0, n=1.46,
    )
    assert r.summary["grating_arcmin"] < 0.3          # lattice invisible
    assert r.summary["beat_arcmin"] > 2.0             # fringes legible
    assert r.passes, [c.to_dict() for c in r.checks if not c.passes]


def test_identical_pitches_with_no_angle_have_no_beat():
    r = moire_readability(
        face="x", back_pitch_um=22.0, front_pitch_um=22.0,
        angle_offset_deg=0.0, thickness_um=500.0, n=1.46,
    )
    assert r.summary["beat_um"] is None               # infinite beat = no fringes
    assert not r.passes


def test_angle_offset_alone_can_carry_the_beat():
    """Even with equal pitches, a crossing angle produces fringes."""
    r = moire_readability(
        face="x", back_pitch_um=22.0, front_pitch_um=22.0,
        angle_offset_deg=3.0, thickness_um=500.0, n=1.46,
    )
    assert r.summary["beat_from_angle_um"] == pytest.approx(420.0, rel=0.05)
    assert next(c for c in r.checks if c.name == "beat visible").passes


def test_coarse_gratings_fail_the_invisibility_gate():
    r = moire_readability(
        face="x", back_pitch_um=200.0, front_pitch_um=220.0,
        angle_offset_deg=3.0, thickness_um=500.0, n=1.46,
    )
    assert not next(c for c in r.checks if c.name == "gratings invisible").passes


# --- rollup -----------------------------------------------------------------


def test_verdict_collects_failures():
    good = switch_readability(face="a", barrier_pitch_um=60.0, thickness_um=500.0, n=1.46)
    bad = switch_readability(face="b", barrier_pitch_um=400.0, thickness_um=1500.0, n=1.52)
    v = verdict([good, bad])
    assert v["passes"] is False
    assert v["n_faces"] == 2
    assert any(f["face"] == "b" for f in v["failures"])
    assert all(f["face"] != "a" for f in v["failures"])


def test_verdict_passes_when_everything_passes():
    v = verdict([switch_readability(face="a", barrier_pitch_um=60.0, thickness_um=500.0, n=1.46)])
    assert v["passes"] is True
    assert v["failures"] == []


# --- combined beat: the vector formula, not min() ---------------------------


def test_combined_beat_reduces_to_both_closed_forms():
    from app.readability import beat_period_combined_um

    # angle 0 -> pure pitch beat
    assert beat_period_combined_um(22.0, 23.98, 0.0) == pytest.approx(
        beat_period_um(22.0, 23.98), rel=1e-9
    )
    # equal pitch -> pure angle beat
    assert beat_period_combined_um(22.0, 22.0, 3.0) == pytest.approx(
        beat_period_angle_um(22.0, 3.0), rel=1e-9
    )


def test_combined_beat_is_finer_than_either_alone():
    """The old min(pitch, angle) shortcut OVER-reported: the contributions add
    as vectors, so the true beat is finer than either component."""
    from app.readability import beat_period_combined_um

    combined = beat_period_combined_um(22.0, 23.98, 3.0)
    assert combined < beat_period_um(22.0, 23.98)
    assert combined < beat_period_angle_um(22.0, 3.0)
    assert combined == pytest.approx(227.7, abs=1.0)


def test_per_bucket_fan_catches_a_sub_acuity_leaf_species():
    """The shipping frame fans the louvre +/-2.5 buckets x 3.5 deg on top of a
    3 deg base, so the outer buckets cross at ~11.75 deg and beat far finer than
    the base direction. Gating on the base alone hides that."""
    base, span, count = 3.0, 3.5, 6
    offsets = [base + (b - (count - 1) / 2.0) * span for b in range(count)]

    base_only = moire_readability(
        face="top", back_pitch_um=22.0, front_pitch_um=23.98,
        angle_offset_deg=base, thickness_um=500.0, n=1.46,
    )
    fanned = moire_readability(
        face="top", back_pitch_um=22.0, front_pitch_um=23.98,
        angle_offset_deg=base, thickness_um=500.0, n=1.46,
        angle_offsets_deg=offsets,
    )
    # The base direction looks fine...
    assert next(c for c in base_only.checks if c.name == "beat visible").passes
    # ...but the worst fabricated direction does not.
    assert not next(c for c in fanned.checks if c.name == "beat visible").passes
    assert fanned.summary["worst_offset_deg"] == pytest.approx(11.75, abs=0.01)
    assert len(fanned.summary["per_offset"]) == count
    assert fanned.summary["beat_um"] < base_only.summary["beat_um"]


def _fan(base: float, span: float, count: int = 6) -> list[float]:
    return [base + (b - (count - 1) / 2.0) * span for b in range(count)]


def _worst_beat_passes(base: float, span: float) -> bool:
    r = moire_readability(
        face="top", back_pitch_um=22.0, front_pitch_um=23.98,
        angle_offset_deg=base, thickness_um=500.0, n=1.46,
        angle_offsets_deg=_fan(base, span),
    )
    return next(c for c in r.checks if c.name == "beat visible").passes


def test_narrowing_the_fan_alone_does_not_fix_it():
    """Halving the span while KEEPING the 3 deg base still leaves the outer
    bucket at ~1.6 arcmin. The base offset is added on top of the fan, so it is
    what pushes the extreme direction out — narrowing alone is not enough."""
    assert not _worst_beat_passes(3.0, 3.5)   # shipping: worst 1.18 arcmin
    assert not _worst_beat_passes(3.0, 2.0)   # narrowed only: worst 1.60


def test_centring_the_fan_on_zero_crossing_is_what_fixes_it():
    """Because the beat depends on |crossing angle|, centring the fan on zero
    minimises the worst direction. base 0 / span 2.0 clears every bucket while
    keeping 10 deg of orientation spread — twice what base 3 can afford."""
    assert _worst_beat_passes(0.0, 2.0)
    assert _worst_beat_passes(0.0, 1.5)
    # ...and the widest spread that still passes is 10 deg, not 17.5.
    assert not _worst_beat_passes(0.0, 2.5)
