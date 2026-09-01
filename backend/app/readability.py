"""Will the fabricated face actually READ to a human eye?

The renderer answers "what does this look like on screen"; this module answers
the question that matters, which is what a person holding the finished box
perceives. Those are not the same question, and the preview is systematically
WORSE than the part: the two-plane renderer filters each layer independently
and then composites, computing <front>*<back>, while physical light (and the
eye) integrate the product <front*back>. The correlation between the layers IS
the effect, so once the structure falls below a screen pixel the preview loses
exactly the term the real object keeps. Everything here is therefore computed
from the FABRICATED geometry, never measured off the render.

The perceptual budget, per effect face:

  ACUITY. The micro-structure must stay INVISIBLE (else the eye sees stripes
  instead of an image) while the effect it produces must be VISIBLE. Normal
  acuity resolves about 1 arcmin; we want the structure comfortably under that
  and the effect comfortably over it.

  PUPIL INTEGRATION. The eye's pupil spans an angular range pupil/distance, so
  it averages the barrier over that cone. That blur must be small against the
  angular separation between the two switch states, or A and B wash together.

  DEFOCUS. The eye focuses on one layer; the other sits an optical gap behind
  and blurs by gap*pupil/distance. That must be small against the lane pitch.

  DIFFRACTION. A slit of width w spreads light by ~lambda/w. On a barrier that
  blurs the angular selectivity, so it is charged against the same separation.

Angles come from the same Snell relation the simulator and the renderer use
(see sim2d.tilt_for_shift_um): a view tilted by theta shifts the back layer by
t*sin(theta)/n / cos(asin(sin(theta)/n)).

Unit-tested against closed forms in tests/test_readability.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Normal (6/6) acuity resolves ~1 arcmin. Structure should sit well under it to
# stay invisible; an effect should sit well over it to be seen.
ACUITY_ARCMIN = 1.0
STRUCTURE_INVISIBLE_ARCMIN = 0.7     # below this, the lattice reads as smooth
EFFECT_VISIBLE_ARCMIN = 2.0          # above this, the fringe/feature is legible

# Michelson-ish contrast the eye reliably notices on a large smooth feature.
CONTRAST_FLOOR = 0.02                # 2%

# Blur budgets, as a fraction of the quantity they degrade.
BLUR_BUDGET = 0.35                   # pupil + diffraction vs switch separation
DEFOCUS_BUDGET = 0.35                # defocus blur vs lane pitch

# Default viewing conditions: a held object at reading distance, indoor pupil.
DEFAULT_DISTANCE_MM = 300.0
DEFAULT_PUPIL_MM = 3.0
PHOTOPIC_LAMBDA_UM = 0.55


def subtense_arcmin(feature_um: float, distance_mm: float = DEFAULT_DISTANCE_MM) -> float:
    """Angular size of a feature (um) at a viewing distance (mm), in arcmin."""
    if distance_mm <= 0:
        raise ValueError(f"distance must be positive (got {distance_mm})")
    return math.degrees((feature_um / 1000.0) / distance_mm) * 60.0


def tilt_for_shift_deg(shift_um: float, thickness_um: float, n: float) -> float:
    """Exterior tilt (deg) that walks the back layer sideways by ``shift_um``.

    Same Snell relation as ``sim2d.tilt_for_shift_um``; returns NaN past the
    grazing-exit limit where no exterior angle produces the shift.
    """
    if thickness_um <= 0:
        raise ValueError(f"thickness must be positive (got {thickness_um})")
    v = n * math.sin(math.atan2(abs(shift_um), thickness_um))
    if v >= 1.0:
        return float("nan")
    return math.degrees(math.asin(v))


def beat_period_um(p1_um: float, p2_um: float) -> float:
    """Moiré beat period of two gratings differing only in pitch."""
    d = abs(p2_um - p1_um)
    if d < 1e-12:
        return float("inf")
    return p1_um * p2_um / d


def beat_period_angle_um(pitch_um: float, angle_deg: float) -> float:
    """Moiré beat period of two EQUAL gratings crossed by a small angle."""
    a = abs(math.radians(angle_deg))
    if a < 1e-12:
        return float("inf")
    return pitch_um / (2.0 * math.sin(a / 2.0))


def beat_period_combined_um(p1_um: float, p2_um: float, angle_deg: float) -> float:
    """Beat period of two gratings differing in BOTH pitch and orientation.

    The moiré of two periodic lattices is the difference of their spatial
    frequency VECTORS, so with k1 = (1/p1, 0) and k2 = (cos/p2, sin/p2):

        beat = 1 / |k1 - k2|

    Taking min(pitch-beat, angle-beat) instead — as this module first did — is
    not the physics: the two contributions add as vectors, not as alternatives.
    That shortcut OVER-reports the beat by up to ~38% on the fanned-out leaf
    buckets, and it always errs toward "looks fine", which is the wrong
    direction for a gate whose whole job is to catch a shimmer nobody will see.
    Reduces exactly to the two closed forms above at angle 0 and equal pitch.
    """
    if p1_um <= 0 or p2_um <= 0:
        raise ValueError("periods must be positive")
    a = math.radians(angle_deg)
    dx = 1.0 / p1_um - math.cos(a) / p2_um
    dy = math.sin(a) / p2_um
    d = math.hypot(dx, dy)
    return float("inf") if d < 1e-12 else 1.0 / d


@dataclass
class Check:
    """One perceptual gate: what it measures, the value, and whether it passes."""

    name: str
    value: float
    unit: str
    limit: float
    passes: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4) if math.isfinite(self.value) else None,
            "unit": self.unit,
            "limit": round(self.limit, 4),
            "passes": self.passes,
            "note": self.note,
        }


@dataclass
class FaceReadability:
    face: str
    kind: str
    checks: list[Check] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        return all(c.passes for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "face": self.face,
            "kind": self.kind,
            "passes": self.passes,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
        }


def switch_readability(
    *,
    face: str,
    barrier_pitch_um: float,
    thickness_um: float,
    n: float,
    duty: float = 0.5,
    distance_mm: float = DEFAULT_DISTANCE_MM,
    pupil_mm: float = DEFAULT_PUPIL_MM,
) -> FaceReadability:
    """Perceptual budget for a parallax-barrier image switch (A/B or scanimation).

    The lanes must be invisible, the swap must land at a comfortable hand tilt,
    and pupil/diffraction blur must stay small against the angular separation
    between the two states.
    """
    lane_um = barrier_pitch_um * duty
    gap_um = thickness_um / n

    swap_deg = tilt_for_shift_deg(barrier_pitch_um / 4.0, thickness_um, n)
    sep_deg = tilt_for_shift_deg(barrier_pitch_um / 2.0, thickness_um, n)
    alias_deg = tilt_for_shift_deg(barrier_pitch_um, thickness_um, n)

    lane_arcmin = subtense_arcmin(lane_um, distance_mm)
    pupil_deg = math.degrees(pupil_mm / distance_mm)
    diffraction_deg = math.degrees(PHOTOPIC_LAMBDA_UM / lane_um)
    defocus_um = gap_um * pupil_mm / distance_mm

    checks = [
        Check(
            "lane invisible", lane_arcmin, "arcmin", STRUCTURE_INVISIBLE_ARCMIN,
            lane_arcmin <= STRUCTURE_INVISIBLE_ARCMIN,
            "barrier lanes must not resolve, or the eye sees stripes instead of an image",
        ),
        Check(
            "swap at hand tilt", swap_deg, "deg", 8.0,
            2.0 <= swap_deg <= 8.0 if math.isfinite(swap_deg) else False,
            "the A/B swap should land in a comfortable 2-8 deg wrist tilt",
        ),
        Check(
            "pupil blur vs separation", pupil_deg / sep_deg if sep_deg else float("inf"),
            "fraction", BLUR_BUDGET,
            (pupil_deg / sep_deg) <= BLUR_BUDGET if sep_deg else False,
            "the eye averages the barrier over its pupil; too much and A/B blend",
        ),
        Check(
            "diffraction vs separation",
            diffraction_deg / sep_deg if sep_deg else float("inf"), "fraction", BLUR_BUDGET,
            (diffraction_deg / sep_deg) <= BLUR_BUDGET if sep_deg else False,
            "slit diffraction softens the barrier's angular selectivity",
        ),
        Check(
            "defocus vs lane", defocus_um / lane_um if lane_um else float("inf"),
            "fraction", DEFOCUS_BUDGET,
            (defocus_um / lane_um) <= DEFOCUS_BUDGET if lane_um else False,
            "the un-focused layer blurs by gap*pupil/distance",
        ),
    ]
    return FaceReadability(
        face=face,
        kind="switch",
        checks=checks,
        summary={
            "barrier_pitch_um": barrier_pitch_um,
            "lane_um": lane_um,
            "optical_gap_um": round(gap_um, 1),
            "swap_deg": round(swap_deg, 2) if math.isfinite(swap_deg) else None,
            "separation_deg": round(sep_deg, 2) if math.isfinite(sep_deg) else None,
            "alias_deg": round(alias_deg, 2) if math.isfinite(alias_deg) else None,
            "lane_arcmin": round(lane_arcmin, 2),
        },
    )


def moire_readability(
    *,
    face: str,
    back_pitch_um: float,
    front_pitch_um: float,
    angle_offset_deg: float,
    thickness_um: float,
    n: float,
    distance_mm: float = DEFAULT_DISTANCE_MM,
    pupil_mm: float = DEFAULT_PUPIL_MM,
    angle_offsets_deg: list[float] | None = None,
) -> FaceReadability:
    """Perceptual budget for a two-grating moiré (the frame foliage shimmer).

    Opposite requirement to the switch: the GRATINGS must be invisible while
    their BEAT must be clearly visible, and the fringes have to travel a useful
    distance per degree of tilt or the shimmer reads as static texture.
    """
    gap_um = thickness_um / n
    beat_pitch = beat_period_um(back_pitch_um, front_pitch_um)
    beat_angle = beat_period_angle_um(back_pitch_um, angle_offset_deg)
    # PER-BUCKET evaluation. The frame encodes a different louvre ORIENTATION
    # per motif species, so one face carries a fan of crossing angles, not one.
    # Gating on the base offset alone reports "passes" while individual leaf
    # species sit below acuity — the gate must judge the WORST direction it
    # actually fabricates.
    offsets = list(angle_offsets_deg) if angle_offsets_deg else [angle_offset_deg]
    per_offset = [
        (off, beat_period_combined_um(back_pitch_um, front_pitch_um, off)) for off in offsets
    ]
    beat = min(b for _, b in per_offset)
    worst_offset = min(per_offset, key=lambda x: x[1])[0]

    grating_arcmin = subtense_arcmin(max(back_pitch_um, front_pitch_um), distance_mm)
    beat_arcmin = subtense_arcmin(beat, distance_mm)

    # Fringe travel: a back-layer shift dx moves the beat by dx * (beat/pitch).
    shift_per_deg = (
        thickness_um * math.sin(math.radians(1.0)) / n
    )  # small-angle back-layer walk per degree
    fringe_travel_per_deg = shift_per_deg * (beat / back_pitch_um)
    travel_frac = fringe_travel_per_deg / beat if beat else 0.0

    checks = [
        Check(
            "gratings invisible", grating_arcmin, "arcmin", STRUCTURE_INVISIBLE_ARCMIN,
            grating_arcmin <= STRUCTURE_INVISIBLE_ARCMIN,
            "the carrier/louvre themselves must not resolve",
        ),
        Check(
            "beat visible", beat_arcmin, "arcmin", EFFECT_VISIBLE_ARCMIN,
            beat_arcmin >= EFFECT_VISIBLE_ARCMIN,
            "the moire fringes must be comfortably above the acuity limit",
        ),
        Check(
            "fringe travel per deg", travel_frac, "beats/deg", 0.02,
            travel_frac >= 0.02,
            "fringes must move a visible fraction of a beat per degree of tilt",
        ),
    ]
    return FaceReadability(
        face=face,
        kind="moire",
        checks=checks,
        summary={
            "back_pitch_um": back_pitch_um,
            "front_pitch_um": front_pitch_um,
            "angle_offset_deg": angle_offset_deg,
            "optical_gap_um": round(gap_um, 1),
            "beat_from_pitch_um": round(beat_pitch, 1) if math.isfinite(beat_pitch) else None,
            "beat_from_angle_um": round(beat_angle, 1) if math.isfinite(beat_angle) else None,
            "beat_um": round(beat, 1) if math.isfinite(beat) else None,
            "worst_offset_deg": round(worst_offset, 2),
            "per_offset": [
                {
                    "offset_deg": round(o, 2),
                    "beat_um": round(b, 1) if math.isfinite(b) else None,
                    "beat_arcmin": round(subtense_arcmin(b, distance_mm), 2)
                    if math.isfinite(b)
                    else None,
                }
                for o, b in per_offset
            ],
            "beat_arcmin": round(beat_arcmin, 2),
            "grating_arcmin": round(grating_arcmin, 3),
            "fringe_travel_um_per_deg": round(fringe_travel_per_deg, 1),
        },
    )


def verdict(reports: list[FaceReadability]) -> dict[str, Any]:
    """Roll up per-face reports into one pass/fail with the failing reasons."""
    failing = [
        {"face": r.face, "check": c.name, "value": c.value, "limit": c.limit, "note": c.note}
        for r in reports
        for c in r.checks
        if not c.passes
    ]
    return {
        "passes": not failing,
        "n_faces": len(reports),
        "failures": failing,
        "faces": [r.to_dict() for r in reports],
    }
