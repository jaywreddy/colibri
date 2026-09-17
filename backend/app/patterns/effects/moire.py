"""Moire / diffraction PHYSICS: the closed forms the design is numbered on.

All lengths in micrometres (um), all angles in DEGREES at the public API (the
internal trig converts to radians).

  1. MOIRE  — two gratings superpose and beat at a period far coarser than
     either. On the box this is in-plane (the garland's leaf gratings) or, on
     the hidden two-ply exemplar, across the slab.
     :func:`beat_period_parallel` and :func:`beat_period_rotated` are the two
     closed forms; ``witness_geom.combined_beat_um`` is the VECTOR form they
     are both special cases of, and ``tests/test_optics_math.py`` pins that.

  2. DIFFRACTION — a grating with period below ~5 um throws first-order rainbow
     fans at accessible angles, so the metal flashes spectral colour as it
     tilts. That is what every written centrepiece on the box now uses (a
     region_art period map, a photo colour zone, a leaf family), which is why
     :func:`diffraction_onset` is the part of this module the pipeline reads —
     through ``gratings.py``, which also takes its litho floor from
     ``MIN_PERIOD_UM`` here so the two cannot drift.

The PARALLAX half of this module — ``parallax_shift_um``, the phase-switch tilt
solvers, ``scanimation_plan``, ``moire_magnification`` — went on 2026-09-16 with
the two-ply optics. The parallax numbers the box still needs are computed in
``witness_geom`` (``parallax_um_per_deg``, ``swap_deg``) against the box's OWN
2.25 mm quartz rather than against the 500 um design substrate below.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --- substrate constants (the plate the client fabs on) ---------------------
SUBSTRATE_T_UM = 500.0   # fused-silica thickness between the two gold faces
SUBSTRATE_N = 1.46       # fused-silica index at ~550 nm

# Litho floor: min gold line 2 µm + min gap 2 µm ⇒ min grating PERIOD 4 µm.
MIN_PERIOD_UM = 4.0
# Below ~5 µm period, first-order diffraction of visible white light throws a
# strongly visible rainbow at accessible tilt angles — treat as a designed
# effect, not an accident. (Onset is wavelength/period dependent; see
# diffraction_onset for the exact per-wavelength angle.)
DIFFRACTION_PERIOD_UM = 5.0

# Visible band used for diffraction onset (µm). 0.40 violet … 0.70 red.
VISIBLE_MIN_UM = 0.40
VISIBLE_MAX_UM = 0.70


def beat_period_parallel(period_a_um: float, period_b_um: float) -> float:
    """Moiré BEAT period (µm) of two PARALLEL line gratings with slightly
    different periods p_a, p_b.

    The superposition of two combs beats at the difference frequency:

        1/P_beat = |1/p_a − 1/p_b|   ⇒   P_beat = p_a·p_b / |p_a − p_b|.

    Equal periods → infinite beat (no moiré). A 6 % period mismatch on a 22 µm
    carrier (22 vs 23.3) beats at ~360 µm — coarse, slow, clearly visible.
    Returns ``inf`` when the periods are equal.
    """
    d = abs(period_a_um - period_b_um)
    if d < 1e-12:
        return math.inf
    return period_a_um * period_b_um / d


def beat_period_rotated(period_um: float, angle_offset_deg: float) -> float:
    """Moiré beat period (µm) of two EQUAL-period gratings rotated by a small
    relative angle α.

    Rotating one comb by α makes the fringe spacing

        P_beat = p / (2·sin(α/2))    ≈ p/α   (α in radians, small angle).

    α = 3° on a 22 µm carrier → ~420 µm fringes running nearly perpendicular to
    the lines. α → 0 → infinite (no moiré). This is the pure-rotation companion
    to :func:`beat_period_parallel`; real designs combine both.
    """
    a = abs(math.radians(angle_offset_deg))
    if a < 1e-9:
        return math.inf
    return period_um / (2.0 * math.sin(a / 2.0))


@dataclass
class Diffraction:
    """Result of :func:`diffraction_onset` — first-order rainbow behaviour."""

    period_um: float
    diffracts_visible: bool          # any visible order at accessible tilt?
    first_order_deg_red: float       # +1 order angle for 0.70 µm (deg, inf=evanescent)
    first_order_deg_violet: float    # +1 order angle for 0.40 µm
    spectral_spread_deg: float       # violet→red fan width of the +1 order
    note: str


def _first_order_angle_deg(period_um: float, wavelength_um: float) -> float:
    """First-order (m=1) diffraction angle for normal-incidence white light off
    a grating of period ``period_um``: sin θ = λ / d. Returns ``inf`` if the
    order is evanescent (λ > d, no propagating first order).
    """
    s = wavelength_um / period_um
    if s >= 1.0:
        return math.inf
    return math.degrees(math.asin(s))


def diffraction_onset(period_um: float) -> Diffraction:
    """Where a single grating starts throwing a visible RAINBOW.

    At normal incidence the m=1 order for wavelength λ leaves at
    sin θ = λ/d. Across the visible band (0.40–0.70 µm) that order fans out into
    a spectrum; the fan is "visible" when the whole band propagates (d > 0.70 µm
    trivially, but the flash is only *strong and eye-catching* once the fan sits
    at large, easily-hit angles — that happens as d drops toward ~5 µm and
    below). We report the exact per-wavelength angles plus a design flag.

    Design guidance embedded in ``note``:
      * d ≳ 20 µm  → first order hugs the surface (θ ≈ 1–2°); no perceptible
        colour — pure moiré carrier, this is what you want for shimmer.
      * d ≈ 5–8 µm → first order at ~4–8°; a subtle iridescent sheen begins.
      * d ≲ 5 µm   → strong rainbow flash at 5–10°; a designed diffraction
        accent. At the 4 µm litho floor the fan is widest and brightest.
    """
    red = _first_order_angle_deg(period_um, VISIBLE_MAX_UM)
    violet = _first_order_angle_deg(period_um, VISIBLE_MIN_UM)
    spread = (red - violet) if math.isfinite(red) and math.isfinite(violet) else math.inf
    diffracts = period_um < DIFFRACTION_PERIOD_UM * 4.0  # first order within ~40°
    if period_um <= DIFFRACTION_PERIOD_UM:
        note = (f"STRONG rainbow flash — first order at {violet:.0f}–{red:.0f}°, "
                f"{spread:.0f}° spectral fan. Designed diffraction accent.")
    elif period_um <= 8.0:
        note = (f"subtle iridescence — first order at {violet:.0f}–{red:.0f}°; "
                "sheen begins")
    elif period_um <= 20.0:
        note = (f"faint colour near grazing — first order at {violet:.0f}–{red:.0f}°")
    else:
        note = (f"no visible colour — first order hugs surface "
                f"({violet:.1f}–{red:.1f}°); clean moiré carrier")
    return Diffraction(period_um, diffracts, red, violet, spread, note)
