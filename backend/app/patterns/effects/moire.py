"""Moiré / parallax / diffraction PHYSICS for the two-layer gold-on-silica stack.

All lengths in micrometres (µm), all angles in DEGREES at the public API (the
internal trig converts to radians). The stack is two gold gratings written on
opposite faces of a single fused-silica plate:

    front gold ── air ──┐
                        │  fused silica  t = 500 µm, n = 1.46
    back  gold ─────────┘

The viewer looks through the front grating at the back grating. Two mechanisms
produce the effects:

  1. MOIRÉ  — the two gratings' line patterns superpose; where lines coincide
     the stack is dark-dense, where they fall between each other it is open.
     The superposition beats at a much coarser BEAT period than either grating,
     which is what the eye reads as slow shimmer / ghost fringes.

  2. PARALLAX — tilting the plate slides the apparent position of the back
     grating relative to the front one, because the line of sight refracts into
     the glass (Snell) and walks a lateral distance across the 500 µm gap. That
     lateral walk is what ANIMATES the moiré (scanimation) and biases which
     phase of a phase-switch pair the eye samples.

A third, single-layer mechanism matters at the fine end:

  3. DIFFRACTION — a grating with period below ~5 µm diffracts VISIBLE white
     light into first-order rainbow fans at large angles; below the period the
     stack flashes spectral colour as it tilts. Designable as an accent.
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


def parallax_shift_um(
    tilt_deg: float,
    t_um: float = SUBSTRATE_T_UM,
    n: float = SUBSTRATE_N,
) -> float:
    """Lateral walk (µm) of the back-face image relative to the front face when
    the plate is tilted ``tilt_deg`` from normal, viewing THROUGH the glass.

    A ray arriving at incidence θ (in air) refracts to θ' inside the glass with
    sin θ' = sin θ / n (Snell). Over thickness ``t`` it walks laterally

        dx = t · tan(θ') = t · tan(asin(sin θ / n)).

    For small θ this is ≈ t·θ/n per radian → for t=500, n=1.46 that is
    ≈ 5.9 µm per DEGREE of tilt (the rule quoted in the fab brief). We evaluate
    the exact expression so it stays accurate out to the ±15–20° a hand-held
    piece actually sees.
    """
    th = math.radians(tilt_deg)
    s = math.sin(th) / n
    s = max(-1.0, min(1.0, s))  # guard rounding past ±1 at grazing angles
    theta_in = math.asin(s)
    return t_um * math.tan(theta_in)


# Convenience: the linearized 5.9 µm/deg slope quoted in the brief (small-angle
# limit of parallax_shift_um at t=500, n=1.46). Exposed for quick estimates.
PARALLAX_UM_PER_DEG = SUBSTRATE_T_UM / SUBSTRATE_N * math.radians(1.0)  # ≈ 5.978


def tilt_for_parallax_deg(
    dx_um: float,
    t_um: float = SUBSTRATE_T_UM,
    n: float = SUBSTRATE_N,
) -> float:
    """Inverse of :func:`parallax_shift_um` — tilt (deg) that walks the back
    image by ``dx_um`` laterally. Solve dx = t·tan(θ'), θ' inside glass, then
    refract back out: sin θ = n · sin θ'.
    """
    theta_in = math.atan2(dx_um, t_um)
    s = n * math.sin(theta_in)
    s = max(-1.0, min(1.0, s))
    return math.degrees(math.asin(s))


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


def beat_angle_deg(period_um: float, angle_offset_deg: float) -> float:
    """Orientation (deg) of the rotated-grating moiré fringes.

    For two equal-period gratings offset by α, the fringes bisect the two line
    directions and run nearly PERPENDICULAR to the lines: fringe normal sits at
    α/2, so the fringes themselves lie at 90° + α/2 from the base line
    direction. Reported relative to the base grating's LINES.
    """
    return 90.0 + angle_offset_deg / 2.0


def moire_magnification(period_sample_um: float, period_reveal_um: float) -> float:
    """Moiré MAGNIFICATION factor for a "revealing-layer" pair (the effect that
    turns a tiny repeated motif into a large floating ghost).

    A sampling grating of period ``p_s`` laid over an image grating of period
    ``p_r`` (the motif pitch) produces a magnified copy of the motif at

        M = p_r / (p_r − p_s)          (signed: negative ⇒ inverted ghost),
        P_ghost = |M| · p_r = p_s·p_r / |p_r − p_s|.

    The magnified image's PITCH equals the parallel-beat period, and each motif
    cell is blown up by |M|. A 20 µm motif sampled at 20.4 µm → M = −50 → the
    motif appears 50× larger (1 mm) and inverted. Big |M| needs a very small
    mismatch and is correspondingly twitchy; keep |M| ≲ 30 for a stable ghost.

    Returns ``inf`` when the periods match (degenerate, no ghost).
    """
    d = period_reveal_um - period_sample_um
    if abs(d) < 1e-12:
        return math.inf
    return period_reveal_um / d


@dataclass
class PhaseSwitch:
    """Result of :func:`phase_switch_tilt_deg` — a two-image tilt switch."""

    period_um: float
    half_switch_tilt_deg: float  # tilt to walk parallax by HALF a period (A→B)
    full_cycle_tilt_deg: float   # tilt to walk a FULL period (B→back to A)
    comfortable: bool            # half-switch lands in the 3–8° sweet spot?
    note: str


def phase_switch_tilt_deg(
    period_um: float,
    t_um: float = SUBSTRATE_T_UM,
    n: float = SUBSTRATE_N,
) -> PhaseSwitch:
    """Tilt (deg) that switches a two-phase interlaced image A↔B.

    The centerpiece packs image A and image B into alternating half-period
    stripes behind a matched front grating. Head-on, the front lines mask one
    phase; tilting walks the back stripes by parallax, and after HALF a period
    of walk the front lines mask the OTHER phase — the image switches. A FULL
    period of walk returns to A (one animation cycle).

    Using the exact parallax law, we solve for the tilt that gives
    dx = period/2 (switch) and dx = period (cycle). We want the half-switch to
    land in a comfortable ~3–8° hand tilt: too small → twitchy (switches with
    the tiniest wobble), too large → you have to crank the piece over.
    """
    half = tilt_for_parallax_deg(period_um / 2.0, t_um, n)
    full = tilt_for_parallax_deg(period_um, t_um, n)
    comfortable = 3.0 <= half <= 8.0
    if half < 3.0:
        note = (f"twitchy — switches at only {half:.1f}° (want 3–8°); "
                "use a coarser period")
    elif half > 8.0:
        note = (f"stiff — needs {half:.1f}° tilt to switch (want 3–8°); "
                "use a finer period")
    else:
        note = f"comfortable — clean switch at {half:.1f}° tilt"
    return PhaseSwitch(period_um, half, full, comfortable, note)


def period_for_switch_tilt_um(
    target_half_switch_deg: float,
    t_um: float = SUBSTRATE_T_UM,
    n: float = SUBSTRATE_N,
) -> float:
    """Inverse design: the interlace period whose A→B half-switch happens at
    ``target_half_switch_deg``. period = 2 · parallax_shift(target).
    """
    return 2.0 * parallax_shift_um(target_half_switch_deg, t_um, n)


@dataclass
class Scanimation:
    """Result of :func:`scanimation_plan` — a barrier-grid animation recipe."""

    n_phases: int
    frame_pitch_um: float        # period of the back interlaced image
    barrier_period_um: float     # front slit-grating period (== frame_pitch)
    slit_duty: float             # open fraction of the front barrier
    tilt_per_frame_deg: float    # tilt to advance ONE frame
    tilt_full_cycle_deg: float   # tilt to run all n_phases and wrap
    note: str


def scanimation_plan(
    n_phases: int,
    frame_pitch_um: float,
    t_um: float = SUBSTRATE_T_UM,
    n: float = SUBSTRATE_N,
) -> Scanimation:
    """Barrier-grid scanimation (a.k.a. "scanimation"/kinegram) plan.

    The BACK layer interleaves ``n_phases`` frames of an animation: within each
    ``frame_pitch_um`` cell, phase k occupies the k-th 1/n slot. The FRONT layer
    is a SLIT grating of the SAME period with an open duty ≈ 1/n, so at any tilt
    it reveals exactly one phase and masks the rest. Tilting walks the slit over
    the back by parallax; each 1/n-period of walk advances ONE frame, so the
    image ANIMATES smoothly as the piece rocks.

    - tilt_per_frame = tilt that walks parallax by frame_pitch/n.
    - full cycle     = tilt that walks a whole frame_pitch (all n phases, wrap).
    - slit_duty      = 1/n (each slot shows one phase; slightly under 1/n keeps
      crisp frames, slightly over blends adjacent frames for motion blur).

    Fewer phases (3–4) → each frame gets a wide 1/n slot (crisp, bright) and a
    larger per-frame tilt (less twitchy). Many phases → smoother motion but dim,
    twitchy, and each slot narrows toward the 2 µm litho floor.
    """
    n_phases = max(2, int(n_phases))
    slot = frame_pitch_um / n_phases
    per_frame = tilt_for_parallax_deg(slot, t_um, n)
    full = tilt_for_parallax_deg(frame_pitch_um, t_um, n)
    slit_duty = 1.0 / n_phases
    min_line = slot * slit_duty  # narrowest gold feature the interlace writes
    if min_line < MIN_PERIOD_UM / 2.0:
        note = (f"slot gold feature {min_line:.1f} µm < 2 µm litho floor — "
                "increase frame_pitch or reduce n_phases")
    elif per_frame < 1.5:
        note = (f"twitchy — {per_frame:.1f}°/frame; coarsen frame_pitch")
    else:
        note = (f"ok — {per_frame:.1f}° advances one of {n_phases} frames, "
                f"{full:.1f}° runs the full cycle")
    return Scanimation(
        n_phases=n_phases,
        frame_pitch_um=frame_pitch_um,
        barrier_period_um=frame_pitch_um,
        slit_duty=slit_duty,
        tilt_per_frame_deg=per_frame,
        tilt_full_cycle_deg=full,
        note=note,
    )


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
