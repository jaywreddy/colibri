from __future__ import annotations

"""Capybara + water SCANIMATION (barrier-grid animation) — BACK face.

A serene capybara sits half-submerged on a waterline; rocking the box makes the
WATER RIPPLES FLOW around it. The mechanism is a classic barrier-grid
scanimation (a.k.a. kinegram) realised in the two-layer gold-on-silica stack:

  FRONT layer (uFront):
    * the capybara silhouette, filled with a fine stripe carrier so the animal
      itself shimmers as metal (it does NOT move — it is the still subject);
    * a SLIT BARRIER (open duty ≈ 1/N) covering the WATER BAND below/around the
      waterline. The barrier period == the back frame pitch.

  BACK layer (uBack + phase_0..phase_{N-1} extra layers):
    * N interleaved ripple frames. Within each frame-pitch cell the k-th 1/N
      slot holds ripple phase k. Head-on, the barrier reveals exactly one slot
      (one phase); the rest hide behind the bars.

Tilting the plate walks the barrier across the back frames by Snell parallax
(≈5.9 µm/deg through 500 µm fused silica). Each 1/N period of walk advances the
visible ripple by ONE phase, so the crests travel horizontally around the
capybara as the box rocks — the water appears to flow.

FLOW FIELD (the ripple frames). Rather than horizontal bands that sway in y
(which read as "bands stepping"), each frame is a field of long, undulating
STREAMLINES — ridges running ALONG the flow (x) axis whose transverse
undulation is a TRAVELLING WAVE: the k-th frame advances the undulation phase by
2π·k/N, so across the N frames the crests march one wavelength downstream in ONE
consistent direction. Under the walking barrier this reads as a continuous
directional CURRENT, and even a small head wobble animates it. A wake opens
around the half-submerged body: streamlines part in y around it (``_WAKE_PART``),
the undulation is shoved downstream behind it into a trailing tongue/V
(``_WAKE_SHOVE``), and a calm elliptical patch sits right under the belly
(``_flow_amplitude``). Depth shear rakes the deeper streamlines so the current
reads as layered, not stacked bars. The exact field lives in
``_flow_streamline_field`` / ``_flow_amplitude`` and is mirrored 1:1 by the
preview shader (``plate.frag::flowStreamlineField``/``flowAmplitude``) — the two
copies MUST stay in lock-step. The same functions feed the fab SVG bake (via
``_interleave_phases`` in ``plates.py::ensure_plate_svg``), so front.svg/back.svg
carry the identical phase ramp.

Physics comes straight from ``effects/moire.scanimation_plan`` (slit period vs
substrate, comfortable per-frame tilt) and the ripple/carrier gratings are baked
with ``effects/gratings`` conventions (center origin, y-up, 4 µm litho floor).

Design defaults: frame pitch 60 µm, N = 4 phases → slit OPEN slot 15 µm, slit
BARRIER BAR 45 µm (= frame_pitch·(1−duty) with duty 1/N = 0.25), ≈2.5°/frame,
full cycle over ≈10° of tilt — the whole animation reads within a natural hand
rock (2–6°). The CONTROLLED minimum features are therefore the 15 µm open slot
and the 45 µm barrier bar; the interleaved back frames are also snapped to the
15 µm slot grid and the ripple ridges are floored to the 2 µm litho minimum in
thickness (see ``_floor_crest_thickness`` and ``_interleave_phases``), so the
baked geometry contains NOTHING between 0 and 2 µm — no uncontrolled slivers.
(The old note claimed a 11.25 µm minimum from a slot·(1−duty) guess; that number
was fictitious — it is neither the slot, the bar, nor the floor. The manifest
now reports ``min_*_gold_um`` MEASURED from the emitted masks.)
"""

import numpy as np

from .._helpers import MAX_LATTICE_CELLS, check_lattice_budget, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, ensure_multipolygon, register
from ..effects import moire
from ..motifs.lab import capybara


# Waterline (normalized art-box y, y-down like Pillow). The capybara belly sits
# near y≈0.82; putting the waterline at 0.66 half-submerges the lower barrel and
# legs while the head/back stay dry — a serene "wading" read.
WATERLINE_Y = 0.66


def _capybara_and_water(
    n: int,
    waterline_y: float = WATERLINE_Y,
    n_cols: int | None = None,
    col_off: int = 0,
):
    """Rasterize the scene geometry at grid height ``n`` (rows), width ``n_cols``.

    Default (``n_cols=None``) is the square ``n×n`` art tile. For WATER FULL WIDTH
    the plate compositor passes a wider ``n_cols`` (the full aperture width) with
    the body square inset at column ``col_off``: the capybara stays in the centered
    ``n×n`` square while the water band spans the whole width — open current flanks
    the animal. Returns bool masks in Pillow/y-down grid order:
      * ``capy``       — full capybara silhouette (both above & below water).
      * ``capy_above`` — capybara ABOVE the waterline (the dry, shimmering body).
      * ``capy_below`` — capybara BELOW the waterline (submerged; hinted only).
      * ``water_band`` — the water region (below waterline, spanning ``n_cols``).
    """
    n_cols = n if n_cols is None else n_cols
    capy_sq = capybara.capybara_silhouette((1.0, 1.0), n_grid=n)
    capy = np.zeros((n, n_cols), dtype=bool)
    capy[:, col_off : col_off + n] = capy_sq
    rows = (np.arange(n)[:, None] / n)  # y in 0..1, y-down
    below = np.broadcast_to(rows >= waterline_y, (n, n_cols))
    water_band = below.copy()
    capy_above = capy & ~below
    capy_below = capy & below
    return {
        "capy": capy,
        "capy_above": capy_above,
        "capy_below": capy_below,
        "water_band": water_band,
    }


# --- Flow-field constants (shared with plate.frag::waterRippleCoverage) -------
# All in NORMALIZED art-box units (x,y in 0..1, y-down). The flow is a field of
# long, undulating STREAMLINES (ridges running along the flow/x axis) whose
# transverse undulation travels downstream by one full wavelength across the N
# animation frames — so the walking slit barrier sweeps the crests laterally and
# the water reads as a continuous DIRECTIONAL current, not stepping bands. A wake
# opens around the half-submerged body (streamlines part in y around it, a calm
# elliptical patch sits under the belly, and the undulation is shoved downstream
# behind it into a trailing tongue). Vision-tuned in the r1..r3 standalone
# harness; keep these in lock-step with the GLSL copy.
RIPPLE_WAVELENGTH_NORM = 0.14  # crest spacing as a fraction of the art box
FLOW_DIR = 1.0            # +1: crests advance toward +x as the frame index grows
_BODY_CX = 0.46           # body-center x (matches capybara motif anchor)
_BODY_CY = 0.60           # body-center y
_STREAM_BAND_FRAC = 0.55  # streamline spacing = wavelength * this
_UNDULATE_A1 = 0.42       # primary transverse undulation amplitude (band units)
_UNDULATE_A2 = 0.14       # second-harmonic amplitude (liveliness)
_DEPTH_SHEAR = 0.35       # deeper streamlines lag → raked, layered current
_WAKE_INFL = 0.30         # gaussian radius of the body's influence on streamlines
_WAKE_PART = 0.22         # how far streamlines part (in y) around the body
_WAKE_SHOVE = 0.22        # downstream shove magnitude (trailing tongue)


def _flow_streamline_field(X, Y, wavelength, n_phases, phase_step, waterline_y):
    """Scalar field whose (near-)integer iso-lines are the flowing streamlines.

    ``s = (y + wake_part) / band_gap - undulation(x - travel - wake_shove)``.
    Crests sit where ``s`` is near an integer; the undulation travels downstream
    with ``phase_step`` so the ridges march laterally (the current). The wake
    terms part the streamlines around the body and drag them downstream behind
    it. Mirrors the GLSL ``flowStreamlineField`` exactly.
    """
    depth = np.clip((Y - waterline_y) / max(1e-6, (1.0 - waterline_y)), 0.0, 1.0)
    travel = wavelength * FLOW_DIR * (phase_step / max(1.0, n_phases))
    shear = 2.0 * np.pi * _DEPTH_SHEAR * depth

    dx = X - _BODY_CX
    dy = Y - _BODY_CY
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    infl = np.exp(-((r / _WAKE_INFL) ** 2))
    # Streamlines PART around the body (smooth tanh through the wake axis).
    y_part = _WAKE_PART * np.tanh(dy / 0.10) * infl
    # Downstream shove → the undulation trails behind the body (a tongue/V-wake).
    downstream = 0.5 + 0.5 * np.tanh(dx * FLOW_DIR / 0.10)
    centerline = np.exp(-((dy / 0.18) ** 2))
    xshove = _WAKE_SHOVE * downstream * centerline * infl * FLOW_DIR

    phase_x = 2.0 * np.pi * (X - travel - xshove) / wavelength + shear
    undulation = (
        _UNDULATE_A1 * np.sin(phase_x) + _UNDULATE_A2 * np.sin(2.0 * phase_x + 0.6)
    ) * (1.0 - 0.3 * depth)

    band_gap = wavelength * _STREAM_BAND_FRAC
    return (Y + y_part) / band_gap - undulation


def _flow_amplitude(X, Y, waterline_y):
    """Crest strength 0..1: strong open water, calm elliptical patch under the
    belly (extended a touch downstream), gently fading with depth. Mirrors the
    GLSL ``flowAmplitude``."""
    depth = np.clip((Y - waterline_y) / max(1e-6, (1.0 - waterline_y)), 0.0, 1.0)
    amp = 1.0 - 0.35 * depth
    bx = _BODY_CX
    by = _BODY_CY + 0.16
    ex = (X - bx - 0.05 * FLOW_DIR) / 0.17
    ey = (Y - by) / 0.11
    rb = np.sqrt(ex * ex + ey * ey)
    calm = np.clip((rb - 0.55) / 0.9, 0.0, 1.0)
    return np.clip(amp * calm, 0.0, 1.0)


# Litho floor for the fab (µm): minimum gold line == minimum gap == 2 µm. The
# scanimation's CONTROLLED minimums are the slit-slot (frame_pitch/N, 15 µm at
# defaults) and the barrier bar (frame_pitch·(1−duty), 45 µm) — both far above
# this floor. The only way sub-floor gold can appear is as an UNCONTROLLED
# sliver: a crest ridge that tapers to <2 µm in y at its tips/edges, or a crest
# fragment left <2 µm wide in x where it straddles a slit-slot boundary. Both
# are clipped at the SOURCE (see _ripple_phase_masks thickness gate and
# _interleave_phases slot snap) so the baked geometry has nothing between 0 and
# 2 µm — only the controlled 15/45 µm features.
LITHO_FLOOR_UM = 2.0


def _ripple_phase_masks(
    n: int,
    cell_um: float,
    frame_pitch_um: float,
    n_phases: int,
    water_band: np.ndarray,
    waterline_y: float = WATERLINE_Y,
    n_cols: int | None = None,
    col_off: int = 0,
) -> list[np.ndarray]:
    """N interleaved ripple frames for the back layer — a flowing CURRENT.

    Each phase advances the traveling-wave undulation by one slot (1/N of a
    wavelength), so under the walking slit barrier the streamline crests sweep
    laterally in ONE consistent direction — the water flows. The field is a set
    of long undulating streamlines (ridges along x) that part around the body
    and go calm under the belly (see ``_flow_streamline_field`` /
    ``_flow_amplitude``); this is the naturalistic wake read the shader renders
    analytically in preview and the fab SVG bakes via ``_interleave_phases``.

    The *interleave* is enforced downstream by the slit barrier (front); here we
    render each animation frame at full width inside the water band, and the
    ``generate`` interleaver slices frame k into its 1/N slot column set.
    """
    # x normalized to the BODY square (width n, inset at col_off): wing columns
    # fall outside [0,1] so the periodic streamline current simply continues past
    # the animal. y is normalized to the square height n (y-down).
    n_cols = n if n_cols is None else n_cols
    xs = (np.arange(n_cols)[None, :].astype(np.float32) - col_off) / n  # normalized x
    ys_row = np.arange(n)[:, None].astype(np.float32) / n  # 0..1 y-down
    X = np.broadcast_to(xs, (n, n_cols)).astype(np.float32)
    Y = np.broadcast_to(ys_row, (n, n_cols)).astype(np.float32)

    # Ripple wavelength in NORMALIZED units (fraction of the art box). Broad and
    # legible (~7 crests across the band) — the r3 vision default. The scanimation
    # *timing* (how much tilt advances one frame) is set by frame_pitch/N via the
    # slit barrier interleave, independent of this visual crest spacing, so the
    # wavelength is a fixed design constant here.
    wavelength = RIPPLE_WAVELENGTH_NORM

    amp = _flow_amplitude(X, Y, waterline_y)
    masks = []
    for k in range(n_phases):
        s = _flow_streamline_field(X, Y, wavelength, n_phases, float(k), waterline_y)
        f = s - np.floor(s + 0.5)          # signed distance to nearest streamline
        half = 0.16 * amp                  # crest half-width (calm → vanishes)
        crest = (np.abs(f) < half) & (amp > 0.05)
        crest = _floor_crest_thickness(crest & water_band, cell_um)
        masks.append(crest)
    return masks


def _floor_crest_thickness(crest: np.ndarray, cell_um: float) -> np.ndarray:
    """Remove sub-floor gold ridges from a crest mask (F3, y-axis source clip).

    A thresholded streamline field (``|f| < half``) tapers to a 1-pixel ridge at
    each crest's tips and where the calm-patch amplitude fades. Those 0.5–1.5 µm
    tall runs are UNCONTROLLED sub-floor slivers. We open the mask vertically by
    the litho floor: any column-run of gold shorter than ``LITHO_FLOOR_UM`` is
    dropped, so every surviving ridge is ≥ 2 µm thick. This is a 1-D open along
    y only (crests run along x, so their controlled thickness is the y-extent);
    it never widens gold, only removes slivers, so it cannot bridge a slit slot.

    Vectorised per-column run-length filter (no GEOS): seconds on the whole
    water band even at the fine analysis raster.
    """
    min_run = max(1, int(round(LITHO_FLOOR_UM / cell_um)))
    if min_run <= 1:
        return crest
    h, _w = crest.shape
    c = crest.astype(np.int32)
    # run_len[i, col] = length of the consecutive-True run ENDING at row i.
    # Both passes loop over ROWS (h, small) and act on all columns at once — no
    # per-column Python loop, so this stays fast at the fine analysis raster.
    run_len = np.zeros_like(c)
    run_len[0] = c[0]
    for i in range(1, h):
        run_len[i] = (run_len[i - 1] + 1) * c[i]
    # Carry each run's TOTAL length (known at its end) back up to every cell of
    # that run: sweep bottom→top, take max with the row below ONLY while still
    # inside the same run (cell is True). At a gap the carry is 0, so it cannot
    # leak into the run above.
    total = run_len.copy()
    for i in range(h - 2, -1, -1):
        total[i] = np.where(c[i] > 0, np.maximum(total[i], total[i + 1] * c[i]), 0)
    out = crest & (total >= min_run)
    return out


def _measure_min_gold_um(mask: np.ndarray, cell_um: float) -> float:
    """Narrowest gold feature in ``mask`` (µm), measured as the shortest
    connected run of gold in x (per row) OR y (per column). This is the honest,
    geometry-derived minimum the manifest reports for F7 — it reflects whatever
    the source clips actually produced, so it cannot silently disagree with the
    baked SVG. Vectorised run-detection; runs in ms on the pattern raster.
    """
    if not mask.any():
        return float("inf")
    best = np.inf
    # x-runs: a gold→boundary transition per row. Count run lengths via diffs of
    # cumulative gold within each contiguous stretch.
    for axis in (0, 1):  # 0: runs along columns (y), 1: runs along rows (x)
        m = mask if axis == 1 else mask.T  # make the run axis the last axis
        # Pad with False columns so edge runs terminate.
        pad = np.zeros((m.shape[0], 1), dtype=bool)
        mm = np.concatenate([pad, m, pad], axis=1)
        d = np.diff(mm.astype(np.int8), axis=1)
        # starts where d==1, ends where d==-1; run length = end-index − start-index.
        starts_r, starts_c = np.where(d == 1)
        ends_r, ends_c = np.where(d == -1)
        # d==1 and d==-1 come in matched order per row, so lengths align.
        if starts_c.size:
            lengths = (ends_c - starts_c) * cell_um
            best = min(best, float(lengths.min()))
    return best


def _stripe_carrier(
    n: int, period_pix: float, phase: float = 0.0, n_cols: int | None = None
) -> np.ndarray:
    """Vertical stripe carrier (bool), True = gold, matching the shader's
    analytic carrier so the baked capybara shimmer lines up head-on."""
    n_cols = n if n_cols is None else n_cols
    cols = np.arange(n_cols)[None, :].astype(np.float32)
    frac = ((cols / period_pix) + phase) % 1.0
    return np.broadcast_to(frac < 0.5, (n, n_cols)).copy()


def _slit_barrier(
    n: int, period_pix: float, duty: float, phase: float = 0.0, n_cols: int | None = None
) -> np.ndarray:
    """Slit barrier: True where the slit is OPEN (transmits one back phase).
    Open fraction == ``duty`` (≈1/n_phases)."""
    n_cols = n if n_cols is None else n_cols
    cols = np.arange(n_cols)[None, :].astype(np.float32)
    frac = ((cols / period_pix) + phase) % 1.0
    return np.broadcast_to(frac < duty, (n, n_cols)).copy()


def _interleave_phases(
    phase_masks: list[np.ndarray],
    period_pix: float,
) -> np.ndarray:
    """Pack N ripple frames into their 1/N slot column-sets → the single baked
    BACK water mask. Phase k occupies columns whose within-period slot is k.

    F3 (x-axis source clip): a crest is a set of gold rows spanning some columns.
    Selecting only the slot-k columns of phase k would leave sub-floor gold where
    a crest STARTS or ENDS partway across a slot — a <2 µm-wide fragment hugging a
    slot boundary. We prevent that by making each slot ATOMIC: within a given
    (row, slot) cell the gold is set for the WHOLE slot width iff phase k covers
    at least the litho floor's worth of that slot's columns, else it is cleared.
    So every emitted gold column-run is either a full slot (≥ slot_um ≈ 15 µm) or
    absent — nothing between 0 and 2 µm, and crest edges are snapped to the slot
    grid exactly as the fab process controls it. Slots are ~15 µm ≫ 2 µm, so the
    ≥floor-coverage rule keeps virtually all of each crest; it only trims the
    ragged partial-slot tail that would otherwise print as a sliver.
    """
    n = phase_masks[0].shape[1]
    n_phases = len(phase_masks)
    cols = np.arange(n).astype(np.float32)
    slot = np.floor(((cols % period_pix) / period_pix) * n_phases).astype(int)
    slot = np.clip(slot, 0, n_phases - 1)
    # Global slot index so each 15 µm slot across the whole width is one group.
    period_idx = np.floor(cols / period_pix).astype(int)
    slot_id = period_idx * n_phases + slot            # unique per physical slot
    uniq = np.unique(slot_id)
    # Columns belonging to each unique slot, and which phase that slot serves.
    cover_floor = max(1, int(round(LITHO_FLOOR_UM / (period_pix / n_phases))))
    out = np.zeros_like(phase_masks[0])
    h = out.shape[0]
    for sid in uniq:
        col_sel = np.where(slot_id == sid)[0]
        k = int(sid % n_phases)
        if k >= n_phases:
            continue
        ph = phase_masks[k]
        # Per-row count of covered columns inside this slot.
        block = ph[:, col_sel]                        # (h, slot_w)
        row_cover = block.sum(axis=1)
        rows_on = row_cover >= cover_floor            # atomic-fill these rows
        if rows_on.any():
            out[np.ix_(rows_on, col_sel)] = True
    return out


def _build(
    extent_um: float,
    frame_pitch_um: float,
    n_phases: int,
    carrier_period_um: float,
    waterline_y: float,
    n_grid: int | None = None,
    water_extent_um: float | None = None,
):
    """Core builder shared by ``generate`` and the standalone vision harness.

    ``water_extent_um`` (optional, ≥ ``extent_um``) widens the WATER BAND / slit
    comb / interleave to span that full width while the capybara body + flow-field
    wake stay registered to the centered ``extent_um`` square — the plate
    compositor uses this for WATER FULL WIDTH (band edge-to-edge across the
    aperture). ``None`` keeps the square art tile (standalone pattern default).
    """
    n_phases = max(2, min(6, int(n_phases)))
    slot_um = frame_pitch_um / n_phases
    if n_grid is None:
        # Aim for ~6 samples across the finest feature (the slit slot) so the
        # interleave and barrier stay crisp, but never exceed the shared 400k
        # lattice cap — the plate compositor upscales this raster to fill the
        # aperture, so a modest grid is plenty (matches the other artistic
        # patterns' resolution).
        want = int(extent_um / max(1.0, slot_um / 6.0))
        n_cap = int((MAX_LATTICE_CELLS) ** 0.5)
        n_grid = max(384, min(want, n_cap))
    cell_um = extent_um / n_grid
    # Water band width: full aperture (WATER FULL WIDTH) or the body square.
    if water_extent_um is not None and water_extent_um > extent_um:
        n_cols = int(round(water_extent_um / cell_um))
    else:
        n_cols = n_grid
    col_off = (n_cols - n_grid) // 2
    # Guard the actual build (a caller-supplied n_grid or an extreme param combo
    # could still overshoot); surfaces as a UI-friendly HTTP 400.
    check_lattice_budget(
        n_grid * n_cols,
        "capybara-scanimation raster",
        extent_um=extent_um,
        frame_pitch_um=frame_pitch_um,
        n_phases=n_phases,
    )

    scene = _capybara_and_water(n_grid, waterline_y, n_cols=n_cols, col_off=col_off)
    frame_pix = frame_pitch_um / cell_um
    carrier_pix = carrier_period_um / cell_um
    slit_duty = 1.0 / n_phases

    phase_masks = _ripple_phase_masks(
        n_grid, cell_um, frame_pitch_um, n_phases, scene["water_band"], waterline_y,
        n_cols=n_cols, col_off=col_off,
    )
    back_water = _interleave_phases(phase_masks, frame_pix)

    # Front capybara body shimmer: carrier stripes clipped to the dry body plus
    # a faint submerged hint below the waterline (dashed reflection feel).
    carrier = _stripe_carrier(n_grid, carrier_pix, n_cols=n_cols)
    capy_shimmer = (scene["capy_above"] & carrier) | (
        scene["capy_below"] & carrier
        & _slit_barrier(n_grid, carrier_pix * 2.0, 0.5, n_cols=n_cols)
    )

    # Front slit barrier over the water band (open where True).
    barrier = _slit_barrier(n_grid, frame_pix, slit_duty, n_cols=n_cols)

    plan = moire.scanimation_plan(n_phases, frame_pitch_um)

    return {
        "n_grid": n_grid,
        "n_cols": n_cols,
        "col_off": col_off,
        "cell_um": cell_um,
        "frame_pix": frame_pix,
        "carrier_pix": carrier_pix,
        "slit_duty": slit_duty,
        "scene": scene,
        "phase_masks": phase_masks,
        "back_water": back_water,
        "capy_shimmer": capy_shimmer,
        "barrier": barrier,
        "water_band": scene["water_band"],
        "plan": plan,
    }


def build_debug(extent_um: float = 1600.0):
    """Vision-harness entry point (masks + plan, no shapely)."""
    b = _build(
        extent_um=extent_um,
        frame_pitch_um=60.0,
        n_phases=4,
        carrier_period_um=24.0,
        waterline_y=WATERLINE_Y,
    )
    # Compose the front mask (capybara shimmer OR the barrier bars over water)
    # the way the shader will read uFront. Barrier bars are gold (block light).
    barrier_bars = (~b["barrier"]) & b["water_band"]
    front_mask = b["capy_shimmer"] | barrier_bars
    return {
        "cell_um": b["cell_um"],
        "front_mask": front_mask,
        "barrier_mask": b["barrier"],       # True = open slit
        "phase_masks": b["phase_masks"],
        "back_water": b["back_water"],
        "water_band": b["water_band"],
        "capy_shimmer": b["capy_shimmer"],
        "plan": b["plan"],
    }


@register
class CapybaraScanimation(Pattern):
    slug = "capybara-scanimation"
    name = "Capybara + water scanimation"
    description = (
        "A serene capybara half-submerged on a waterline; rocking the box walks "
        "a slit barrier across an N-phase interleaved ripple field so the water "
        "appears to FLOW around the animal. Front layer: the capybara silhouette "
        "filled with a fine shimmer carrier, plus a slit barrier (open duty 1/N) "
        "over the water band. Back layer: N ripple frames packed into 1/N slots. "
        "Snell parallax through the 500 µm substrate advances one ripple phase "
        "per ≈2.5° of tilt (full cycle ≈10°), a natural hand rock. A barrier-grid "
        "scanimation (kinegram) in gold-on-silica."
    )
    tags = ["scanimation", "kinegram", "tilt-animate", "water", "capybara", "Colombia"]
    tier = 1
    theme = "Colombia"
    # Ships a slit barrier + interleaved back frames as separate layers, exactly
    # like the stereo_lenticular recipe — see integration notes for the small
    # shader step the N-phase advance needs beyond the 2-view A/B blend.
    render_recipe = "stereo_lenticular"
    params = [
        ParamSpec("frame_pitch_um", "Frame pitch", "float", 60.0, 24.0, 120.0, 2.0, "μm"),
        ParamSpec("n_phases", "Ripple phases", "int", 4, 2, 6, 1),
        ParamSpec("carrier_period_um", "Body shimmer period", "float", 24.0, 6.0, 60.0, 1.0, "μm"),
        ParamSpec("waterline", "Waterline (0=top,1=bottom)", "float", WATERLINE_Y, 0.4, 0.85, 0.01),
        ParamSpec("extent_um", "Extent", "float", 1600.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        frame_pitch_um: float = 60.0,
        n_phases: int = 4,
        carrier_period_um: float = 24.0,
        waterline: float = WATERLINE_Y,
        extent_um: float = 1600.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)
        b = _build(
            extent_um=extent_um,
            frame_pitch_um=frame_pitch_um,
            n_phases=int(n_phases),
            carrier_period_um=carrier_period_um,
            waterline_y=waterline,
            n_grid=None,
        )
        cell_um = b["cell_um"]
        n_grid = b["n_grid"]

        # FRONT = capybara shimmer + slit-barrier bars over the water band.
        barrier_bars = (~b["barrier"]) & b["water_band"]
        front_mask = b["capy_shimmer"] | barrier_bars

        # BACK = interleaved ripple frames (all N phases packed into slots).
        back_mask = b["back_water"]

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        # Extra layers: each ripple phase as its own full-width frame, so the
        # shader (or a future baked animation) can step them, plus the clean
        # capybara silhouette for compositing / masking. Named phase_0..N-1.
        extra_layers = {
            f"phase_{k}": ensure_multipolygon(
                raster_to_polygons(ph.astype(np.uint8), cell_um, extent)
            )
            for k, ph in enumerate(b["phase_masks"])
        }
        extra_layers["capy"] = ensure_multipolygon(
            raster_to_polygons(b["scene"]["capy"].astype(np.uint8), cell_um, extent)
        )

        plan = b["plan"]
        slot_um = frame_pitch_um / int(n_phases)
        # F7: the controlled minimum gold features, MEASURED from the emitted
        # geometry rather than guessed from a duty formula. After the source
        # sliver clips (`_floor_crest_thickness` + slot-atomic `_interleave_phases`)
        # the back layer's smallest gold is the litho floor in y (crest thickness)
        # and one slit slot in x; the front's smallest is the narrower of the
        # shimmer stripe and the slit BARRIER BAR. We probe the actual masks so
        # the manifest cannot drift from the geometry again.
        min_back_gold = _measure_min_gold_um(back_mask, cell_um)
        min_front_gold = _measure_min_gold_um(front_mask, cell_um)
        # Barrier bar (front slit closed fraction) — a controlled design minimum.
        barrier_bar_um = frame_pitch_um * (1.0 - b["slit_duty"])
        slit_slot_um = slot_um  # front slit OPEN width == back interleave slot
        min_feature = min(min_back_gold, min_front_gold)

        return GeneratedPattern(
            front=ensure_multipolygon(front_poly),
            back=ensure_multipolygon(back_poly),
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=float(min_feature),
            extra={
                "n_phases": int(n_phases),
                "frame_pitch_um": frame_pitch_um,
                "slit_duty": float(b["slit_duty"]),
                "slot_um": float(slot_um),
                # F7: controlled minimums, MEASURED from geometry (not a formula
                # guess). slit slot (open) and barrier bar (closed) are the
                # design-controlled features; min_*_gold are what the cleaned
                # masks actually contain (litho floor for the crest thickness).
                "slit_slot_um": float(slit_slot_um),
                "barrier_bar_um": float(barrier_bar_um),
                "min_back_gold_um": float(min_back_gold),
                "min_front_gold_um": float(min_front_gold),
                "min_slit_gold_um": float(min_front_gold),  # legacy key, now measured
                "litho_floor_um": float(LITHO_FLOOR_UM),
                "tilt_per_frame_deg": float(plan.tilt_per_frame_deg),
                "tilt_full_cycle_deg": float(plan.tilt_full_cycle_deg),
                "scanimation_note": plan.note,
                "waterline": float(waterline),
            },
            extra_layers=extra_layers,
            recipe_data={
                # Slit-normal axis: vertical bars → horizontal walk (+X), like
                # the lenticular. The integrator advances phase by parallax on
                # this axis (see integration notes).
                "slit_axis_deg": 0.0,
                "slit_period_um": frame_pitch_um,
                "n_phases": int(n_phases),
                "frame_pitch_um": frame_pitch_um,
                "carrier_period_um": carrier_period_um,
            },
        )
