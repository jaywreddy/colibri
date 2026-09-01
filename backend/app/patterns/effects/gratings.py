"""Parameterized fine-pitch grating GENERATORS.

Each generator returns a :class:`GratingResult` carrying a numpy BOOL mask
(``True`` = gold) at a caller-chosen raster pitch, plus the parameters the fab
baker needs. The bool mask is the interchange format the shader-preview raster
and the moiré demos both consume; :func:`bool_to_row_spans` /
:func:`row_spans_to_verts` turn a mask into the SAME row-span rectangle arrays
``plates.raster_to_polygons`` emits, so the integrator can drop these straight
into ``ensure_plate_svg`` with no new geometry code.

Conventions (kept identical to ``plates._grating_grid`` so results compose):
  * grids are ``(h_px, w_px)`` row-major;
  * physical origin at grid CENTER, x grows RIGHT, y grows UP;
  * ``pitch_um`` is the µm size of one pixel;
  * ``angle_deg`` rotates the grating lines CCW; a linear grating at angle 0
    has VERTICAL gold lines (constant-x stripes), matching the vertical
    centerpiece switch carrier.

Budget: every generator estimates the row-span rectangle count it will bake and
refuses (``coarsen=False``) or auto-coarsens the period (``coarsen=True``) when
that would exceed the shared 400k-lattice cap. The estimate is line-count based
(lines across the extent × rows they span) so it matches what the fab path
actually allocates, not the raw pixel count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .moire import DIFFRACTION_PERIOD_UM, MIN_PERIOD_UM

# --- diffraction rainbow accent (sub-5 µm grating) --------------------------
# A dedicated fine grating for the "hologram-foil sticker" accent zones (colibrí
# gorget, steam-curl tips, gear hub, monogram flourish tips). Its period is
# deliberately BELOW the ~5 µm diffraction onset so first-order visible light
# fans into a rainbow at accessible hand-tilt angles, yet SAFELY ABOVE the 2 µm
# line / 2 µm gap litho floor: 4.4 µm period = 2.2 µm line + 2.2 µm gap. A fixed
# 45° line angle gives the fan a consistent diagonal sweep no matter which face
# it lands on. See moire.diffraction_onset(4.4) for the exact fan geometry.
DIFFRACTION_ACCENT_PERIOD_UM = 4.4   # 2.2 µm line + 2.2 µm gap (> 4 µm floor)
DIFFRACTION_ACCENT_ANGLE_DEG = 45.0  # fixed diagonal grating orientation
DIFFRACTION_ACCENT_DUTY = 0.5

# Shared with app.patterns._helpers.MAX_LATTICE_CELLS. Imported lazily in
# grating_line_budget so this module stays importable stand-alone (the demos
# runner imports it before the app package is on the path in some harnesses).
_DEFAULT_BUDGET = 400_000


def _lattice_budget() -> int:
    try:
        from .._helpers import MAX_LATTICE_CELLS

        return int(MAX_LATTICE_CELLS)
    except Exception:  # noqa: BLE001 — stand-alone demo import
        return _DEFAULT_BUDGET


@dataclass
class GratingResult:
    """A rasterized grating: bool mask + the knobs the fab baker needs."""

    mask: np.ndarray                 # (h_px, w_px) bool, True = gold
    pitch_um: float                  # µm per pixel
    extent_um: tuple[float, float]   # (W, H) the mask spans
    period_um: float                 # effective period AFTER any coarsening
    requested_period_um: float       # what the caller asked for
    duty: float
    kind: str                        # "linear" | "radial" | "chirped" | "checker"
    coarsened: bool = False          # True if budget forced a coarser period
    meta: dict = field(default_factory=dict)

    @property
    def gold_fraction(self) -> float:
        return float(self.mask.mean())

    def row_spans(self):
        """Row-span (x0,x1,y0,y1) rectangle arrays for the fab/SVG path."""
        return bool_to_row_spans(self.mask, self.pitch_um)


# --- budget accounting ------------------------------------------------------

def grating_line_budget(
    period_um: float,
    extent_um: tuple[float, float],
    *,
    budget: int | None = None,
) -> dict:
    """Estimate the row-span rectangle count a linear grating would bake and
    report whether it fits the lattice budget.

    A rotated linear grating rasterized at ≥4 samples/period allocates roughly
    one row-span rectangle per line per row it crosses. With ``n_lines`` lines
    across the diagonal and ``h_px`` rows, the worst case (lines nearly
    horizontal) is ``n_lines · w_px``; the typical (near-vertical) case is
    ``n_lines · h_px``. We budget on the pixel grid size ``w_px · h_px`` since
    that upper-bounds the run count and matches ``plates`` conservatism.

    Returns a dict: ``fits``, ``pixels``, ``lines``, ``budget``, and a
    ``min_pitch_um`` (the coarsest pitch that fits) and ``coarsen_period_um``
    (period rescaled to that pitch, preserving 4-samples/period).
    """
    budget = budget or _lattice_budget()
    w, h = extent_um
    ideal_pitch = max(period_um / 4.0, 1e-6)
    w_px = max(1, int(round(w / ideal_pitch)))
    h_px = max(1, int(round(h / ideal_pitch)))
    pixels = w_px * h_px
    diag = math.hypot(w, h)
    lines = int(diag / period_um) + 2
    # Coarsest pitch that fits the budget on this extent.
    min_pitch = math.sqrt(w * h / max(1, budget))
    coarsen_pitch = max(ideal_pitch, min_pitch)
    coarsen_period = period_um * (coarsen_pitch / ideal_pitch)
    return {
        "fits": pixels <= budget,
        "pixels": pixels,
        "lines": lines,
        "budget": budget,
        "ideal_pitch_um": ideal_pitch,
        "min_pitch_um": coarsen_pitch,
        "coarsen_period_um": coarsen_period,
    }


def _resolve_pitch_period(
    period_um: float,
    extent_um: tuple[float, float],
    pitch_um: float | None,
    coarsen: bool,
    budget: int | None,
) -> tuple[float, float, bool]:
    """Common pitch/period resolution + budget guard for the generators.

    Returns ``(pitch_um, effective_period_um, coarsened)``. Enforces the 4 µm
    litho floor on the REQUESTED period, picks a ≥4-samples/period pitch, and
    either coarsens the period to fit the budget (``coarsen=True``) or raises
    (``coarsen=False``).
    """
    if period_um < MIN_PERIOD_UM - 1e-9:
        raise ValueError(
            f"period {period_um:g} µm < {MIN_PERIOD_UM:g} µm litho floor "
            "(min line 2 µm + min gap 2 µm)."
        )
    budget = budget or _lattice_budget()
    info = grating_line_budget(period_um, extent_um, budget=budget)
    if pitch_um is None:
        pitch = info["ideal_pitch_um"]
    else:
        pitch = pitch_um
    w, h = extent_um
    w_px = max(1, int(round(w / pitch)))
    h_px = max(1, int(round(h / pitch)))
    if w_px * h_px <= budget:
        return pitch, period_um, False
    if not coarsen:
        raise ValueError(
            f"grating over {w:g}×{h:g} µm at period {period_um:g} µm would build "
            f"{w_px * h_px:,} cells (cap {budget:,}). Coarsen the period, shrink "
            "the extent, or pass coarsen=True."
        )
    # Coarsen: rescale pitch (and period, to keep 4 samples/period + the moiré
    # ratio) up to the budget-fitting pitch.
    new_pitch = math.sqrt(w * h / (0.98 * budget))
    scale = new_pitch / info["ideal_pitch_um"]
    return new_pitch, period_um * scale, True


# --- generators -------------------------------------------------------------

def _linear_frac(
    w_px: int,
    h_px: int,
    pitch_um: float,
    period_um: float,
    angle_deg: float,
    phase: float,
) -> np.ndarray:
    """Fractional phase 0..1 of a rotated linear grating on the pixel grid.

    Mirrors ``plates._grating_grid``'s projected-coordinate math exactly so a
    grating baked here lines up with one baked there (origin at center, y up).
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    xs = (np.arange(w_px) - (w_px - 1) / 2.0) * pitch_um
    ys = ((h_px - 1) / 2.0 - np.arange(h_px)) * pitch_um
    X, Y = np.meshgrid(xs, ys)
    coord = (X * ca + Y * sa) / period_um + phase
    return coord - np.floor(coord)


def linear_grating_mask(
    period_um: float,
    extent_um: tuple[float, float],
    *,
    duty: float = 0.5,
    angle_deg: float = 0.0,
    phase: float = 0.0,
    pitch_um: float | None = None,
    coarsen: bool = True,
    budget: int | None = None,
) -> GratingResult:
    """Linear (line/space) grating. angle 0 ⇒ vertical gold lines.

    ``duty`` is the gold-line fraction of a period; ``phase`` shifts the grating
    by that fraction of a period (0.5 = half-period interlace). Budget-aware via
    :func:`_resolve_pitch_period`.
    """
    pitch, eff_period, coarsened = _resolve_pitch_period(
        period_um, extent_um, pitch_um, coarsen, budget
    )
    w, h = extent_um
    w_px = max(1, int(round(w / pitch)))
    h_px = max(1, int(round(h / pitch)))
    frac = _linear_frac(w_px, h_px, pitch, eff_period, angle_deg, phase)
    mask = frac < duty
    return GratingResult(
        mask=mask, pitch_um=pitch, extent_um=(w, h), period_um=eff_period,
        requested_period_um=period_um, duty=duty, kind="linear",
        coarsened=coarsened, meta={"angle_deg": angle_deg, "phase": phase},
    )


def radial_grating_mask(
    period_um: float,
    extent_um: tuple[float, float],
    *,
    duty: float = 0.5,
    center: tuple[float, float] = (0.0, 0.0),
    n_spokes: int = 0,
    pitch_um: float | None = None,
    coarsen: bool = True,
    budget: int | None = None,
) -> GratingResult:
    """Concentric-ring (radial) grating; optional angular ``n_spokes`` turns it
    into a rosette (rings × spokes → sparkle that shimmers as it tilts).

    Rings are gold when ``frac(r/period) < duty``. The LOCAL ring period is
    ``period_um`` everywhere (a true radial carrier), which is what beats
    against a linear back carrier into a circular moiré rosette. The 4 µm floor
    applies to the ring period; near the center the effective azimuthal feature
    of a spoked rosette can still fall below the floor, so we flag it in meta.
    """
    pitch, eff_period, coarsened = _resolve_pitch_period(
        period_um, extent_um, pitch_um, coarsen, budget
    )
    w, h = extent_um
    w_px = max(1, int(round(w / pitch)))
    h_px = max(1, int(round(h / pitch)))
    cx, cy = center
    xs = (np.arange(w_px) - (w_px - 1) / 2.0) * pitch - cx
    ys = ((h_px - 1) / 2.0 - np.arange(h_px)) * pitch - cy
    X, Y = np.meshgrid(xs, ys)
    R = np.hypot(X, Y)
    ring = (R / eff_period) % 1.0 < duty
    mask = ring
    if n_spokes > 0:
        theta = np.arctan2(Y, X)
        spoke = ((theta / (2.0 * math.pi) * n_spokes) % 1.0) < duty
        mask = ring & spoke
    return GratingResult(
        mask=mask, pitch_um=pitch, extent_um=(w, h), period_um=eff_period,
        requested_period_um=period_um, duty=duty, kind="radial",
        coarsened=coarsened, meta={"center": center, "n_spokes": n_spokes},
    )


def chirped_grating(
    period_start_um: float,
    period_end_um: float,
    extent_um: tuple[float, float],
    *,
    duty: float = 0.5,
    angle_deg: float = 0.0,
    pitch_um: float | None = None,
    coarsen: bool = True,
    budget: int | None = None,
) -> GratingResult:
    """Linear grating whose period SWEEPS from ``period_start`` to ``period_end``
    across the extent (a chirp).

    A chirp makes a graded diffraction/moiré accent: one edge diffracts a
    rainbow (fine end < 5 µm), the other reads as a clean carrier (coarse end).
    Beating a chirp against a uniform back carrier gives a moiré whose fringe
    spacing sweeps too — a "zipper" that reads as motion under tilt.

    Phase is the integral of instantaneous frequency so the line spacing is
    continuous (no kink). Budget uses the FINEST period (worst case).
    """
    finest = min(period_start_um, period_end_um)
    pitch, _eff, coarsened = _resolve_pitch_period(
        finest, extent_um, pitch_um, coarsen, budget
    )
    # If coarsening kicked in, scale both ends by the same factor.
    scale = pitch / max(finest / 4.0, 1e-6) if coarsened else 1.0
    p0 = period_start_um * scale
    p1 = period_end_um * scale
    w, h = extent_um
    w_px = max(1, int(round(w / pitch)))
    h_px = max(1, int(round(h / pitch)))
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    xs = (np.arange(w_px) - (w_px - 1) / 2.0) * pitch
    ys = ((h_px - 1) / 2.0 - np.arange(h_px)) * pitch
    X, Y = np.meshgrid(xs, ys)
    u = X * ca + Y * sa            # coordinate along the grating normal
    span = max(w, h)
    s = (u + span / 2.0) / span    # 0..1 across the extent
    s = np.clip(s, 0.0, 1.0)
    # Local period p(s) = p0 + (p1-p0)·s. Phase = ∫ ds/p(s) → log form for a
    # linear period ramp; falls back to u/p0 when p0≈p1 (uniform).
    dp = p1 - p0
    if abs(dp) < 1e-9:
        phase = u / p0
    else:
        # analytic ∫₀^u du'/p(u') with p linear in u: (span/dp)·ln(p(s)/p0)
        phase = (span / dp) * np.log((p0 + dp * s) / p0)
    frac = phase - np.floor(phase)
    mask = frac < duty
    return GratingResult(
        mask=mask, pitch_um=pitch, extent_um=(w, h),
        period_um=(p0 + p1) / 2.0, requested_period_um=(period_start_um + period_end_um) / 2.0,
        duty=duty, kind="chirped", coarsened=coarsened,
        meta={"period_start_um": p0, "period_end_um": p1, "angle_deg": angle_deg},
    )


def checker_grating(
    period_um: float,
    extent_um: tuple[float, float],
    *,
    duty: float = 0.5,
    angle_deg: float = 0.0,
    crosshatch: bool = False,
    pitch_um: float | None = None,
    coarsen: bool = True,
    budget: int | None = None,
) -> GratingResult:
    """Checkerboard (``crosshatch=False``) or crosshatch (``True``) grating.

    Checker = XOR of two perpendicular square waves → a 2-D lattice that beats
    against a back carrier in BOTH axes (a dot-lattice moiré). Crosshatch = OR
    → a grid of gold lines with open square windows (the inverse feel). Both
    are budget-aware; the 4 µm floor applies per axis.
    """
    pitch, eff_period, coarsened = _resolve_pitch_period(
        period_um, extent_um, pitch_um, coarsen, budget
    )
    w, h = extent_um
    w_px = max(1, int(round(w / pitch)))
    h_px = max(1, int(round(h / pitch)))
    fx = _linear_frac(w_px, h_px, pitch, eff_period, angle_deg, 0.0) < duty
    fy = _linear_frac(w_px, h_px, pitch, eff_period, angle_deg + 90.0, 0.0) < duty
    mask = (fx ^ fy) if not crosshatch else (fx | fy)
    return GratingResult(
        mask=mask, pitch_um=pitch, extent_um=(w, h), period_um=eff_period,
        requested_period_um=period_um, duty=duty,
        kind="checker", coarsened=coarsened,
        meta={"angle_deg": angle_deg, "crosshatch": crosshatch},
    )


# --- fab bakers (bool mask -> row-span rectangle arrays) --------------------

def bool_to_row_spans(mask: np.ndarray, cell_um: float) -> np.ndarray:
    """Row-length-merge a bool grid into (N, 4) [x0, x1, y0, y1] rectangles in
    µm, origin at grid center, y up.

    This is the numeric core of ``plates.raster_to_polygons`` (one rectangle per
    horizontal run of gold cells), returned as a plain float array so the caller
    can feed it to ``shapely.polygons`` (via :func:`row_spans_to_verts`) or emit
    SVG rects directly — no shapely dependency here.
    """
    g = np.asarray(mask) > 0
    h, w = g.shape
    hx = w * cell_um / 2.0
    hy = h * cell_um / 2.0
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = g
    d = np.diff(padded, axis=1)
    rows, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)
    if starts.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    x0 = starts * cell_um - hx
    x1 = ((ends - 1) * cell_um - hx) + cell_um
    y0 = hy - (rows + 1) * cell_um
    y1 = y0 + cell_um
    return np.stack([x0, x1, y0, y1], axis=1)


def row_spans_to_verts(spans: np.ndarray) -> np.ndarray:
    """(N,4) [x0,x1,y0,y1] rectangles → (N,4,2) CCW vertex array ready for
    ``shapely.polygons`` (matching ``plates.raster_to_polygons`` winding).
    """
    n = spans.shape[0]
    verts = np.empty((n, 4, 2), dtype=np.float64)
    x0, x1, y0, y1 = spans[:, 0], spans[:, 1], spans[:, 2], spans[:, 3]
    verts[:, 0, 0] = x0; verts[:, 0, 1] = y0
    verts[:, 1, 0] = x1; verts[:, 1, 1] = y0
    verts[:, 2, 0] = x1; verts[:, 2, 1] = y1
    verts[:, 3, 0] = x0; verts[:, 3, 1] = y1
    return verts


def clip_mask(mask: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
    """AND a grating mask with a silhouette bool grid (same shape) — the
    fill-inside-a-shape operation the frame/centerpiece bake needs. Pure numpy,
    no GEOS.
    """
    if mask.shape != silhouette.shape:
        raise ValueError(f"shape mismatch: {mask.shape} vs {silhouette.shape}")
    return mask & (np.asarray(silhouette) > 0)


# --- diffraction accent fab baker -------------------------------------------

def diffraction_accent_grating(
    zone: np.ndarray,
    pitch_um: float,
    *,
    period_um: float = DIFFRACTION_ACCENT_PERIOD_UM,
    angle_deg: float = DIFFRACTION_ACCENT_ANGLE_DEG,
    duty: float = DIFFRACTION_ACCENT_DUTY,
) -> np.ndarray:
    """Fill an accent ZONE bool grid with the fixed diffraction-rainbow grating,
    returning the gold bool grid the integrator ORs into the plate's front grid.

    ``zone`` is the (h_px, w_px) bool grid the integrator paints from the
    RAINBOW_LEVEL graylevel (see plates.py encoding spec); ``pitch_um`` is the
    plate raster pitch already in use by ``ensure_plate_svg`` (so the accent grid
    aligns cell-for-cell with the frame/centerpiece grids and can be OR-ed with
    them before ``raster_to_polygons``).

    The grating is a 45° linear line/space at ``period_um`` (default 4.4 µm =
    2.2 µm line + 2.2 µm gap), sub-5 µm so it diffracts a visible rainbow yet
    above the 4 µm litho floor. The line phase is generated the SAME way as
    ``plates._grating_grid`` / :func:`_linear_frac` so it composes with the other
    baked layers. No GEOS.

    NOTE for the integrator: at a coarse fab pitch the 4.4 µm period cannot be
    resolved (needs ≥ ~1.1 µm pitch for 4 samples/period). This baker does NOT
    coarsen — the diffraction physics REQUIRES the true sub-5 µm period, so a
    zone that only survives at a coarse budget pitch must be exported through a
    tiled/streamed GDS path at the native pitch, not this single-shot raster.
    When ``pitch_um`` is too coarse to resolve the period the returned grid will
    alias; the caller should assert ``pitch_um <= period_um / 4`` for accent
    zones or route them to the fine-pitch export. The physical mask ALWAYS uses
    ``period_um`` regardless of preview pitch.
    """
    zone = np.asarray(zone) > 0
    h_px, w_px = zone.shape
    frac = _linear_frac(w_px, h_px, pitch_um, period_um, angle_deg, 0.0)
    lines = frac < duty
    return lines & zone


def diffraction_accent_meta(
    period_um: float = DIFFRACTION_ACCENT_PERIOD_UM,
) -> dict:
    """Fan geometry for the accent grating (delegates to moire.diffraction_onset).

    Handy for manifests / QA: reports whether the chosen period diffracts and
    the first-order violet→red fan angles. Kept here so the fab baker and its
    physics description travel together.
    """
    from .moire import diffraction_onset

    d = diffraction_onset(period_um)
    return {
        "period_um": period_um,
        "line_um": period_um * DIFFRACTION_ACCENT_DUTY,
        "gap_um": period_um * (1.0 - DIFFRACTION_ACCENT_DUTY),
        "angle_deg": DIFFRACTION_ACCENT_ANGLE_DEG,
        "diffracts_visible": d.diffracts_visible,
        "first_order_violet_deg": d.first_order_deg_violet,
        "first_order_red_deg": d.first_order_deg_red,
        "spectral_spread_deg": d.spectral_spread_deg,
        "onset_period_um": DIFFRACTION_PERIOD_UM,
        "litho_floor_um": MIN_PERIOD_UM,
        "note": d.note,
    }


# --- sub-acuity interleaving -------------------------------------------------
# Band pitch for spatially interleaving two gratings in one zone. 48 µm subtends
# 0.55 arcmin at 300 mm — under the ~0.7 arcmin invisibility limit, so the eye
# cannot resolve the bands and simply sees BOTH effects superimposed over the
# same area. This is how a diffraction grating and a moiré louvre coexist:
# nesting the fine grating INSIDE the coarse one instead would make the coarse
# envelope split the spectrum into orders ~1.3° apart against a ~7° useful
# lobe, washing the colour toward white. Interleaved, each band stays a full
# clean grating of its own kind (the 24 µm diffraction band holds ~5.5 periods
# of 4.4 µm, giving orders ~1.3° wide against 7.2° separation) at the cost of
# roughly half the area each.
INTERLEAVE_BAND_PITCH_UM = 48.0


def band_select(
    local_rects: np.ndarray,
    band_pitch_um: float = INTERLEAVE_BAND_PITCH_UM,
    *,
    want_odd: bool = False,
    duty: float = 0.5,
    phase: float = 0.0,
) -> np.ndarray:
    """Keep the grating lines that fall in alternating bands.

    ``local_rects`` are ``(N,4)`` ``[x0,x1,y0,y1]`` in a GRATING-LOCAL frame —
    the frame where the lines are vertical, so ``x`` is the across-the-lines
    axis and a band is simply an x interval. Two gratings generated at the SAME
    angle therefore share one local frame, which is what lets both be clipped to
    the same band lattice with a scalar test instead of polygon clipping.

    ``want_odd`` picks the complementary set, so the two callers tile the zone
    exactly once with no overlap and no bare gap.
    """
    if local_rects.size == 0:
        return local_rects
    cx = 0.5 * (local_rects[:, 0] + local_rects[:, 1])
    frac = np.mod(cx / band_pitch_um - phase, 1.0)
    in_first = frac < duty
    keep = ~in_first if want_odd else in_first
    return local_rects[keep]


def beat_delta_um(period_um: float, beat_um: float) -> float:
    """Pitch mismatch that puts a shading-moire beat at ``beat_um``.

    ``beat = p*(p+d)/d``, so ``d = p^2 / (beat - p)``. Solve for d rather than
    fixing it, because the BEAT is the design intent -- how far apart the bands
    sit on the face -- while the carrier pitch is not free: on this build it is
    gap-scaled with the glass (``plates.parallax_period_scale``), so a carrier
    designed at 22 um becomes 63.5 um on 1.5 mm stock. A delta pinned at the
    22 um value would follow the carrier up and stretch the beat by the same
    2.9x, turning a dozen bands across the lid into one and a half.
    """
    if period_um <= 0.0:
        raise ValueError(f"period_um must be > 0 (got {period_um})")
    if beat_um <= period_um:
        raise ValueError(
            f"beat_um {beat_um} must exceed the {period_um} um carrier it beats against"
        )
    return period_um * period_um / (beat_um - period_um)


def shimmer_moire_layers(
    silhouette: np.ndarray,
    *,
    back_period_um: float,
    delta_um: float,
    cell_um: float,
    angle_deg: float = 0.0,
    duty: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """(front, back) for a single-figure motif that BEATS with the back carrier.

    The back plate already carries a uniform carrier across the whole exposed
    face, so a "front-only" motif is not short of a second grating — it is short
    of a reason for the two to beat. Filling the silhouette with a grating that
    runs PARALLEL to that carrier at a slightly different pitch turns the figure
    into a shading moiré: bands sweep through it where the two drift in and out
    of phase, so the letterform goes solid gold, then sinks back into the
    surrounding carrier haze, as the piece tilts.

    The beat comes from ``delta_um`` (the pitch mismatch) rather than from a
    crossing angle, and that is a fabrication choice, not an aesthetic one. With
    no backside alignment the front-to-back ROTATION is the one thing the build
    cannot hold, and a beat derived from a crossing angle is at its mercy: a
    0.45 deg design gives 2801 um, but 869 um if the flip lands a degree off,
    and INFINITE -- no fringes at all -- if it happens to land square. A beat
    derived from delta is anchored in the mask geometry instead. Rotation error
    still perturbs it (the two contributions add as vectors), but only by ~19%
    at half a degree, and it can never collapse to nothing.

    ``angle_deg`` orients both gratings together; it does not affect the beat.
    """
    if back_period_um <= 0.0:
        raise ValueError(f"back_period_um must be > 0 (got {back_period_um})")
    if back_period_um + delta_um <= 0.0:
        raise ValueError(f"delta_um {delta_um} cancels the {back_period_um} um carrier")
    h, w = silhouette.shape
    a = math.radians(angle_deg)
    # Projection of each cell centre onto the grating vector, in micrometres.
    # Built as an outer sum rather than a full meshgrid: two 1-D ramps instead
    # of two h x w float arrays, which matters on a plate-sized raster.
    u = (
        (np.arange(w, dtype=np.float32) * (math.cos(a) * cell_um))[None, :]
        + (np.arange(h, dtype=np.float32) * (math.sin(a) * cell_um))[:, None]
    )
    d = float(np.clip(duty, 0.01, 0.99))
    back = ((u / float(back_period_um)) % 1.0) < d
    front = (((u / float(back_period_um + delta_um)) % 1.0) < d) & silhouette
    return front, back
