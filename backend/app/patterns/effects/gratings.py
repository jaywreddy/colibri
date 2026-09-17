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


# --- budget accounting ------------------------------------------------------


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


# --- diffraction accent fab baker -------------------------------------------


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
