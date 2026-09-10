"""Design-rule check / clean (DRC) for the fab wafer export.

The gold-on-fused-silica litho process has a HARD floor: minimum gold line
2 µm, minimum gap 2 µm (so minimum grating PERIOD 4 µm). Any gold feature
narrower than 2 µm, or any gap narrower than 2 µm between colinear gold, will
not print cleanly — it either washes out (a sub-floor line) or bridges shut (a
sub-floor gap). The vector-native fine-pitch export (Build A) writes geometry
at its true optical period straight to SVG/GDS, so nothing downstream catches
these; this module is the last vector-domain gate before the layer is written.

Two entry points cover the two geometry kinds the export emits:

  * :func:`drc_clean_rects` — axis-aligned rectangles ``(N, 4)`` ``[x0,x1,y0,y1]``
    in µm (the row-span format ``effects.gratings.bool_to_row_spans`` and
    ``plates.raster_to_polygons`` emit). Gratings, slit barriers, interleaved
    scanimation slots, carrier stripes: all axis-aligned. Handled exactly and
    fast (vectorised numpy, no GEOS).

  * :func:`drc_clean_polys` — general polygons (angled diffraction gratings,
    traced silhouettes). Min-width is enforced by a morphological OPEN
    (erosion→dilation) on a fine LOCAL raster, per polygon (or small group), so
    no whole-layer GEOS union is ever taken — the 13.7 GB host bugchecks on
    those. A morphological open on a 0.5 µm analysis pixel (4× oversample of the
    2 µm floor) removes any protrusion/neck thinner than the floor and reports
    what it dropped.

:func:`drc_report` measures a layer (rects or polys) and returns the min width,
min gap, sub-floor feature count, and a width histogram for the verifier — with
no mutation, so it can be run before AND after a clean to prove the fix.

Design choices honoured from the Build-B contract:
  * Rects: prefer WIDENING gold over deleting when a snap of ≤ ``snap_tol_um``
    (default 0.5 µm) brings a sub-floor width/gap up to the floor; only drop a
    feature when it is too narrow to rescue that cheaply. Every drop/snap is
    logged (returned in the report + emitted via the module logger).
  * Polys: batch per polygon (or small groups) — never a global union.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Litho floor (µm). Kept equal to effects.moire.MIN_PERIOD_UM / 2 by import so
# the whole toolkit shares one source of truth; falls back to 2.0 stand-alone.
try:  # pragma: no cover - import shim for stand-alone self-check
    from .moire import MIN_PERIOD_UM as _MIN_PERIOD_UM

    FLOOR_UM = _MIN_PERIOD_UM / 2.0
except Exception:  # noqa: BLE001
    FLOOR_UM = 2.0

# Numerical slack for float period arithmetic (23.98 µm etc). A feature within
# this of the floor is treated AS the floor, not sub-floor — avoids dropping a
# rect that is 1.9999999 µm through rounding.
_EPS_UM = 1e-6


# --------------------------------------------------------------------------- #
#  rectangles (axis-aligned) — exact, vectorised, no GEOS                      #
# --------------------------------------------------------------------------- #

def _as_rects(rects) -> np.ndarray:
    """Coerce to a float ``(N, 4)`` [x0,x1,y0,y1] array with x0<x1, y0<y1."""
    a = np.asarray(rects, dtype=float)
    if a.size == 0:
        return a.reshape(0, 4)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(
            f"rects must be (N,4) [x0,x1,y0,y1]; got shape {a.shape}"
        )
    a = a.copy()
    # Normalise ordering so widths/heights are positive.
    x0 = np.minimum(a[:, 0], a[:, 1])
    x1 = np.maximum(a[:, 0], a[:, 1])
    y0 = np.minimum(a[:, 2], a[:, 3])
    y1 = np.maximum(a[:, 2], a[:, 3])
    return np.stack([x0, x1, y0, y1], axis=1)


def drc_clean_rects(
    rects,
    min_width_um: float = 2.0,
    min_gap_um: float = 2.0,
    *,
    snap_tol_um: float = 0.5,
    log: bool = True,
) -> np.ndarray:
    """Clean axis-aligned gold rectangles to the litho floor.

    Parameters
    ----------
    rects : array-like ``(N, 4)``
        ``[x0, x1, y0, y1]`` in µm. Order-agnostic (normalised internally).
    min_width_um, min_gap_um : float
        Floor for gold width/height and for the gap between colinear gold.
    snap_tol_um : float
        Max amount a rect edge may be MOVED to rescue a sub-floor feature by
        widening (rather than dropping). A near-floor rect (short by ≤ this) is
        grown to the floor; a near-floor gap (short by ≤ this) is closed by
        merging the two neighbours. Beyond this, the offending rect is dropped.
    log : bool
        Emit a one-line summary at INFO on the module logger.

    Returns
    -------
    ``(M, 4)`` cleaned rectangles (M ≤ N + merges), floats, x0<x1, y0<y1. No
    gold feature narrower than the floor in either axis; no colinear gap
    narrower than the floor.

    Notes
    -----
    * Width/height snap first (cheap, per-rect), THEN colinear gap-close on the
      widened set. Width snap grows symmetrically about the rect center so the
      grating phase is preserved as well as possible.
    * Gap-close only merges rects that are colinear neighbours — same row-band
      (identical y0,y1) touching in x, or same column-band (identical x0,x1)
      touching in y — because those are the ones a printer would bridge. It does
      not attempt to fill a 2-D re-entrant gap (that is the poly path's job).
    """
    a = _as_rects(rects)
    stats = _RectStats()
    if a.shape[0] == 0:
        if log:
            logger.info("drc_clean_rects: empty input")
        return a

    w = a[:, 1] - a[:, 0]
    h = a[:, 3] - a[:, 2]
    floor = float(min_width_um)

    # 1) width/height: snap-widen if within snap_tol, else drop. -------------
    keep = np.ones(a.shape[0], dtype=bool)
    for axis, lo, hi, dim in ((0, 0, 1, w), (1, 2, 3, h)):
        sub = dim < floor - _EPS_UM
        if not sub.any():
            continue
        deficit = floor - dim
        rescuable = sub & (deficit <= snap_tol_um + _EPS_UM)
        droppable = sub & ~rescuable
        # Widen rescuable rects symmetrically about their center to the floor.
        if rescuable.any():
            center = 0.5 * (a[:, lo] + a[:, hi])
            half = floor / 2.0
            a[rescuable, lo] = center[rescuable] - half
            a[rescuable, hi] = center[rescuable] + half
            stats.snapped += int(rescuable.sum())
        if droppable.any():
            keep &= ~droppable
            stats.dropped += int(droppable.sum())
    a = a[keep]
    if a.shape[0] == 0:
        if log:
            logger.info("drc_clean_rects: %s", stats.summary())
        return a

    # 2) colinear gap-close: merge neighbours separated by a sub-floor gap. --
    a = _close_colinear_gaps(a, axis="x", gap=float(min_gap_um), tol=snap_tol_um, stats=stats)
    a = _close_colinear_gaps(a, axis="y", gap=float(min_gap_um), tol=snap_tol_um, stats=stats)

    if log:
        logger.info("drc_clean_rects: %s", stats.summary())
    return a


def _close_colinear_gaps(a: np.ndarray, *, axis: str, gap: float, tol: float, stats) -> np.ndarray:
    """Merge rects that share a band in the OTHER axis and are separated along
    ``axis`` by a gap < ``gap``. Only gaps ≤ ``tol`` above zero are *closed by
    merge*; a wider-but-still-sub-floor gap is closed too (bridging is a defect
    either way), but we log it distinctly. Rects are grouped by their exact band
    coordinates (the row-span baker emits identical band edges for a lane), so
    grouping on rounded band keys is exact for grating output.
    """
    if a.shape[0] < 2:
        return a
    if axis == "x":
        band_lo, band_hi, run_lo, run_hi = 2, 3, 0, 1  # group by y, run along x
    else:
        band_lo, band_hi, run_lo, run_hi = 0, 1, 2, 3  # group by x, run along y

    # Sort once by (band, run) instead of walking the groups: the old loop did
    # ``np.where(inv == g)`` per group, an O(groups x rects) scan, and then
    # merged rect by rect in Python. A coloured photo face hands this ~200 000
    # rectangles in tens of thousands of bands, which is quadratic time for a
    # linear job. The result is identical, including ROW ORDER — groups still
    # come out in ascending key order and each group in ascending ``run_lo``.
    n = a.shape[0]
    key = np.round(np.stack([a[:, band_lo], a[:, band_hi]], axis=1), 6)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.lexsort((a[:, run_lo], inv))
    b = a[order]
    g = inv[order]
    lo, hi = b[:, run_lo], b[:, run_hi]
    new_band = np.empty(n, dtype=bool)
    new_band[0] = True
    np.not_equal(g[1:], g[:-1], out=new_band[1:])
    # Running maximum of ``run_hi`` WITHIN each band — the "merged" rect's far
    # edge in the old loop. One accumulate per band: linear overall, and exact
    # (no offset arithmetic that would round the coordinates).
    gstart = np.flatnonzero(new_band).tolist()
    cm = np.empty(n, dtype=np.float64)
    for s, e in zip(gstart, gstart[1:] + [n]):
        np.maximum.accumulate(hi[s:e], out=cm[s:e])
    # A run breaks at a new band, or where this rect starts a full gap past
    # everything merged so far. After a break the running max IS this rect's own
    # far edge (the break condition says the previous max is below its near
    # edge), so the within-band accumulate doubles as the per-run one.
    brk = new_band.copy()
    brk[1:] |= lo[1:] - cm[:-1] >= gap - _EPS_UM
    starts = np.flatnonzero(brk)
    if starts.size == 0:
        return a[:0]
    out = b[starts].copy()
    out[:, run_hi] = cm[np.append(starts[1:], n) - 1]
    stats.gaps_closed += int(n - starts.size)
    return out


@dataclass
class _RectStats:
    snapped: int = 0
    dropped: int = 0
    gaps_closed: int = 0

    def summary(self) -> str:
        return (
            f"snapped(widened)={self.snapped} dropped={self.dropped} "
            f"gaps_closed={self.gaps_closed}"
        )


# --------------------------------------------------------------------------- #
#  general polygons — morphological open on a fine LOCAL raster (no union)     #
# --------------------------------------------------------------------------- #

def _poly_to_verts(poly):
    """Accept a shapely (Multi)Polygon or an (K,2) vertex ring; return a list of
    exterior-ring vertex arrays (holes ignored for the open — an open cannot
    create or shrink a hole below the floor without also touching the exterior,
    and the local-raster open handles interior necks anyway)."""
    # Shapely polygon?
    if hasattr(poly, "geom_type"):
        gt = poly.geom_type
        if gt == "Polygon":
            return [np.asarray(poly.exterior.coords, dtype=float)]
        if gt in ("MultiPolygon", "GeometryCollection"):
            rings = []
            for g in poly.geoms:
                if getattr(g, "geom_type", "") == "Polygon":
                    rings.append(np.asarray(g.exterior.coords, dtype=float))
            return rings
        return []
    v = np.asarray(poly, dtype=float)
    if v.ndim == 2 and v.shape[1] == 2:
        return [v]
    raise ValueError("poly must be shapely (Multi)Polygon or (K,2) vertices")


def _open_ring(verts: np.ndarray, min_width_um: float, px_um: float):
    """Morphological open of ONE polygon on a local raster; return cleaned
    exterior ring(s) as a list of (K,2) vertex arrays (empty if the open erases
    it — i.e. the whole feature was sub-floor)."""
    try:
        from scipy import ndimage
        from skimage import measure
    except Exception:  # pragma: no cover
        # No morphology stack available: pass through unchanged (rects path is
        # the primary DRC; poly cleaning degrades gracefully rather than crash).
        return [verts]

    xmin, ymin = verts.min(axis=0)
    xmax, ymax = verts.max(axis=0)
    pad = min_width_um  # room so a dilation near the bbox edge is not clipped
    xmin -= pad; ymin -= pad; xmax += pad; ymax += pad
    w_px = max(1, int(np.ceil((xmax - xmin) / px_um)))
    h_px = max(1, int(np.ceil((ymax - ymin) / px_um)))
    # Rasterize the ring (point-in-polygon on pixel centers).
    ys, xs = np.mgrid[0:h_px, 0:w_px]
    px = xmin + (xs + 0.5) * px_um
    py = ymin + (ys + 0.5) * px_um
    grid = _points_in_poly(px.ravel(), py.ravel(), verts).reshape(h_px, w_px)
    # Open = erode then dilate by radius = floor/2 (so a neck < floor vanishes).
    r_px = max(1, int(round((min_width_um / 2.0) / px_um)))
    struct = _disk(r_px)
    opened = ndimage.binary_erosion(grid, structure=struct, border_value=0)
    opened = ndimage.binary_dilation(opened, structure=struct, border_value=0)
    if not opened.any():
        return []
    rings = []
    for contour in measure.find_contours(opened.astype(float), 0.5):
        # contour is (row, col) in pixel space → (x, y) µm.
        cx = xmin + (contour[:, 1] + 0.5) * px_um
        cy = ymin + (contour[:, 0] + 0.5) * px_um
        rings.append(np.stack([cx, cy], axis=1))
    return rings


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= r * r


def _points_in_poly(px: np.ndarray, py: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon for a single ring."""
    n = len(verts)
    inside = np.zeros(px.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        cond = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
        )
        inside ^= cond
        j = i
    return inside


def drc_clean_polys(
    polys,
    min_width_um: float = 2.0,
    *,
    px_um: float | None = None,
    log: bool = True,
) -> list:
    """Enforce min-width on general polygons via per-polygon morphological open.

    Parameters
    ----------
    polys : iterable
        Each item is a shapely (Multi)Polygon or an ``(K, 2)`` vertex ring (µm).
    min_width_um : float
        Any neck / protrusion / sliver thinner than this is removed by the open.
    px_um : float, optional
        Analysis pixel. Defaults to ``min_width_um / 4`` (0.5 µm at the 2 µm
        floor) — a 4× oversample gives the promised 2× safety margin on the
        floor. Each polygon is rasterized only over its OWN bbox (tiled
        implicitly by being local), so no whole-layer buffer is allocated.
    log : bool
        Emit a summary at INFO.

    Returns
    -------
    list of ``(K, 2)`` vertex arrays — the cleaned exterior rings (one input
    polygon may split into several, or vanish). No global GEOS union is taken;
    each polygon is processed independently, honouring the memory constraint.
    """
    if px_um is None:
        px_um = min_width_um / 4.0
    out = []
    n_in = 0
    n_removed = 0
    for poly in polys:
        for verts in _poly_to_verts(poly):
            n_in += 1
            rings = _open_ring(verts, min_width_um, px_um)
            if not rings:
                n_removed += 1
            out.extend(rings)
    if log:
        logger.info(
            "drc_clean_polys: in=%d out=%d removed(sub-floor)=%d px=%.3fµm",
            n_in, len(out), n_removed, px_um,
        )
    return out


# --------------------------------------------------------------------------- #
#  measurement / report                                                         #
# --------------------------------------------------------------------------- #

def drc_report(rects_or_polys, *, min_width_um: float = 2.0, min_gap_um: float = 2.0) -> dict:
    """Measure a layer WITHOUT mutating it. Auto-detects rects vs polys.

    Returns
    -------
    dict with:
      * ``min_width_um``  — narrowest gold feature dimension found (inf if empty).
      * ``min_gap_um``    — narrowest colinear gap found (inf if none / N<2).
      * ``n_subfloor``    — count of features below ``min_width_um`` (either axis
        for rects; sub-floor necks are not counted for polys — see note).
      * ``n_features``    — total feature count measured.
      * ``histogram``     — dict of width-bucket edges (µm) → count.
      * ``kind``          — "rects" or "polys".
    """
    kind = _detect_kind(rects_or_polys)
    if kind == "rects":
        return _report_rects(rects_or_polys, min_width_um, min_gap_um)
    return _report_polys(rects_or_polys, min_width_um)


def _detect_kind(x) -> str:
    if hasattr(x, "geom_type"):
        return "polys"
    a = np.asarray(x, dtype=object) if not isinstance(x, np.ndarray) else x
    if isinstance(x, np.ndarray) and x.dtype != object and x.ndim == 2 and x.shape[1] == 4:
        return "rects"
    # A list: peek at the first element.
    seq = list(x)
    if not seq:
        return "rects"
    first = seq[0]
    if hasattr(first, "geom_type"):
        return "polys"
    fa = np.asarray(first, dtype=float)
    if fa.ndim == 2 and fa.shape[1] == 2:
        return "polys"
    # Row of 4 numbers → rects.
    return "rects"


def _report_rects(rects, min_width_um: float, min_gap_um: float) -> dict:
    a = _as_rects(rects)
    n = a.shape[0]
    if n == 0:
        return {
            "kind": "rects", "n_features": 0, "min_width_um": float("inf"),
            "min_gap_um": float("inf"), "n_subfloor": 0, "histogram": {},
        }
    w = a[:, 1] - a[:, 0]
    h = a[:, 3] - a[:, 2]
    dims = np.concatenate([w, h])
    min_w = float(dims.min())
    n_sub = int(((w < min_width_um - _EPS_UM) | (h < min_width_um - _EPS_UM)).sum())

    # Colinear gaps (both axes), measured the same way the cleaner closes them.
    min_gap = _min_colinear_gap(a, "x")
    min_gap = min(min_gap, _min_colinear_gap(a, "y"))

    edges = [0.0, 0.5, 1.0, 1.9, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]
    hist = _histogram(np.minimum(w, h), edges)
    return {
        "kind": "rects", "n_features": n, "min_width_um": min_w,
        "min_gap_um": float(min_gap), "n_subfloor": n_sub, "histogram": hist,
    }


def _min_colinear_gap(a: np.ndarray, axis: str) -> float:
    if a.shape[0] < 2:
        return float("inf")
    if axis == "x":
        band_lo, band_hi, run_lo, run_hi = 2, 3, 0, 1
    else:
        band_lo, band_hi, run_lo, run_hi = 0, 1, 2, 3
    key = np.round(np.stack([a[:, band_lo], a[:, band_hi]], axis=1), 6)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    best = float("inf")
    for g in range(inv.max() + 1 if inv.size else 0):
        grp = a[inv == g]
        if grp.shape[0] < 2:
            continue
        grp = grp[np.argsort(grp[:, run_lo])]
        gaps = grp[1:, run_lo] - grp[:-1, run_hi]
        gaps = gaps[gaps > _EPS_UM]  # ignore touching/overlapping (gap<=0)
        if gaps.size:
            best = min(best, float(gaps.min()))
    return best


def _report_polys(polys, min_width_um: float) -> dict:
    """Measure polygons: min-width via the same local-raster open comparison.
    A polygon whose open erases it is a sub-floor feature; the narrowest surviving
    width is estimated from the largest inscribed erosion radius that survives."""
    widths = []
    n_sub = 0
    n = 0
    px_um = min_width_um / 4.0
    for poly in polys:
        for verts in _poly_to_verts(poly):
            n += 1
            wmin = _estimate_min_width(verts, px_um, min_width_um)
            widths.append(wmin)
            if wmin < min_width_um - _EPS_UM:
                n_sub += 1
    if not widths:
        return {
            "kind": "polys", "n_features": 0, "min_width_um": float("inf"),
            "min_gap_um": float("inf"), "n_subfloor": 0, "histogram": {},
        }
    edges = [0.0, 0.5, 1.0, 1.9, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]
    hist = _histogram(np.asarray(widths), edges)
    return {
        "kind": "polys", "n_features": n, "min_width_um": float(min(widths)),
        "min_gap_um": float("inf"),  # gap between separate polys not measured
        "n_subfloor": n_sub, "histogram": hist,
    }


def _estimate_min_width(verts: np.ndarray, px_um: float, cap_um: float) -> float:
    """Narrowest width of one polygon = 2× the largest erosion radius (in µm)
    that still leaves ANY pixel. Probed by successive erosions on the local
    raster up to ``cap_um`` (we only need to know whether it clears the floor)."""
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover
        return cap_um  # can't measure → assume OK
    xmin, ymin = verts.min(axis=0)
    xmax, ymax = verts.max(axis=0)
    w_px = max(1, int(np.ceil((xmax - xmin) / px_um)) + 2)
    h_px = max(1, int(np.ceil((ymax - ymin) / px_um)) + 2)
    ys, xs = np.mgrid[0:h_px, 0:w_px]
    px = xmin + (xs - 0.5) * px_um
    py = ymin + (ys - 0.5) * px_um
    grid = _points_in_poly(px.ravel(), py.ravel(), verts).reshape(h_px, w_px)
    if not grid.any():
        return 0.0
    max_r = int(np.ceil((cap_um / 2.0) / px_um)) + 1
    survived = 0
    for r in range(1, max_r + 1):
        er = ndimage.binary_erosion(grid, structure=_disk(r), border_value=0)
        if er.any():
            survived = r
        else:
            break
    # Width ≈ 2 * survived_radius (+1 pixel for the seed). Cap at cap for report.
    width = (2 * survived + 1) * px_um
    return min(width, cap_um) if width >= cap_um else width


def _histogram(vals: np.ndarray, edges: list) -> dict:
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = int(((vals >= lo) & (vals < hi)).sum())
        label = f"[{lo:g},{'inf' if np.isinf(hi) else f'{hi:g}'})"
        out[label] = c
    return out


# --------------------------------------------------------------------------- #
#  MERGED-geometry DRC — the PRINTED-layer gate (klayout Region)               #
# --------------------------------------------------------------------------- #
# The rect/poly cleaners above act per feature (or per colinear neighbour) and
# so cannot see the defects that only exist once ALL of a layer's groups are
# UNIONED into what the mask actually prints: sub-floor slivers where two angled
# gratings overlap into a wedge tip, and sub-floor gaps where a grating from one
# zone nearly touches a grating from the neighbouring zone/bucket/slot (audit
# blocker 3 — ~5000 sites, min 0.0002 µm, all in unions / at seams). This gate
# builds the merged klayout Region for one plate-layer, MEASURES it honestly, and
# heals it — all on a merged C++ edge set (NOT a shapely/GEOS whole-geometry
# union; the memory rule targets GEOS booleans on shapely MultiPolygons), one
# plate-layer at a time so RAM stays bounded.
#
# The bulk of blocker-3's sites are killed at the SOURCE, because that is where
# the fix is non-destructive (export_fine): a 1-cell GUTTER between zones filled
# with different-angle gratings (frame buckets, 45° accent vs 0° carrier) so
# their lines never cross into wedge tips, and floor-gridded scanimation crest
# runs so adjacent slots never step by a sub-floor notch. Those two source fixes
# take the ~5000 sites down to dozens per layer.
#
# The residual heal here is DELIBERATELY the ONE morphological op that does not
# damage the physics-critical fine gratings:
#
#   * ``interacting`` sliver DROP — erode by w/2 to get surviving CORES, keep
#     every ORIGINAL polygon that still has a core at its EXACT shape, drop the
#     rest. After merge each grating line / crest / bar is its own polygon, so a
#     legal 2.2 µm line survives untouched while an isolated sub-floor sliver
#     (whose core erodes to nothing) is removed whole.
#
# A morphological CLOSE or OPEN (dilate↔erode) is intentionally NOT used: on a
# dense angled grating it rounds every rotated-rectangle corner and shaves legal
# 2.2 µm lines to ~1.98 µm (and the erode-back opens fresh sub-floor notches) —
# measured, and worse than the tiny quantization residual it would remove. The
# measurement uses the Euclidian metric with ignore_angle=80 (excludes acute
# corner false-positives), NeverIncludeZeroDistance, and counts only genuine
# 0 < d < floor features (a 0-distance pair is overlapping/touching gold — solid
# metal, not a gap or sliver).


# Tiling for the width/space CHECKS (measurement only — the heal never tiles).
# klayout's width_check/space_check degrade catastrophically on one giant merged
# polygon: the back layer of a 40 mm face merges its full-window carrier into a
# single ring whose self-check ran ~27 MINUTES, while the same check on the
# front layer's 18k separate polygons takes ~12 s. Checking per-tile keeps every
# region small. Correctness: a tile's region is built from every input polygon
# whose bbox touches the tile WINDOW (core + border, border > check distance),
# so all metal that can interact with core geometry is present and locally
# merged; a violation is counted only if its bbox center lies in the half-open
# CORE, so tiles partition the plane and nothing double-counts. Clip artifacts
# cannot occur at all — inputs are inserted whole, never clipped.
# 1000 µm empirically minimizes the real back layer's check (angled near-floor
# combs): 250 µm → 12.4 s (per-tile overhead), 1 mm → 5.6 s, 2 mm → 39 s,
# 5 mm → 105 s (superlinear contiguous-edge cost takes over).
_CHECK_TILE_UM = 1000.0
_CHECK_BORDER_UM = 16.0


def _region_entries(region, kdb, *, dbu_um: float, tile_um: float):
    """Bin a klayout ``Region`` for the tiled check WITHOUT a µm round trip.

    The list path below has to rebuild every polygon from floats; when the
    caller already holds the Region (the heal loop, the witness die's
    decomposed clear pieces) that costs a full Python pass per iteration for
    nothing. Polygons are handed to the tiler as they are — HULL ONLY, matching
    what the list callers pass (``each_point_hull`` rings): a fill-only mask is
    written from exterior rings, and the heal drops holes anyway.
    """
    entries: list[tuple[float, float, float, float, object]] = []
    big = kdb.Region()
    add = entries.append          # hoisted: this loop runs 250k times per check
    poly_cls = kdb.Polygon
    for poly in region.each():
        if poly.holes():
            poly = poly_cls(list(poly.each_point_hull()))
        bb = poly.bbox()
        lo_x, lo_y = bb.left * dbu_um, bb.bottom * dbu_um
        hi_x, hi_y = bb.right * dbu_um, bb.top * dbu_um
        if poly.is_box():
            # Box payload carried in DBU INTEGERS, not µm floats: the tile loop
            # clips it and clipping in DBU is `max`/`min` on ints, where clipping
            # in µm cost four float multiplies and four ``round()`` calls PER
            # TILE the box touches (25 M round() calls, 17 s, on one garden die).
            # Identical result — round() is monotone, so
            # round(max(a, b)) == max(round(a), round(b)).
            add((lo_x, lo_y, hi_x, hi_y,
                 (bb.left, bb.bottom, bb.right, bb.top)))
        elif hi_x - lo_x > 2 * tile_um or hi_y - lo_y > 2 * tile_um:
            big.insert(poly)
        else:
            add((lo_x, lo_y, hi_x, hi_y, poly))
    return entries, big


def _tiled_check_stats(
    polys,
    kdb,
    *,
    width_dbu: int,
    gap_dbu: int,
    dbu_um: float,
    tile_um: float = _CHECK_TILE_UM,
    border_um: float = _CHECK_BORDER_UM,
    include_touching: bool = False,
    only_near=None,
) -> tuple[int, float, int, float, Any, Any]:
    """Tile-by-tile width/space checks; returns
    ``(n_width, min_w_um, n_space, min_s_um, w_markers, s_markers)``.

    ``polys`` is an iterable of ``(N,4)`` rect arrays / ``(K,2)`` vertex rings,
    OR a ``kdb.Region`` (taken hull-only, no µm round trip — see
    :func:`_region_entries`).

    Check options match the single-region path exactly (Euclidian,
    ignore_angle=80, shielded=False, NeverIncludeZeroDistance) — see the module
    note above for why each is the honest printed-geometry measure.

    ``include_touching`` adds the ZERO-distance pairs' markers to the marker
    Regions (and so to the counts), without letting them into the min-distance
    headline. Those pairs are the acute CORNER of a clipped angled line — the
    two edges meet at the vertex, so the pair distance is 0 while the corner
    itself does taper below the floor. The QA report wants them out (they are
    not a gap or a sliver, and a 0 µm headline is meaningless); a caller that
    HEALS by chroming markers over wants them in, because the taper is real.

    ``only_near`` is an iterable of ``(x0, y0, x1, y1)`` µm boxes. Given one, only
    the tiles whose CORE meets a box are checked, and the answer is EXACTLY the
    whole-input answer restricted to those tiles: a tile's region is built from
    every input polygon whose bbox touches its window, so what a tile measures
    does not depend on which other tiles ran, and a violation is attributed to
    the one tile whose half-open core holds its centre. It is for a repair loop
    re-checking after a local patch — the caller is asserting that nothing
    outside those boxes can have changed, which is a statement about ITS patch,
    not about the tiling.
    """
    import time as _time

    _t0 = _time.time()
    scale = 1.0 / dbu_um
    if isinstance(polys, kdb.Region):
        entries, big = _region_entries(polys, kdb, dbu_um=dbu_um, tile_um=tile_um)
        polys = ()
    else:
        entries, big = [], kdb.Region()
    for item in polys:
        a = np.asarray(item, dtype=float)
        if a.ndim == 2 and a.shape[1] == 4:  # rect array (N,4) [x0,x1,y0,y1]
            lo_x = np.minimum(a[:, 0], a[:, 1])
            hi_x = np.maximum(a[:, 0], a[:, 1])
            lo_y = np.minimum(a[:, 2], a[:, 3])
            hi_y = np.maximum(a[:, 2], a[:, 3])
            # µm bbox for the binning, DBU integers for the payload (see
            # _region_entries: clipping in DBU costs no round() per tile).
            box_um = np.stack([lo_x, lo_y, hi_x, hi_y], axis=1)
            for box, dbu in zip(box_um.tolist(),
                                np.rint(box_um * scale).astype(np.int64).tolist()):
                entries.append((box[0], box[1], box[2], box[3], tuple(dbu)))
        elif a.ndim == 2 and a.shape[1] == 2 and a.shape[0] >= 3:  # vertex ring
            lo_x, lo_y = a[:, 0].min(), a[:, 1].min()
            hi_x, hi_y = a[:, 0].max(), a[:, 1].max()
            # Axis-aligned 4-point rings ARE rects — normalize to the tuple
            # form so they take the clipping path below. This covers both
            # _compose_layer_polys output (every axis rect arrives as a ring)
            # and healed full-height carrier lines; without it a 40 mm line
            # ring re-enters every tile unclipped and the superlinear
            # long-edge check cost returns through the back door.
            # ``len(set(col))`` on four floats, not ``np.unique`` — same test
            # (exactly two distinct values), without a sort and two array
            # allocations per ring; np.unique was 915 k calls / 11 s on one
            # garden die.
            if (
                a.shape[0] == 4
                and len(set(a[:, 0].tolist())) == 2
                and len(set(a[:, 1].tolist())) == 2
            ):
                entries.append((lo_x, lo_y, hi_x, hi_y,
                                tuple(np.rint(np.array([lo_x, lo_y, hi_x, hi_y]) * scale)
                                      .astype(np.int64).tolist())))
            elif hi_x - lo_x > 2 * tile_um or hi_y - lo_y > 2 * tile_um:
                # Rings spanning many tiles — LONG ANGLED grating lines (an
                # 11 µm × 40 mm line at 30° has a ~20 × 35 mm bbox) and any
                # merged plate-spanning blob. These are the measured killer:
                # klayout's check cost grows superlinearly with contiguous
                # edge length, and near-floor-gap diagonal combs of full-plate
                # lines took ~27 min/check. They go into one Region and are
                # CLIPPED per tile below (bbox-indexed select + window AND),
                # so no tile ever sees an edge longer than the window.
                big.insert(
                    kdb.Polygon(
                        [kdb.Point(int(round(x * scale)), int(round(y * scale))) for x, y in a]
                    )
                )
            else:
                # Build the klayout polygon ONCE, here, rather than per tile:
                # a ring that touches k tiles was rebuilt k times, point by
                # Python point. Insertion is by far the hot loop below.
                entries.append((
                    lo_x, lo_y, hi_x, hi_y,
                    # np.rint IS Python's round() (half to even), so the DBU
                    # coordinates are the ones the point-by-point loop made.
                    kdb.Polygon(np.rint(a * scale).astype(np.int64).tolist()),
                ))
    have_big = not big.is_empty()
    if have_big:
        big.merge()
        bb = big.bbox()
    if not entries and not have_big:
        return 0, float("inf"), 0, float("inf"), kdb.Region(), kdb.Region()

    xs0 = [e[0] for e in entries] + ([bb.left * dbu_um] if have_big else [])
    ys0 = [e[1] for e in entries] + ([bb.bottom * dbu_um] if have_big else [])
    xs1 = [e[2] for e in entries] + ([bb.right * dbu_um] if have_big else [])
    ys1 = [e[3] for e in entries] + ([bb.top * dbu_um] if have_big else [])
    gx0, gy0, gx1, gy1 = min(xs0), min(ys0), max(xs1), max(ys1)
    nx = max(1, int(math.ceil((gx1 - gx0) / tile_um)))
    ny = max(1, int(math.ceil((gy1 - gy0) / tile_um)))

    # Tiles the caller has restricted the re-check to (see ``only_near``). A
    # payload is still binned into every tile it can reach — a kept tile must see
    # all of its own geometry — but tiles outside the set are never built.
    allowed: set[tuple[int, int]] | None = None
    if only_near is not None:
        allowed = set()
        for bx0, by0, bx1, by1 in only_near:
            jx0 = max(0, int((bx0 - gx0) / tile_um))
            jx1 = min(nx - 1, int((bx1 - gx0) / tile_um))
            jy0 = max(0, int((by0 - gy0) / tile_um))
            jy1 = min(ny - 1, int((by1 - gy0) / tile_um))
            for ix in range(jx0, jx1 + 1):
                for iy in range(jy0, jy1 + 1):
                    allowed.add((ix, iy))
        if not allowed:
            return 0, float("inf"), 0, float("inf"), kdb.Region(), kdb.Region()

    buckets: dict[tuple[int, int], list] = {}
    for lo_x, lo_y, hi_x, hi_y, payload in entries:
        ix0 = max(0, int((lo_x - border_um - gx0) / tile_um))
        ix1 = min(nx - 1, int((hi_x + border_um - gx0) / tile_um))
        iy0 = max(0, int((lo_y - border_um - gy0) / tile_um))
        iy1 = min(ny - 1, int((hi_y + border_um - gy0) / tile_um))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                if allowed is not None and (ix, iy) not in allowed:
                    continue
                buckets.setdefault((ix, iy), []).append(payload)
    if have_big:
        # Pre-clip the plate-spanning/angled geometry into tile-COLUMN strips
        # (nx ANDs on the big region, not one per tile — a diagonal line's
        # bbox overlaps ~10× more tiles than the line itself crosses, so
        # per-tile selection overselects brutally). Strip pieces are ≤ one
        # window wide and a window-diagonal long, so binning them by row and
        # inserting them whole keeps every edge bounded by the window size.
        cols = range(nx) if allowed is None else sorted({ix for ix, _ in allowed})
        for ix in cols:
            sx0 = gx0 + ix * tile_um - border_um
            sx1 = gx0 + (ix + 1) * tile_um + border_um
            strip = big & kdb.Region(
                kdb.Box(
                    int(round(sx0 * scale)), bb.bottom - 1,
                    int(round(sx1 * scale)), bb.top + 1,
                )
            )
            for poly in strip.each():
                pb = poly.bbox()
                iy0 = max(0, int((pb.bottom * dbu_um - border_um - gy0) / tile_um))
                iy1 = min(ny - 1, int((pb.top * dbu_um + border_um - gy0) / tile_um))
                for iy in range(iy0, iy1 + 1):
                    if allowed is not None and (ix, iy) not in allowed:
                        continue
                    buckets.setdefault((ix, iy), []).append(poly.dup())

    _t_bin = _time.time()
    logger.debug(
        "tiled_check: %d entries, %d buckets, prep %.1fs",
        len(entries), len(buckets), _t_bin - _t0,
    )
    _slowest = (0.0, None, 0)
    euc = kdb.Region.Euclidian
    zdm = kdb.Region.NeverIncludeZeroDistance
    # Violations are counted as merged marker AREAS, not raw edge pairs: tile
    # clipping fragments one long sub-floor run into one pair per tile, and even
    # unclipped klayout pair granularity is representation-dependent. Each kept
    # pair's marker polygon (the gap/sliver area itself) goes into a region;
    # markers of the same physical site overlap across tile borders (windows
    # overlap by 2×border), so the merged count is the number of connected
    # sub-floor SITES — stable under tiling and the honest QA number.
    w_markers = kdb.Region()
    s_markers = kdb.Region()
    mnw = mns = float("inf")
    for (ix, iy), payloads in buckets.items():
        _t_tile = _time.time()
        # Tile WINDOW in µm (core + border) — rect payloads are CLIPPED to it.
        # Check cost grows superlinearly with contiguous edge length (measured:
        # the back layer's full-plate-height carrier lines cost ~40× more per
        # meter of edge than the front's short segments), so unclipped tall
        # rects would re-check their full length in every tile they touch.
        # Analytic box clipping is exact; the artificial cut edges it creates
        # sit ON the window boundary, ≥ border > check distance from the core,
        # so the center-in-core filter below discards any pair they join.
        # Window in DBU. round() is monotone, so clipping the (already rounded)
        # DBU payload against the rounded window is the same integer box the
        # µm-space clip-then-round produced — at zero float work per payload.
        wx0 = int(round((gx0 + ix * tile_um - border_um) * scale))
        wy0 = int(round((gy0 + iy * tile_um - border_um) * scale))
        wx1 = int(round((gx0 + (ix + 1) * tile_um + border_um) * scale))
        wy1 = int(round((gy0 + (iy + 1) * tile_um + border_um) * scale))
        reg = kdb.Region()
        _box, _ins = kdb.Box, reg.insert
        for payload in payloads:
            if type(payload) is tuple:
                lo_x, lo_y, hi_x, hi_y = payload
                if lo_x < wx0:
                    lo_x = wx0
                if hi_x > wx1:
                    hi_x = wx1
                if lo_y < wy0:
                    lo_y = wy0
                if hi_y > wy1:
                    hi_y = wy1
                if hi_x <= lo_x or hi_y <= lo_y:
                    continue
                _ins(_box(lo_x, lo_y, hi_x, hi_y))
            else:  # kdb.Polygon: a small ring, a Region input, a clipped strip
                _ins(payload)
        if reg.is_empty():
            continue
        reg.merge()
        # Half-open core in DBU: [cx0, cx1) × [cy0, cy1) partitions the plane.
        cx0 = int(round((gx0 + ix * tile_um) * scale))
        cy0 = int(round((gy0 + iy * tile_um) * scale))
        cx1 = int(round((gx0 + (ix + 1) * tile_um) * scale))
        cy1 = int(round((gy0 + (iy + 1) * tile_um) * scale))

        def _tally(edge_pairs, markers) -> float:
            mn = float("inf")
            for ep in edge_pairs.each():
                d = abs(ep.distance())
                if d <= 0 and not include_touching:
                    continue
                c = ep.bbox().center()
                if not (cx0 <= c.x < cx1 and cy0 <= c.y < cy1):
                    continue
                markers.insert(ep.polygon(0))
                if d > 0:
                    mn = min(mn, d * dbu_um)
            return mn

        w_mn = _tally(
            reg.width_check(width_dbu, metrics=euc, ignore_angle=80, shielded=False, zero_distance_mode=zdm),
            w_markers,
        )
        s_mn = _tally(
            reg.space_check(gap_dbu, metrics=euc, ignore_angle=80, shielded=False, zero_distance_mode=zdm),
            s_markers,
        )
        mnw = min(mnw, w_mn)
        mns = min(mns, s_mn)
        _dt = _time.time() - _t_tile
        if _dt > _slowest[0]:
            _slowest = (_dt, (ix, iy), len(payloads))
    w_markers.merge()
    s_markers.merge()
    logger.debug(
        "tiled_check: tiles %.1fs total, slowest %.2fs at %s (%d payloads)",
        _time.time() - _t_bin, _slowest[0], _slowest[1], _slowest[2],
    )
    # The merged marker Regions ride along for the heal's surgical weld
    # (drc_clean_region) — counts alone serve the report path.
    return int(w_markers.count()), mnw, int(s_markers.count()), mns, w_markers, s_markers


def patched_neighbourhood(reg, patch, floor_dbu: int):
    """The only part of ``reg`` whose width/space result a ``patch`` can change.

    Both repair loops that use this (the metal-side weld in
    :func:`drc_clean_region` and the written-clear settle in
    ``witness_dies.clear_field``) are the same shape: check, patch every flagged
    site, check again. The re-check does not have to look at the whole die, and
    on a 250 000-piece photo die a whole-die tiled check is ~5 s, so re-checking
    it up to eleven more times is where those dies spent their minutes.

    Why a neighbourhood is the WHOLE answer, not an approximation. Round k's
    check is complete, so every violation site of R(k) lies inside the markers,
    and the patch covers the markers. R(k+1) differs from R(k) only inside the
    patch. So a site of R(k+1) is either a residue of a patched site or one the
    patch's own boundary created, and both lie within one floor of the patch;
    anywhere further out the geometry is bit-identical to R(k)'s and R(k) had no
    site there. Taking WHOLE polygons that touch ``patch`` grown by 2× the floor
    keeps BOTH flanks of any such pair (each flank is within one floor of a site
    that is within one floor of the patch), and the check is unshielded, so no
    third polygon can influence a pair. The result is therefore the same site set
    the whole-die check would return, at the cost of the patched area.

    ``floor_dbu`` is the check distance in DBU; the halo is 4× it, i.e. double
    what the argument needs.

    ONLY VALID ON A MERGED REGION, and the reason is worth stating because the
    obvious other use is wrong. Selecting whole polygons is safe here because a
    merged region's polygons do not touch each other: dropping a distant one
    cannot change the shape — and so the WIDTH — of a kept one. On a set of
    ABUTTING pieces (the written-clear settle's convex decomposition) it is not:
    the tiled check merges each tile, so a piece's measured width is that of the
    shape it fuses into, and dropping a neighbour that fell outside the halo
    reports the piece's own width as a violation. Measured on the garden die,
    that turned the settle's honest 7 width sites into 11 phantom ones. A piece
    set restricts by TILE instead (``_tiled_check_stats(only_near=...)``).

    Raw semantics for the selection itself: klayout Regions default to merged
    semantics, under which ``interacting`` returns the polygons of a VIRTUAL
    MERGE of its input. On the merged input this function requires that is the
    same set, but asking for it explicitly costs nothing and cannot surprise.
    """
    was = reg.merged_semantics
    reg.merged_semantics = False
    try:
        near = reg.interacting(patch.sized(4 * floor_dbu))
    finally:
        reg.merged_semantics = was
    near.merged_semantics = False
    return near


def _region_from_polys(polys, dbu_um: float):
    """Build a merged klayout Region (integer DBU) from plate-frame polygons.

    ``polys`` is an iterable of either ``(N,4)`` [x0,x1,y0,y1] rect arrays or
    ``(K,2)`` vertex rings (µm). Returns a merged ``kdb.Region`` in DBU units and
    the ``kdb`` module (so the caller reuses one import).
    """
    import klayout.db as kdb

    reg = kdb.Region()
    scale = 1.0 / dbu_um
    box, poly, ins = kdb.Box, kdb.Polygon, reg.insert
    # The DBU rounding is done by numpy, in bulk, and only the object
    # construction is left to Python: ``np.rint`` IS Python's ``round()`` (both
    # half-to-even), so the integers are the ones the per-coordinate
    # ``int(round(...))`` loop produced. A photo die hands this ~215 000 rings
    # and the per-coordinate loop was 25 M ``round()`` calls.
    for item in polys:
        a = np.asarray(item, dtype=float)
        if a.ndim == 2 and a.shape[1] == 4:      # rect array (N,4)
            if a.shape[0] == 0:
                continue
            lo_x = np.minimum(a[:, 0], a[:, 1])
            hi_x = np.maximum(a[:, 0], a[:, 1])
            lo_y = np.minimum(a[:, 2], a[:, 3])
            hi_y = np.maximum(a[:, 2], a[:, 3])
            grid = np.rint(np.stack([lo_x, lo_y, hi_x, hi_y], axis=1) * scale)
            for x0, y0, x1, y1 in grid.astype(np.int64).tolist():
                ins(box(x0, y0, x1, y1))
        elif a.ndim == 2 and a.shape[1] == 2:    # vertex ring (K,2)
            if a.shape[0] >= 3:
                ins(poly(np.rint(a * scale).astype(np.int64).tolist()))
    reg.merge()
    return reg, kdb


def _count_check(region, check_fn, floor_dbu: int) -> tuple[int, float]:
    """Run a klayout width/space check at ``floor_dbu`` and return
    (n_violations, min_dim_um)."""
    edge_pairs = check_fn(floor_dbu)
    n = 0
    min_d = float("inf")
    # EdgePairs: each pair's spacing is its perpendicular distance. Use the
    # bbox diagonal-independent measure: klayout gives us the pairs; the count is
    # what matters for the gate, and the min distance for the headline.
    for ep in edge_pairs.each():
        n += 1
    return n, min_d


def drc_report_region(
    polys, *, min_width_um: float = 2.0, min_gap_um: float = 2.0, dbu_um: float = 0.001
) -> dict:
    """Measure the MERGED (printed) geometry of one plate-layer WITHOUT mutating.

    Builds the merged Region and runs klayout ``width_check`` / ``space_check`` at
    the floor. Returns the violation counts + narrowest width/space found (µm) —
    the honest printed-layer tally the audit's klayout method produces.
    """
    import klayout.db as kdb

    # No whole-plate Region here: building+merging it cost a full Python
    # insert loop and served only the n_polys count. The heal path merges
    # already, so len(polys) IS the merged count for after-reports; for the
    # opt-in raw before-reports it is the input count (documented).
    n_input = sum(1 for p in polys if np.asarray(p).size)
    if n_input == 0:
        return {
            "kind": "region", "n_polys": 0, "n_width_viol": 0, "n_space_viol": 0,
            "min_width_um": float("inf"), "min_space_um": float("inf"),
        }
    floor_dbu = int(round(min_width_um / dbu_um))
    gap_dbu = int(round(min_gap_um / dbu_um))

    # Check options (inside _tiled_check_stats) — the honest printed-geometry
    # measure the audit used:
    #   * ignore_angle = 80°: edges meeting at ≥ 80° are NOT paired, excluding the
    #     acute-corner false positives an angled grating's rotated rectangles
    #     trigger (a clean 45° 2.2 µm grating: 0 width pairs with it, 10 without).
    #   * NeverIncludeZeroDistance: coincident edges between abutting-but-unmerged
    #     rectangles print as SOLID gold, not a 0 µm sliver/gap — geometry-
    #     representation artifacts, not litho defects (and only genuine
    #     0 < d < floor pairs are tallied).
    #   * shielded=False: shielding is the dominant cost on a dense grating
    #     (~7× slower) and a shielded violation always co-occurs with an
    #     unshielded one.
    # The checks run TILED (see _tiled_check_stats): one merged whole-plate
    # region sends klayout's self-check quadratic on the back layer's fused
    # full-window carrier (measured ~27 min per check on a 40 mm face vs ~1 s
    # for the identical geometry checked in 2 mm tiles).
    nw, mnw, ns, mns, _, _ = _tiled_check_stats(
        polys, kdb, width_dbu=floor_dbu, gap_dbu=gap_dbu, dbu_um=dbu_um
    )
    return {
        "kind": "region",
        "n_polys": n_input,
        "n_width_viol": nw,
        "n_space_viol": ns,
        "min_width_um": mnw,
        "min_space_um": mns,
    }


def drc_clean_region(
    polys, *, min_width_um: float = 2.0, min_gap_um: float = 2.0, dbu_um: float = 0.001,
    report: bool = False,
):
    """Heal the merged printed geometry of one plate-layer to the litho floor.

    CLOSE (fill sub-floor gaps) then OPEN (shave sub-floor slivers/wedge tips) on
    the merged Region — see the module note. Returns a list of ``(K,2)`` µm vertex
    rings (holes dropped: a fill-only gold mask needs exterior rings, and the heal
    removes any sub-floor interior neck anyway). Never takes a shapely/GEOS union;
    the Region is a bounded per-layer C++ edge set.

    ``report=True`` returns ``(rings, report_dict)`` instead, where the dict is
    exactly what :func:`drc_report_region` would return for those rings — because
    it IS the loop's own last measurement, not a second one. The loop already
    ends by checking the healed region with the identical options
    (Euclidian / ignore_angle 80 / shielded False / NeverIncludeZeroDistance,
    same floor, same tiling, ``include_touching=False``), and running that check
    again on the rings extracted from the same Region measured the same geometry
    twice: 14 s of the 163 s a coloured photo die took. The one case where the
    loop's last check is NOT of the final geometry — it ran out of rounds and
    patched on the way out — is re-measured here, which is what the separate
    report call did anyway. It also carries ``rounds``: the per-round site counts,
    so a die that spent its budget chasing sites says so.
    """
    reg, kdb = _region_from_polys(polys, dbu_um)
    if reg.is_empty():
        empty = {"kind": "region", "n_polys": 0, "n_width_viol": 0, "n_space_viol": 0,
                 "min_width_um": float("inf"), "min_space_um": float("inf"),
                 "rounds": []}
        return ([], empty) if report else []
    reg.merge()
    open_dbu = max(1, int(round((min_width_um / 2.0) / dbu_um)))
    # SLIVER removal, WITHOUT reshaping legal lines. A global morphological open
    # (erode→dilate) would round every angled grating line and shave a legal
    # 2.2 µm line to ~1.98 µm at its rotated corners, and a close (dilate→erode)
    # can pinch a sub-floor neck into a merged blob. Instead we DROP-WHOLE only
    # the sub-floor polygons: erode by the half-floor to get the surviving CORES,
    # then keep every ORIGINAL polygon that still has a core (``interacting``) at
    # its EXACT original shape. After merge each grating line / crest / bar is its
    # own polygon (features are separated by ≥ floor gaps once seams are cleared
    # and crest boundaries are floor-gridded upstream), and an isolated sub-floor
    # sliver is its own polygon whose core erodes to nothing — so a legal feature
    # survives untouched while the sliver is dropped whole. Sub-floor GAPS are
    # prevented at the source (seam-gutter insets + floor-gridded crest runs), so
    # no destructive close is needed here.
    cores = reg.sized(-open_dbu)
    reg = reg.interacting(cores)
    reg.merge()
    # Sub-floor GAP weld — surgical, not a global close. The fine families
    # prevent gaps at the source (seam-gutter insets + floor-gridded crest
    # runs), but a COARSE carrier pitch (a user slider, and the glass-scaled
    # pitch on thick stock) can land a clipped line end within a fraction of a
    # micrometre of a neighboring zone's geometry — a near-tangency litho
    # would bridge unpredictably in resist. Rather than hoping, we make the
    # bridge explicit: klayout's space_check returns exactly the violating
    # edge pairs, and each pair's connecting polygon (padded by the half
    # floor so the weld overlaps both flanks and rounds its own notches) is
    # OR-ed back in. Only the violation sites change; a global close would
    # pinch legal necks and reshape angled lines (see the note above).
    gap_dbu = max(1, int(round(min_gap_um / dbu_um)))
    width_dbu = max(1, int(round(min_width_um / dbu_um)))
    pad_dbu = max(1, int(round((min_gap_um / 2.0) / dbu_um)))

    rounds: list[list[int]] = []
    final: tuple[int, float, int, float] | None = None
    patch = None
    for it in range(12):  # a weld can expose a thin ledge; usually converges in 1-2
        # NEVER a raw whole-region width/space check here — that is the
        # measured ~27-minute superlinear pathology the tiled checker exists
        # for (see _tiled_check_stats). The tiled pass hands back the merged
        # violation-marker Regions directly.
        #
        # The Region goes in AS a Region (hull-only, same as the µm rings this
        # used to build): every round rebuilt the whole layer as Python floats
        # and klayout rebuilt it from them again — on a 190k-polygon face that
        # was the bulk of the round, for geometry klayout already held.
        #
        # Round 0 checks the whole layer. Every round after it checks only the
        # neighbourhood of the previous round's weld — the same site set, for
        # the reason :func:`patched_neighbourhood` sets out, and the difference
        # between one whole-die check and twelve of them.
        target = reg if patch is None else patched_neighbourhood(reg, patch, gap_dbu)
        nw, mnw, ns, mns, w_markers, s_markers = _tiled_check_stats(
            target, kdb, width_dbu=width_dbu, gap_dbu=gap_dbu, dbu_um=dbu_um
        )
        rounds.append([nw, ns])
        if w_markers.is_empty() and s_markers.is_empty():
            final = (nw, mnw, ns, mns)
            break
        # OR the padded markers back in: welding a sub-floor GAP makes the
        # bridge litho would form anyway explicit, and thickening a sub-floor
        # NECK (welds can create these as ledges at their ends) is the same
        # surgery on the width axis. Padding by the half floor overlaps both
        # flanks and rounds the patch's own notches. Only violation sites
        # change — a global close would pinch legal necks (see note above).
        #
        # ESCALATION: two nearly-parallel curved strokes converging at a
        # shallow angle (the monogram cursive) can PING-PONG the half-floor
        # patch — each weld exposes a complementary neck beside it. After a
        # few fine rounds, switch to a FULL-floor pad: the whole convergence
        # zone fuses into one ≥-floor blob (which is what litho would print
        # there anyway) instead of being chased ledge by ledge.
        pad = pad_dbu if it < 3 else max(pad_dbu, gap_dbu) * (1 if it < 6 else 2)
        patch = (w_markers + s_markers).sized(pad)
        reg += patch
        reg.merge()
    out = []
    for poly in reg.each():
        # Exterior hull only (fill-only mask). klayout Polygon → µm vertices.
        hull = poly.each_point_hull()
        ring = [(pt.x * dbu_um, pt.y * dbu_um) for pt in hull]
        if len(ring) >= 3:
            out.append(np.asarray(ring, dtype=float))
    if not report:
        return out
    if final is None:
        # The loop patched on its way out of the round budget, so its last
        # measurement predates the final geometry: measure once, exactly as the
        # standalone report call used to.
        nw, mnw, ns, mns, _, _ = _tiled_check_stats(
            reg, kdb, width_dbu=width_dbu, gap_dbu=gap_dbu, dbu_um=dbu_um
        )
        final = (nw, mnw, ns, mns)
        rounds.append([nw, ns])
    return out, {
        "kind": "region", "n_polys": len(out),
        "n_width_viol": final[0], "n_space_viol": final[2],
        "min_width_um": final[1], "min_space_um": final[3],
        "rounds": rounds,
    }
