"""Analytic rect-space line screen — the halftone at plate scale.

``halftone.generate`` builds the screen by rasterizing a full-image boolean
lattice and run-merging it. That is exactly right at the few-millimetre extents
a box face uses, and it does not survive contact with a 5-inch witness plate.
Two separate walls, both worth stating because they set every size in the DoE:

RASTER. The working cell is ``line_period / tone_steps`` — 2.0 um for the
reference 44 um / 22-step screen. A 30 mm image is then 15000 x 15000 = 225 M
cells, 560x over ``_helpers.MAX_LATTICE_CELLS`` and about a quarter of a
gigabyte per temporary on a host that bugchecks under memory load.

RECTANGLES. Worse, and less obvious. A row-run merge emits one rectangle per
raster ROW, so a 22 um band on a 2 um grid costs 11 rectangles where one would
do. Sub-grating that band for colour then cuts every run at the grating period,
and a full-width run at 4.4 um across 30 mm is 6818 stripes. Multiply out and
an all-colour 30 mm image is ~4.7 M rectangles; the whole 120 mm plate would be
74 M.

So this module never rasterizes. It samples the source darkness once per line,
run-merges along the line, and emits ONE rectangle per tonal run spanning the
band's full height — the minimum-rectangle form of the same screen. Everything
downstream is ``(N, 4)`` ``[x0, x1, y0, y1]`` float arrays in micrometres, the
representation ``export_fine`` already speaks.

What actually bounds the rectangle count is then the SOURCE asset's resolution,
not the plate size: upsampling a 1400 px asset across 15000 cells makes every
run at least ~10 cells long, so a line holds ~1400 runs however large the plate
gets. That is the lever to reach for when a cell is too heavy — prep the asset
smaller, not the plate.

The periodic sub-grating stays cheap for a different reason: it is periodic, so
:func:`stripe_plan` can hand a GDS writer array references instead of polygons
(see ``app.export_witness``). Rectangles are only materialized when something
actually needs them one by one, such as an SVG preview.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# A single generate call may not emit more rectangles than this. Sized from the
# machine constraint rather than from GDS limits: ~4 M rectangles is roughly
# 400 MB of float64 vertex data before klayout sees any of it, and the host has
# bugchecked under less. Cells that need more must shrink, coarsen, or use the
# array-reference path.
MAX_RECTS = 4_000_000


def _sample_rows(src: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Nearest-neighbour resample of ``src`` onto an (n_rows, n_cols) grid.

    Nearest rather than bilinear on purpose: the prep pipeline has already put
    dither into the asset, and interpolating it back out would smooth away the
    noise that exists to break up the screen's banding.
    """
    h, w = src.shape
    ri = np.minimum((np.arange(n_rows) * (h / n_rows)).astype(np.int64), h - 1)
    ci = np.minimum((np.arange(n_cols) * (w / n_cols)).astype(np.int64), w - 1)
    return src[ri[:, None], ci[None, :]]


def _runs(key: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-length decomposition of each ROW of an integer array.

    Returns ``(row, col_start, col_end)`` with ``col_end`` exclusive. Same
    pad/diff idiom as ``_helpers.raster_to_polygons``, generalized from "runs of
    True" to "runs of equal key" so a tonal level and a colour period can be
    merged on jointly — a run must break where EITHER changes.
    """
    n_rows, n_cols = key.shape
    changed = np.empty(key.shape, dtype=bool)
    changed[:, 0] = True
    np.not_equal(key[:, 1:], key[:, :-1], out=changed[:, 1:])
    rows, starts = np.nonzero(changed)
    # The run ends where the next run in the SAME row starts, else at the edge.
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n_cols
    ends[rows != np.roll(rows, -1)] = n_cols
    return rows, starts, ends


def screen_bands(
    dark: np.ndarray,
    *,
    extent_um: float,
    line_period_um: float,
    tone_steps: int,
    period_id: np.ndarray | None = None,
    cols_per_line: int | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Screen a darkness map into centred band rectangles.

    ``dark`` is darkness in [0, 1] at whatever resolution the asset has (larger
    means more gold, matching ``imageprep.prep_darkness``). ``period_id`` is an
    optional integer map, at the SAME resolution, naming which colour period
    each pixel belongs to; runs break where it changes so every emitted
    rectangle has exactly one period.

    Returns ``(rects, pid, report)`` where ``rects`` is ``(N, 4)``
    ``[x0, x1, y0, y1]`` in micrometres about ``origin``, and ``pid`` is the
    per-rectangle period id (all zeros when ``period_id`` is None).

    The band is CENTRED in its line, matching ``halftone.generate``'s triangular
    carrier: band height is ``floor(tone*steps)/steps * line_period``, so the
    quantisation is the screen's real tone ladder and not a rounding artefact.
    """
    dark = np.asarray(dark, dtype=np.float32)
    if dark.ndim != 2:
        raise ValueError(f"dark must be 2-D, got shape {dark.shape}")
    if extent_um <= 0.0 or line_period_um <= 0.0:
        raise ValueError("extent_um and line_period_um must be > 0")
    steps = max(2, int(tone_steps))

    n_lines = max(1, int(round(extent_um / line_period_um)))
    # One column per tone cell is the finest the screen can express; there is no
    # point sampling the source finer than that, nor finer than the source is.
    want_cols = int(round(extent_um / (line_period_um / steps)))
    n_cols = max(8, min(want_cols, cols_per_line or want_cols, dark.shape[1] * 4))

    tone = _sample_rows(dark, n_lines, n_cols)
    level = np.floor(np.clip(tone, 0.0, 1.0) * steps).astype(np.int32)

    if period_id is None:
        pid_grid = np.zeros(level.shape, dtype=np.int32)
    else:
        period_id = np.asarray(period_id)
        if period_id.shape != dark.shape:
            raise ValueError(
                f"period_id shape {period_id.shape} != dark shape {dark.shape}"
            )
        pid_grid = _sample_rows(period_id.astype(np.float32), n_lines, n_cols)
        pid_grid = pid_grid.astype(np.int32)

    n_pid = int(pid_grid.max()) + 1
    rows, starts, ends = _runs(level * n_pid + pid_grid)

    lvl = level[rows, starts]
    keep = lvl > 0
    rows, starts, ends, lvl = rows[keep], starts[keep], ends[keep], lvl[keep]
    pid = pid_grid[rows, starts]

    ox, oy = origin
    col_um = extent_um / n_cols
    half = extent_um / 2.0
    band_h = (lvl.astype(np.float64) / steps) * line_period_um
    # Line j occupies [j*P, (j+1)*P) measured DOWN from the top edge, so the
    # emitted y matches an image whose first row is the top one.
    cy = (half - (rows + 0.5) * line_period_um) + oy

    rects = np.empty((rows.size, 4), dtype=np.float64)
    rects[:, 0] = starts * col_um - half + ox
    rects[:, 1] = ends * col_um - half + ox
    rects[:, 2] = cy - band_h / 2.0
    rects[:, 3] = cy + band_h / 2.0

    report = {
        "n_lines": int(n_lines),
        "n_cols": int(n_cols),
        "n_band_rects": int(rects.shape[0]),
        "tone_steps": steps,
        "line_period_um": float(line_period_um),
        "cell_um": float(line_period_um / steps),
        "col_um": float(col_um),
        "finest_band_um": float(line_period_um / steps),
        "coverage": float(np.clip(tone, 0, 1).mean()),
        # Rectangles per line is the number to watch when sizing a cell: it is
        # set by the ASSET, so it barely moves when the plate grows.
        "rects_per_line": round(rects.shape[0] / float(n_lines), 1),
    }
    return rects, pid, report


def stripe_plan(
    rects: np.ndarray,
    period_um: np.ndarray | float,
    duty: np.ndarray | float,
    *,
    phase_um: float = 0.0,
) -> dict[str, np.ndarray]:
    """Plan the vertical sub-grating of each band rectangle, without expanding it.

    Returns the arrays a writer needs to emit either polygons
    (:func:`stripe_rects`) or one array reference per band: ``k0``/``k1`` are the
    inclusive stripe indices on the global lattice ``phase + k*period``, ``n``
    the stripe count, and ``line_um`` the gold width.

    Lines run VERTICALLY (constant x) because the screen's own bands run
    horizontally; a sub-grating parallel to the screen would merely re-cut the
    band along its own axis and produce no second periodicity at all.
    """
    rects = np.asarray(rects, dtype=np.float64)
    if rects.ndim != 2 or rects.shape[1] != 4:
        raise ValueError(f"rects must be (N, 4), got {rects.shape}")
    d = np.broadcast_to(np.asarray(period_um, dtype=np.float64), (rects.shape[0],))
    c = np.broadcast_to(np.asarray(duty, dtype=np.float64), (rects.shape[0],))
    if np.any(d <= 0.0):
        raise ValueError("period_um must be > 0 everywhere")
    if np.any((c <= 0.0) | (c >= 1.0)):
        raise ValueError("duty must be strictly between 0 and 1")

    x0, x1 = rects[:, 0], rects[:, 1]
    line = d * c
    # WHOLE stripes only, selected by whether a stripe's CENTRE falls inside the
    # band. The alternative — running the lattice across the band and clipping
    # the two ends — is exact, and it costs one extra explicit rectangle per
    # band END, which on a plate full of halftones was 2.75 M boxes and 150 MB
    # of GDS on its own. It also destroys the periodicity that lets the whole
    # band become a single array reference.
    #
    # Snapping moves each band edge by at most half a period (2.5 um at the
    # reference 5 um), against a screen whose own column is 2 um — a sub-column
    # change to where a tonal run ends. Centre-in rather than inward-snapping
    # keeps it UNBIASED: a band is as likely to gain a stripe as to lose one, so
    # mean coverage, and therefore tone, is preserved across the image.
    k0 = np.ceil((x0 - phase_um - line / 2.0) / d).astype(np.int64)
    k1 = np.floor((x1 - phase_um - line / 2.0) / d).astype(np.int64)
    n = np.maximum(k1 - k0 + 1, 0)
    return {
        "k0": k0,
        "k1": k1,
        "n": n,
        "period_um": d,
        "duty": c,
        "line_um": line,
        "phase_um": np.float64(phase_um),
        "total": int(n.sum()),
    }


def stripe_rects(
    rects: np.ndarray,
    period_um: np.ndarray | float,
    duty: np.ndarray | float,
    *,
    phase_um: float = 0.0,
    max_rects: int = MAX_RECTS,
) -> np.ndarray:
    """Materialize the sub-grating as rectangles, clipped to each band.

    This is the expensive form — see the module docstring. It is what an SVG or
    a raster preview needs; a GDS writer should prefer :func:`stripe_plan`.
    """
    rects = np.asarray(rects, dtype=np.float64)
    plan = stripe_plan(rects, period_um, duty, phase_um=phase_um)
    n = plan["n"]
    total = plan["total"]
    if total > max_rects:
        raise ValueError(
            f"sub-grating would emit {total:,} rectangles (cap {max_rects:,}). "
            f"Coarsen the colour period, shrink the extent, or colour a smaller "
            f"fraction of the image — see screenrects module docstring."
        )
    if total == 0:
        return np.empty((0, 4), dtype=np.float64)

    src = np.repeat(np.arange(rects.shape[0]), n)
    # Offset of each stripe within its parent rectangle: 0,1,2,... restarting
    # per parent, built without a Python loop.
    off = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    d = plan["period_um"][src]
    k = plan["k0"][src] + off

    sx0 = phase_um + k * d
    out = np.empty((total, 4), dtype=np.float64)
    out[:, 0] = sx0
    out[:, 1] = sx0 + plan["line_um"][src]
    out[:, 2] = rects[src, 2]
    out[:, 3] = rects[src, 3]
    # No clipping: stripe_plan already selected only whole stripes, so what a
    # preview draws and what the GDS writer emits are the same geometry. Letting
    # these two diverge would put the fab mask and the preview out of agreement,
    # which is the failure mode CLAUDE.md's "fab SVG = preview PNG" rule exists
    # to prevent.
    return out


def split_by_colour(
    rects: np.ndarray, pid: np.ndarray, periods_um: dict[int, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition band rectangles into (plain, coloured, coloured periods).

    ``periods_um`` maps a period id to its period; id 0 and any id absent from
    the mapping stay plain gold, which is what makes "plain" a degenerate case
    of the same code path rather than a separate branch.
    """
    rects = np.asarray(rects, dtype=np.float64)
    pid = np.asarray(pid)
    lut = np.zeros(int(pid.max()) + 1 if pid.size else 1, dtype=np.float64)
    for k, v in periods_um.items():
        if 0 <= int(k) < lut.size:
            lut[int(k)] = float(v)
    per = lut[pid] if pid.size else np.zeros(0)
    coloured = per > 0.0
    return rects[~coloured], rects[coloured], per[coloured]


def rect_area_um2(rects: np.ndarray) -> float:
    r = np.asarray(rects, dtype=np.float64)
    if r.size == 0:
        return 0.0
    return float(((r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2])).sum())
