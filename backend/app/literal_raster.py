"""EXACT area-coverage rasterisation of the fine litho polygons.

``plates._literal_layer_raster`` publishes ``literal_front.png`` /
``literal_back.png``: the DRC-healed gold rings ``export_fine.build_plate_fine``
hands the GDS writer, sampled as AREA COVERAGE (255 = chrome everywhere in the
texel, 0 = bare glass). The renderer samples those rasters instead of
synthesising gratings, so a systematic error in the coverage is a systematic
error in the preview's brightness — and the features are far below a texel:
at 2048 px on a 27.5 mm face a texel is ~13.4 µm, while the colour
sub-gratings are 2.5 µm lines on a 5 µm period.

WHY THIS IS NOT A SUPERSAMPLED FILL
-----------------------------------
The original implementation filled the rings with PIL ``ImageDraw.polygon`` at
2x and box-downsampled. PIL's polygon fill is boundary-INCLUSIVE *and*
phase-quantised: a rectangle spanning x ∈ [10.0, 20.0] (10 px wide) fills
columns 10..20 — ELEVEN pixels — and so does one spanning [10.25, 20.25]. Every
line is therefore fattened by a whole sample, in steps of a whole sample. That
put a measured +6% on a 50%-duty carrier and drove the 5 µm colour bands to a
0.71 mean with 46% of their texels pinned at 255.

The obvious repair — shrink each polygon by half a sample before filling —
cannot work, precisely because the fattening is quantised rather than
proportional: at ss=4 a 50% grating measures 0.526 / 0.515 / 0.528 / 0.529 at
0° / 30° / 45° / 63°, and no single shrink brings all four inside ±0.005 (a
half-sample shrink lands them at 0.490 / 0.485 / 0.490 / 0.497). So this module
computes the coverage ANALYTICALLY instead — exactly, for arbitrary polygons,
with no sample grid and therefore no bias, no aliasing and no thin-feature
floor. Two paths, both exact:

1. AXIS-ALIGNED RECTANGLES (``_accumulate_rects``) — every halftone band, every
   colour stripe, the comb and the switch lanes, i.e. the overwhelming majority
   of a photo face's rings. Each rect is clipped to the texels it touches and
   its exact overlap area is accumulated. (Same idea as ``coverage_grid`` in
   ``tools/dev/validate_dies.py``, re-expressed as a flat run expansion so the
   cost is O(texels touched) rather than O(rects x longest span).)

2. EVERYTHING ELSE (``_accumulate_rings``) — angled carrier and leaf gratings,
   healed rings, rotated trapezoids. Exact signed-area accumulation by Green's
   theorem: for one texel row, the area of the polygon left of a vertical line
   x is the integral of the crossing measure, so each edge deposits its exact
   contribution into the texel it passes through plus a carry into the next
   one, and a prefix sum along the row turns the carries into coverage. Cost is
   O(texel crossings), which for the near-vertical grating lines here is
   O(rows), not O(area).

Both accumulate into the same float grid, so a texel that a rect and an angled
line both touch adds up correctly. The rings are a MERGED region (holes already
dropped upstream — the mask is fill-only) so they do not overlap and the sum is
a true coverage; it is clipped to [0, 1] anyway, which also absorbs the rare
self-touching healed ring.

Grid frame: plate µm with the origin at the plate centre and y UP map to texels
exactly as ``plates._raster_compose_plate`` maps them — x + W/2 scaled across
the width, H/2 - y scaled down the height, so row 0 is the top.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Expanded-run budget per accumulation chunk. Each expanded element costs a
# handful of float64/int64 temporaries, so ~1M keeps a chunk's working set in
# the tens of MB on a face whose long gratings cross thousands of texels.
_CHUNK = 1_000_000


def _runs(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run expansion: ``(src, k)`` with ``src`` the source index repeated
    ``counts[src]`` times and ``k`` the 0-based position within its run."""
    counts = np.asarray(counts, dtype=np.int64)
    src = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    starts = np.zeros(counts.size, dtype=np.int64)
    if counts.size > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    k = np.arange(src.size, dtype=np.int64) - starts[src]
    return src, k


def _chunks(counts: np.ndarray, budget: int = _CHUNK):
    """Yield ``(lo, hi)`` slices of ``counts`` whose expansions stay under
    ``budget`` (always at least one element, however wide it expands)."""
    n = int(counts.size)
    if n == 0:
        return
    cum = np.concatenate(([0], np.cumsum(np.asarray(counts, dtype=np.int64))))
    i = 0
    while i < n:
        j = int(np.searchsorted(cum, cum[i] + budget, side="right")) - 1
        j = max(j, i + 1)
        j = min(j, n)
        yield i, j
        i = j


# --- path 1: exact axis-aligned rectangles -----------------------------------


def _accumulate_rects(
    acc: np.ndarray, x0: np.ndarray, x1: np.ndarray, y0: np.ndarray, y1: np.ndarray
) -> None:
    """Add each rect's EXACT area fraction into every texel of ``acc`` it covers.

    Coordinates are in texel units (``acc`` spans [0, W] x [0, H]) with y down
    and ``x0 < x1``, ``y0 < y1`` already ordered.
    """
    h_px, w_px = acc.shape
    x0 = np.clip(x0, 0.0, float(w_px))
    x1 = np.clip(x1, 0.0, float(w_px))
    y0 = np.clip(y0, 0.0, float(h_px))
    y1 = np.clip(y1, 0.0, float(h_px))
    keep = (x1 > x0) & (y1 > y0)
    if not keep.any():
        return
    x0, x1, y0, y1 = x0[keep], x1[keep], y0[keep], y1[keep]

    ix0 = np.floor(x0).astype(np.int64)
    ix1 = np.maximum(np.minimum(np.ceil(x1).astype(np.int64) - 1, w_px - 1), ix0)
    iy0 = np.floor(y0).astype(np.int64)
    iy1 = np.maximum(np.minimum(np.ceil(y1).astype(np.int64) - 1, h_px - 1), iy0)
    cx = ix1 - ix0 + 1
    cy = iy1 - iy0 + 1
    ncell = cx * cy

    flat = acc.reshape(-1)
    for lo, hi in _chunks(ncell, _CHUNK):
        src, k = _runs(ncell[lo:hi])
        cxs = cx[lo:hi][src]
        ix = ix0[lo:hi][src] + (k % cxs)
        iy = iy0[lo:hi][src] + (k // cxs)
        ox = np.minimum(x1[lo:hi][src], ix + 1.0) - np.maximum(x0[lo:hi][src], ix)
        oy = np.minimum(y1[lo:hi][src], iy + 1.0) - np.maximum(y0[lo:hi][src], iy)
        flat += np.bincount(
            iy * w_px + ix, weights=ox * oy, minlength=h_px * w_px
        )


def rect_texel_overlaps(
    x0: np.ndarray, x1: np.ndarray, y0: np.ndarray, y1: np.ndarray, w_px: int, h_px: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(flat_texel_index, rect_index, overlap_area)`` for axis-aligned rects.

    Texel units, y down, ``x0 < x1`` / ``y0 < y1``. Same touched-texel set as a
    floor/ceil outset — a rect thinner than a texel still reports the texel it
    lands in — but with the EXACT overlap area attached, so a caller stacking
    several rects can pick the one that really dominates a texel instead of
    whichever it happened to paint last. Used by ``period_front.png``.
    """
    x0 = np.clip(x0, 0.0, float(w_px))
    x1 = np.clip(x1, 0.0, float(w_px))
    y0 = np.clip(y0, 0.0, float(h_px))
    y1 = np.clip(y1, 0.0, float(h_px))
    keep = np.nonzero((x1 > x0) & (y1 > y0))[0]
    if keep.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0)
    x0, x1, y0, y1 = x0[keep], x1[keep], y0[keep], y1[keep]
    ix0 = np.floor(x0).astype(np.int64)
    ix1 = np.maximum(np.minimum(np.ceil(x1).astype(np.int64) - 1, w_px - 1), ix0)
    iy0 = np.floor(y0).astype(np.int64)
    iy1 = np.maximum(np.minimum(np.ceil(y1).astype(np.int64) - 1, h_px - 1), iy0)
    cx = ix1 - ix0 + 1
    src, k = _runs(cx * (iy1 - iy0 + 1))
    cxs = cx[src]
    ix = ix0[src] + (k % cxs)
    iy = iy0[src] + (k // cxs)
    ox = np.minimum(x1[src], ix + 1.0) - np.maximum(x0[src], ix)
    oy = np.minimum(y1[src], iy + 1.0) - np.maximum(y0[src], iy)
    return iy * w_px + ix, keep[src], ox * oy


# --- path 2: exact arbitrary polygons ----------------------------------------


def _edges_from_rings(rings: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Flatten rings (texel units, y down) to oriented edge arrays.

    Returns ``(ex0, ey0, ex1, ey1, ori)`` where ``ori`` is +1/-1 per edge,
    carrying its ring's winding so a ring supplied clockwise and one supplied
    counter-clockwise both accumulate POSITIVE coverage.
    """
    lens = np.fromiter((len(r) for r in rings), dtype=np.int64, count=len(rings))
    verts = np.concatenate(rings) if rings else np.zeros((0, 2))
    ring_id = np.repeat(np.arange(lens.size, dtype=np.int64), lens)
    starts = np.zeros(lens.size, dtype=np.int64)
    if lens.size > 1:
        np.cumsum(lens[:-1], out=starts[1:])
    nxt = np.arange(verts.shape[0], dtype=np.int64) + 1
    nxt[starts + lens - 1] = starts  # wrap each ring's last vertex to its first

    x0, y0 = verts[:, 0], verts[:, 1]
    x1, y1 = verts[nxt, 0], verts[nxt, 1]
    # Shoelace per ring, in THIS (y-down) frame: >0 means the ring is wound so
    # that the derivation below yields positive coverage.
    area2 = np.bincount(ring_id, weights=x0 * y1 - x1 * y0, minlength=lens.size)
    ori = np.where(area2[ring_id] >= 0.0, 1.0, -1.0)
    return x0, y0, x1, y1, ori


def _accumulate_rings(edge_acc: np.ndarray, rings: Sequence[np.ndarray]) -> None:
    """Add the rings' exact coverage into ``edge_acc``, an ``(H, W+3)`` buffer.

    ``edge_acc`` holds the DERIVATIVE of coverage along a row: texel column
    ``c`` lives at index ``c + 1`` (index 0 is the off-grid column -1, which is
    where geometry hanging off the left edge parks its carry; the two columns
    past W absorb the right-hand carries), and a cumulative sum along axis 1
    turns it into coverage. See the module docstring.
    """
    if not rings:
        return
    h_px = edge_acc.shape[0]
    w_px = edge_acc.shape[1] - 3
    ex0, ey0, ex1, ey1, ori = _edges_from_rings(rings)

    # Order each edge downward and fold the traversal direction into `dirn`:
    # an edge running UP the screen carries +1 for a positively-wound ring.
    up = ey1 < ey0
    ylo = np.where(up, ey1, ey0)
    yhi = np.where(up, ey0, ey1)
    xa = np.where(up, ex1, ex0)
    xb = np.where(up, ex0, ex1)
    dirn = ori * np.where(up, 1.0, -1.0)

    span = yhi - ylo
    keep = span > 0.0  # horizontal edges contribute nothing
    if not keep.any():
        return
    ylo, yhi, xa, xb, dirn, span = (
        ylo[keep],
        yhi[keep],
        xa[keep],
        xb[keep],
        dirn[keep],
        span[keep],
    )

    # Clip to the visible rows (rows outside cannot affect any texel), moving
    # the x endpoints along the edge so the remaining segment is still exact.
    c0 = np.clip(ylo, 0.0, float(h_px))
    c1 = np.clip(yhi, 0.0, float(h_px))
    keep = c1 > c0
    if not keep.any():
        return
    t0 = (c0[keep] - ylo[keep]) / span[keep]
    t1 = (c1[keep] - ylo[keep]) / span[keep]
    dxe = (xb - xa)[keep]
    xa, xb = xa[keep] + dxe * t0, xa[keep] + dxe * t1
    ylo, yhi, dirn = c0[keep], c1[keep], dirn[keep]

    # Clip to the horizontal strip [-1, W+1]. Geometry off the LEFT still
    # covers every texel to its right, so it is projected onto the boundary
    # (which preserves its dy exactly and parks it in column -1); geometry off
    # the RIGHT covers nothing visible and is dropped. Straddling edges are
    # split at the crossings first — build_plate_fine's output is already
    # plate-clipped so in practice nothing straddles, but a clamp without the
    # split would skew the visible part of one that did.
    lo_x, hi_x = -1.0, float(w_px) + 1.0
    xl_e = np.minimum(xa, xb)
    xr_e = np.maximum(xa, xb)
    straddle = ((xl_e < lo_x) & (xr_e > lo_x)) | ((xl_e < hi_x) & (xr_e > hi_x))
    if straddle.any():
        keep = ~straddle
        parts = _split_at_strip(
            ylo[straddle], yhi[straddle], xa[straddle], xb[straddle],
            dirn[straddle], lo_x, hi_x,
        )
        ylo = np.concatenate([ylo[keep], parts[0]])
        yhi = np.concatenate([yhi[keep], parts[1]])
        xa = np.concatenate([xa[keep], parts[2]])
        xb = np.concatenate([xb[keep], parts[3]])
        dirn = np.concatenate([dirn[keep], parts[4]])
    xa = np.clip(xa, lo_x, hi_x)
    xb = np.clip(xb, lo_x, hi_x)
    drop = (xa >= float(w_px)) & (xb >= float(w_px))
    if drop.any():
        keep = ~drop
        ylo, yhi, xa, xb, dirn = ylo[keep], yhi[keep], xa[keep], xb[keep], dirn[keep]
    if ylo.size == 0:
        return

    span = yhi - ylo
    r0 = np.floor(ylo).astype(np.int64)
    r1 = np.maximum(np.minimum(np.ceil(yhi).astype(np.int64) - 1, h_px - 1), r0)
    nrow = r1 - r0 + 1
    flat = edge_acc.reshape(-1)
    stride = w_px + 3

    for lo, hi in _chunks(nrow, _CHUNK):
        src, k = _runs(nrow[lo:hi])
        row = r0[lo:hi][src] + k
        ya = np.maximum(ylo[lo:hi][src], row)
        yb = np.minimum(yhi[lo:hi][src], row + 1.0)
        s = span[lo:hi][src]
        xas = xa[lo:hi][src]
        dxs = xb[lo:hi][src] - xas
        base = ylo[lo:hi][src]
        rxa = xas + dxs * ((ya - base) / s)
        rxb = xas + dxs * ((yb - base) / s)
        # Signed vertical extent of this row segment.
        dsum = dirn[lo:hi][src] * (yb - ya)

        xl = np.minimum(rxa, rxb)
        xr = np.maximum(rxa, rxb)
        cl = np.floor(xl).astype(np.int64)
        cr = np.maximum(np.ceil(xr).astype(np.int64) - 1, cl)
        ncol = cr - cl + 1
        _deposit(flat, stride, row, xl, xr, cl, ncol, dsum)


def _deposit(
    flat: np.ndarray,
    stride: int,
    row: np.ndarray,
    xl: np.ndarray,
    xr: np.ndarray,
    cl: np.ndarray,
    ncol: np.ndarray,
    dsum: np.ndarray,
) -> None:
    """Split each row segment across the texel columns it crosses and deposit
    its exact ``coverage'`` contribution (value into column c, carry into c+1)."""
    for lo, hi in _chunks(ncol, _CHUNK):
        src, k = _runs(ncol[lo:hi])
        col = cl[lo:hi][src] + k
        a = np.maximum(xl[lo:hi][src], col)
        b = np.minimum(xr[lo:hi][src], col + 1.0)
        wide = xr[lo:hi][src] - xl[lo:hi][src]
        # x is linear in y along the segment, so the share of dy landing in this
        # column is its share of the x span. A vertical segment (wide == 0)
        # lives entirely in one column and takes the whole share.
        frac = np.where(
            wide > 0.0, np.maximum(b - a, 0.0) / np.where(wide > 0.0, wide, 1.0), 1.0
        )
        d = dsum[lo:hi][src] * frac
        xm = 0.5 * (a + b) - col
        idx = row[lo:hi][src] * stride + (col + 1)
        flat += np.bincount(
            np.concatenate([idx, idx + 1]),
            weights=np.concatenate([d * (1.0 - xm), d * xm]),
            minlength=flat.size,
        )


def _split_at_strip(ylo, yhi, xa, xb, dirn, lo_x, hi_x):
    """Split edges that cross x = ``lo_x`` / ``hi_x`` into pieces that each lie
    on one side, so a later clamp only ever moves a wholly-outside piece.

    Cold path: ``build_plate_fine`` clips to the plate, so this normally sees
    nothing. Kept because clamping a straddling edge would bend the part of it
    that IS visible.
    """
    out_y0, out_y1, out_xa, out_xb, out_d = [], [], [], [], []
    dx = xb - xa
    for i in range(ylo.size):
        cuts = [0.0, 1.0]
        if dx[i] != 0.0:
            for bx in (lo_x, hi_x):
                t = (bx - xa[i]) / dx[i]
                if 0.0 < t < 1.0:
                    cuts.append(t)
        cuts.sort()
        for t0, t1 in zip(cuts[:-1], cuts[1:]):
            if t1 <= t0:
                continue
            out_y0.append(ylo[i] + (yhi[i] - ylo[i]) * t0)
            out_y1.append(ylo[i] + (yhi[i] - ylo[i]) * t1)
            out_xa.append(xa[i] + dx[i] * t0)
            out_xb.append(xa[i] + dx[i] * t1)
            out_d.append(dirn[i])
    return (
        np.asarray(out_y0),
        np.asarray(out_y1),
        np.asarray(out_xa),
        np.asarray(out_xb),
        np.asarray(out_d),
    )


# --- public entry point ------------------------------------------------------


def _is_axis_rect(quads: np.ndarray) -> np.ndarray:
    """Boolean mask over an ``(M,4,2)`` stack: exactly 2 distinct x and 2
    distinct y, i.e. an axis-aligned rectangle. Exact comparison — the rings
    come off a DBU grid, and a ring that misses by a rounding step is still
    exact down the polygon path."""
    x = quads[:, :, 0]
    y = quads[:, :, 1]
    xmin, xmax = x.min(axis=1), x.max(axis=1)
    ymin, ymax = y.min(axis=1), y.max(axis=1)
    on_x = ((x == xmin[:, None]) | (x == xmax[:, None])).all(axis=1)
    on_y = ((y == ymin[:, None]) | (y == ymax[:, None])).all(axis=1)
    return on_x & on_y & (xmax > xmin) & (ymax > ymin)


def layer_coverage(
    polys: Iterable[np.ndarray],
    w_um: float,
    h_um: float,
    w_px: int,
    h_px: int,
) -> np.ndarray:
    """Plate-frame polygon rings -> exact per-texel area coverage in [0, 1].

    ``polys`` are ``(K,2)`` µm vertex rings, origin at the plate centre, y UP.
    Returns a ``(h_px, w_px)`` float64 array, row 0 = top.
    """
    cov = np.zeros((h_px, w_px), dtype=np.float64)
    rings = [np.asarray(r, dtype=np.float64) for r in polys]
    rings = [r for r in rings if r.ndim == 2 and r.shape[0] >= 3 and r.shape[1] == 2]
    if not rings:
        return cov

    w_um = max(1e-6, float(w_um))
    h_um = max(1e-6, float(h_um))
    sx = w_px / w_um
    sy = h_px / h_um
    hx, hy = 0.5 * w_um, 0.5 * h_um

    # Split the quads that are axis-aligned rectangles off the front — on a
    # photo face they are ~97% of the rings and path 1 handles them in one
    # vectorised sweep.
    quad_idx = [i for i, r in enumerate(rings) if r.shape[0] == 4]
    rect_mask = np.zeros(len(rings), dtype=bool)
    if quad_idx:
        quads = np.stack([rings[i] for i in quad_idx])
        is_rect = _is_axis_rect(quads)
        rect_mask[np.asarray(quad_idx, dtype=np.int64)[is_rect]] = True
        rq = quads[is_rect]
        if rq.shape[0]:
            gx = (rq[:, :, 0] + hx) * sx
            gy = (hy - rq[:, :, 1]) * sy
            _accumulate_rects(
                cov, gx.min(axis=1), gx.max(axis=1), gy.min(axis=1), gy.max(axis=1)
            )

    others = [rings[i] for i in range(len(rings)) if not rect_mask[i]]
    if others:
        grid = [
            np.stack([(r[:, 0] + hx) * sx, (hy - r[:, 1]) * sy], axis=1) for r in others
        ]
        edge_acc = np.zeros((h_px, w_px + 3), dtype=np.float64)
        _accumulate_rings(edge_acc, grid)
        np.cumsum(edge_acc, axis=1, out=edge_acc)
        cov += edge_acc[:, 1 : w_px + 1]

    np.clip(cov, 0.0, 1.0, out=cov)
    return cov
