from __future__ import annotations

import drawsvg as dw
import numpy as np
import shapely
from shapely.geometry import MultiPolygon


def to_svg(
    polys: MultiPolygon,
    extent_um: tuple[float, float],
    fill: str = "#E6BC50",
    background: str | None = None,
) -> str:
    """Emit a standards-compliant SVG whose user units are μm, origin at the center.

    This is the authoritative vector form for the fab upgrade path
    (SVG -> shapely -> gdsfactory -> GDS -> DXF).

    Axis-aligned-rectangle input (everything ``raster_to_polygons`` produces —
    both baked plate layers) collapses into ONE ``<path>`` whose ``d`` is built
    with numpy; a plate face is ~17k rects and the per-polygon drawsvg path cost
    ~2.5 s/layer of pure string formatting. Anything else (holes, non-rect rings)
    keeps the one-path-per-polygon form.
    """
    w, h = extent_um
    d = dw.Drawing(w, h, origin=(-w / 2, -h / 2))
    if background is not None:
        d.append(dw.Rectangle(-w / 2, -h / 2, w, h, fill=background))

    if polys.is_empty:
        return d.as_svg()

    parts = (
        shapely.get_parts(polys)
        if isinstance(polys, MultiPolygon)
        else np.array([polys], dtype=object)
    )
    # Same skip rule the per-polygon loop below applies: polygons only, no empties.
    parts = parts[(shapely.get_type_id(parts) == 3) & ~shapely.is_empty(parts)]
    if parts.size == 0:
        return d.as_svg()

    rects = _as_rects(parts)
    if rects is not None:
        # Canonical winding (all rings emitted min->max) means fill-rule
        # "nonzero" fills the union, matching the one-path-per-rect rendering
        # even where rects overlap — "evenodd" would XOR overlaps into holes
        # now that they share a single path. Safe because _as_rects proves
        # every ring is a non-degenerate rectangle with no interiors.
        d.append(dw.Path(_rect_path_d(rects), fill=fill, stroke="none", fill_rule="nonzero"))
        return d.as_svg()

    for poly in parts:
        path = dw.Path(fill=fill, stroke="none", fill_rule="evenodd")
        ext = list(poly.exterior.coords)
        _move_line(path, ext)
        for ring in poly.interiors:
            _move_line(path, list(ring.coords))
        d.append(path)

    return d.as_svg()


def _as_rects(parts: np.ndarray) -> np.ndarray | None:
    """``(N, 4)`` array of ``[xmin, xmax, ymin, ymax]`` if EVERY part is a
    hole-free axis-aligned rectangle, else ``None``.

    Rejection is the general-path signal, so the test is exact (no tolerance):
    each of the 4 cyclic edges must hold exactly one coordinate constant, and
    each of x/y must take exactly two values twice over. Together those bar
    diagonals, repeated vertices, bow-ties and spurs — so the min/max bbox is
    provably the ring itself.
    """
    if shapely.get_num_interior_rings(parts).any():
        return None
    if not np.all(shapely.get_num_coordinates(parts) == 5):
        return None

    v = shapely.get_coordinates(parts).reshape(-1, 5, 2)
    if not np.all(v[:, 4, :] == v[:, 0, :]):
        return None
    xs = v[:, :4, 0]
    ys = v[:, :4, 1]
    axis_aligned = (np.roll(xs, -1, axis=1) == xs) ^ (np.roll(ys, -1, axis=1) == ys)
    if not np.all(axis_aligned):
        return None

    x0 = xs.min(axis=1)
    x1 = xs.max(axis=1)
    y0 = ys.min(axis=1)
    y1 = ys.max(axis=1)
    if not np.all((x0 < x1) & (y0 < y1)):
        return None
    paired = (
        ((xs == x0[:, None]).sum(axis=1) == 2)
        & ((xs == x1[:, None]).sum(axis=1) == 2)
        & ((ys == y0[:, None]).sum(axis=1) == 2)
        & ((ys == y1[:, None]).sum(axis=1) == 2)
    )
    if not np.all(paired):
        return None

    return np.stack([x0, x1, y0, y1], axis=1)


def _rect_path_d(rects: np.ndarray) -> str:
    """``d`` for one path covering every rectangle in ``rects`` (N, 4).

    Vertex order and y flip match what ``_move_line`` emits for a
    ``raster_to_polygons`` ring (lower-left, lower-right, upper-right,
    upper-left), and ``.astype(str)`` on float64 is numpy's shortest
    round-trip repr — the same text ``str(float)`` hands drawsvg — so every
    coordinate is textually identical to the general path. The only difference
    is the ring's redundant final lineto back to the start, which ``Z``
    already implies.
    """
    # SVG y points down; our convention has y up, so flip.
    x0, x1, y0, y1 = (
        rects[:, 0].astype(str),
        rects[:, 1].astype(str),
        (-rects[:, 2]).astype(str),
        (-rects[:, 3]).astype(str),
    )
    # One object-dtype token matrix joined in a single pass: a chain of
    # np.char.add reallocates a wider U-array per column and costs ~4x this.
    tok = np.empty((len(rects), 17), dtype=object)
    tok[:, 0] = " M"
    tok[:, 1] = x0
    tok[:, 3] = y0
    tok[:, 5] = x1
    tok[:, 7] = y0
    tok[:, 9] = x1
    tok[:, 11] = y1
    tok[:, 13] = x0
    tok[:, 15] = y1
    tok[:, 2] = tok[:, 6] = tok[:, 10] = tok[:, 14] = ","
    tok[:, 4] = tok[:, 8] = tok[:, 12] = " L"
    tok[:, 16] = " Z"
    # [1:] drops the separator space that leads the first rect.
    return "".join(tok.ravel().tolist())[1:]


def _move_line(path, coords: list[tuple[float, float]]) -> None:
    if not coords:
        return
    x0, y0 = coords[0]
    # SVG y points down; our convention has y up, so flip.
    path.M(x0, -y0)
    for x, y in coords[1:]:
        path.L(x, -y)
    path.Z()
