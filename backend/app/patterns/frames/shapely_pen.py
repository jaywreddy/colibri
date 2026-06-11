"""Pen backend that accumulates strokes + fills into a Shapely MultiPolygon.

Stroked polylines become thin polygons via ``LineString.buffer(width/2)`` with
flat caps so neighboring segments butt cleanly. Filled subpaths become
polygons. The pen flushes its accumulated geometry on demand via
``finish() -> MultiPolygon`` so callers can compose it onto a layer
(by concatenation — see ``finish`` for why union is opt-in only).

Transforms are tracked as a 2x3 affine matrix; coordinates from motifs pass
through ``_xform`` before joining the accumulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..base import ensure_multipolygon
from .pen import Pen


@dataclass
class _Affine:
    # column-vector convention: [x', y']ᵀ = M · [x, y, 1]ᵀ
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def copy(self) -> "_Affine":
        return _Affine(self.a, self.b, self.c, self.d, self.tx, self.ty)

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (self.a * x + self.c * y + self.tx, self.b * x + self.d * y + self.ty)

    def translate(self, dx: float, dy: float) -> None:
        self.tx, self.ty = self.apply(dx, dy)

    def rotate(self, rad: float) -> None:
        cs, sn = math.cos(rad), math.sin(rad)
        a, b, c, d = self.a, self.b, self.c, self.d
        self.a = a * cs + c * sn
        self.b = b * cs + d * sn
        self.c = -a * sn + c * cs
        self.d = -b * sn + d * cs

    def scale(self, sx: float, sy: float) -> None:
        self.a *= sx
        self.b *= sx
        self.c *= sy
        self.d *= sy

    def uniform_scale(self) -> float:
        """RMS of the column lengths — used to convert local stroke widths."""
        sx = math.hypot(self.a, self.b)
        sy = math.hypot(self.c, self.d)
        return 0.5 * (sx + sy)


class ShapelyPen(Pen):
    """Collects strokes and fills into geometry buffers.

    Strokes and fills are tracked separately for efficient union — many small
    subpaths balloon shapely's vertex count if every path is unioned eagerly.
    """

    def __init__(self) -> None:
        self._tx_stack: list[_Affine] = [_Affine()]
        # Accumulators — flushed by ``finish``. Stored as raw vertex lists
        # (not shapely objects) so the per-stroke / per-fill cost is just a
        # Python list append; the shapely allocation happens once at finish().
        self._stroke_paths: list[tuple[list[tuple[float, float]], float]] = []
        self._fill_paths: list[list[tuple[float, float]]] = []
        # Disks come in a separate bucket so we can buffer them as Points in
        # one batch (motifs use them for berries, centers, etc).
        self._disks: list[tuple[float, float, float]] = []  # (cx, cy, r)
        # Current subpath, in *world* coords (already transformed).
        self._path: list[tuple[float, float]] = []

    # --- transform stack ---------------------------------------------------
    def save(self) -> None:
        self._tx_stack.append(self._tx_stack[-1].copy())

    def restore(self) -> None:
        if len(self._tx_stack) > 1:
            self._tx_stack.pop()

    def _tx(self) -> _Affine:
        return self._tx_stack[-1]

    def translate(self, dx: float, dy: float) -> None:
        self._tx().translate(dx, dy)

    def rotate(self, radians: float) -> None:
        self._tx().rotate(radians)

    def scale(self, sx: float, sy: float | None = None) -> None:
        self._tx().scale(sx, sx if sy is None else sy)

    # --- path construction -------------------------------------------------
    def _w(self, x: float, y: float) -> tuple[float, float]:
        return self._tx().apply(x, y)

    def move_to(self, x: float, y: float) -> None:
        # Starting a new subpath: discard any in-progress (uncommitted) one.
        # A motif always calls stroke_path / fill_path before move_to-ing
        # to a new region.
        self._path = [self._w(x, y)]

    def line_to(self, x: float, y: float) -> None:
        if not self._path:
            self._path.append(self._w(x, y))
            return
        self._path.append(self._w(x, y))

    def quadratic_to(self, cx: float, cy: float, x: float, y: float) -> None:
        if not self._path:
            self._path.append(self._w(cx, cy))
        # 8-segment polyline approximation — enough for the motif scale.
        p0 = self._path[-1]
        p1 = self._w(cx, cy)
        p2 = self._w(x, y)
        for i in range(1, 9):
            tt = i / 8.0
            u = 1.0 - tt
            bx = u * u * p0[0] + 2 * u * tt * p1[0] + tt * tt * p2[0]
            by = u * u * p0[1] + 2 * u * tt * p1[1] + tt * tt * p2[1]
            self._path.append((bx, by))

    def bezier_to(self, c1x: float, c1y: float, c2x: float, c2y: float, x: float, y: float) -> None:
        if not self._path:
            self._path.append(self._w(c1x, c1y))
        p0 = self._path[-1]
        p1 = self._w(c1x, c1y)
        p2 = self._w(c2x, c2y)
        p3 = self._w(x, y)
        for i in range(1, 13):
            tt = i / 12.0
            u = 1.0 - tt
            bx = u * u * u * p0[0] + 3 * u * u * tt * p1[0] + 3 * u * tt * tt * p2[0] + tt * tt * tt * p3[0]
            by = u * u * u * p0[1] + 3 * u * u * tt * p1[1] + 3 * u * tt * tt * p2[1] + tt * tt * tt * p3[1]
            self._path.append((bx, by))

    def close_path(self) -> None:
        if len(self._path) >= 2:
            self._path.append(self._path[0])

    # --- finishing ---------------------------------------------------------
    def stroke_path(self, width: float) -> None:
        if len(self._path) < 2:
            self._path = []
            return
        # Width is in *local* units when the motif sets it; scale through to
        # world units so consistent stroke gauges hold across save/scale layers.
        world_w = max(width * self._tx().uniform_scale(), 1e-6)
        self._stroke_paths.append((self._path, world_w))
        self._path = []

    def fill_path(self) -> None:
        if len(self._path) < 3:
            self._path = []
            return
        # Ensure closed ring.
        if self._path[0] != self._path[-1]:
            self._path.append(self._path[0])
        self._fill_paths.append(self._path)
        self._path = []

    def circle(self, cx: float, cy: float, r: float, *, fill: bool = True, stroke: float = 0.0) -> None:
        center_world = self._w(cx, cy)
        r_world = max(r * self._tx().uniform_scale(), 1e-6)
        if fill:
            self._disks.append((center_world[0], center_world[1], r_world))
        # Stroked-only circles are rare in the current motif set; treat them
        # as a fat disk minus a thin disk by adding two entries.
        if stroke > 0 and not fill:
            stroke_world = max(stroke * self._tx().uniform_scale(), 1e-6)
            self._disks.append((center_world[0], center_world[1], r_world + stroke_world / 2))

    # --- output ------------------------------------------------------------
    def finish(self, *, merge: bool = False) -> MultiPolygon:
        """Materialize accumulated paths into a MultiPolygon.

        The pen accumulates raw vertex lists during draw, so this is the only
        place we pay GEOS/Shapely costs:
          - Strokes: each polyline buffered INDIVIDUALLY at its exact width
            via the vectorized array API.
          - Fills: constructed per ring (``buffer(0)`` repair on demand).
          - Disks: each center buffered individually at its exact radius.

        Per-part buffering is load-bearing, not a style choice: buffering a
        whole MultiLineString makes GEOS union every member's buffer in one
        overlay — with thousands of mutually-overlapping vine strokes that
        noding committed gigabytes and froze the host (2026-06-10 bugchecks).
        Individual buffers are local and linear, and the concatenated
        (possibly overlapping) output covers the identical area, so the
        fill-only consumers (rasterize / SVG) are unaffected.

        ``merge=False`` (default) skips the final ``unary_union``; overlapping
        polygons rasterize identically and the union step is the most
        expensive op in the pipeline. Set ``merge=True`` for consumers that
        need clean topology.
        """
        out: list[Polygon] = []

        def extend(geom: BaseGeometry) -> None:
            if isinstance(geom, Polygon) and not geom.is_empty:
                out.append(geom)
            elif isinstance(geom, MultiPolygon):
                out.extend(g for g in geom.geoms if not g.is_empty)

        # --- strokes — one local buffer per polyline, exact width ---
        kept = [(p, w) for p, w in self._stroke_paths if len(p) >= 2]
        if kept:
            lines = np.array([LineString(p) for p, _ in kept], dtype=object)
            half_widths = np.array([w for _, w in kept], dtype=np.float64) / 2.0
            for buf in shapely.buffer(lines, half_widths, cap_style="flat", join_style="mitre"):
                extend(buf)

        # --- fills — per ring; ``buffer(0)`` repairs self-intersecting rings.
        for ring in self._fill_paths:
            if len(ring) < 4:
                continue
            try:
                poly = Polygon(ring)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                extend(poly)
            except Exception:  # noqa: BLE001 — drop malformed rings
                continue

        # --- disks — one local buffer per center, exact radius ---
        if self._disks:
            centers = shapely.points(np.array([(cx, cy) for cx, cy, _ in self._disks]))
            radii = np.array([r for _, _, r in self._disks], dtype=np.float64)
            for buf in shapely.buffer(centers, radii, quad_segs=12):
                extend(buf)

        if not out:
            return MultiPolygon()
        if merge:
            # KEEP: deliberate opt-in escape hatch. No caller passes
            # merge=True today (api.py uses merge=False explicitly), so this
            # union costs nothing at runtime — it exists for any future
            # consumer that needs clean topology (e.g. a real GDS boolean
            # pipeline) rather than the fill-only concatenation below.
            merged = unary_union(out)
            return ensure_multipolygon(merged)
        return MultiPolygon(out)
