"""Pen backend that accumulates strokes + fills into a Shapely MultiPolygon.

Stroked polylines become thin polygons via ``LineString.buffer(width/2)`` with
flat caps so neighboring segments butt cleanly. Filled subpaths become
polygons. The pen flushes its accumulated geometry on demand via
``finish() -> MultiPolygon`` so callers can union it onto a layer.

Transforms are tracked as a 2x3 affine matrix; coordinates from motifs pass
through ``_xform`` before joining the accumulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
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
        # Accumulators — flushed by ``finish``.
        self._stroke_lines: list[tuple[LineString, float]] = []  # (line, world-space width)
        self._fill_polys: list[Polygon] = []
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
        try:
            line = LineString(self._path)
        except Exception:
            self._path = []
            return
        self._stroke_lines.append((line, world_w))
        self._path = []

    def fill_path(self) -> None:
        if len(self._path) < 3:
            self._path = []
            return
        # Ensure closed ring for shapely.
        if self._path[0] != self._path[-1]:
            self._path.append(self._path[0])
        try:
            poly = Polygon(self._path)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if isinstance(poly, Polygon) and not poly.is_empty:
                self._fill_polys.append(poly)
            elif isinstance(poly, MultiPolygon):
                self._fill_polys.extend(p for p in poly.geoms if not p.is_empty)
        except Exception:
            pass
        self._path = []

    def circle(self, cx: float, cy: float, r: float, *, fill: bool = True, stroke: float = 0.0) -> None:
        center_world = self._w(cx, cy)
        r_world = max(r * self._tx().uniform_scale(), 1e-6)
        disk = Point(center_world).buffer(r_world, quad_segs=18)
        if fill:
            self._fill_polys.append(disk)
        if stroke > 0:
            stroke_world = max(stroke * self._tx().uniform_scale(), 1e-6)
            ring = disk.boundary.buffer(stroke_world / 2, cap_style=2)
            if isinstance(ring, Polygon):
                self._fill_polys.append(ring)
            elif isinstance(ring, MultiPolygon):
                self._fill_polys.extend(ring.geoms)

    # --- output ------------------------------------------------------------
    def finish(self) -> MultiPolygon:
        polys: list[BaseGeometry] = list(self._fill_polys)
        for line, w in self._stroke_lines:
            polys.append(line.buffer(w / 2, cap_style=2, join_style=2))  # flat cap, mitre join
        if not polys:
            return MultiPolygon()
        merged = unary_union(polys)
        return ensure_multipolygon(merged)
