"""Pen that emits raw SVG path strings — bypasses Shapely entirely.

Built so the fab export survives 30 mm plates: the Shapely polygon path
constructs ~6000 Polygon objects per face and calls ``buffer`` per stroke,
which OOMs / timeouts at this scale. This pen accumulates vertex lists like
RasterPen does, then converts each to a single SVG <path> element.

Stroke widths land as ``stroke-width`` on the path (drawn as a stroked
polyline rather than buffered into a polygon). Fills become closed paths
with the gold fill color. Disks become ``<circle>`` elements directly.

This is the canonical fab output path — engravers consume SVG with one
gold color and per-layer groups, which is exactly what this emits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from io import StringIO

from .pen import Pen


@dataclass
class _Affine:
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
        sx = math.hypot(self.a, self.b)
        sy = math.hypot(self.c, self.d)
        return 0.5 * (sx + sy)


class SvgPen(Pen):
    """Emits SVG <path>/<circle> fragments accumulated as strings.

    Coordinates are in the same μm space the motifs use. The caller stamps
    the surrounding <svg> wrapper. Y is FLIPPED (SVG y-down vs. our y-up
    convention) when writing path data.
    """

    def __init__(self, fill: str = "#E6BC50", stroke: str = "#E6BC50") -> None:
        self._tx_stack: list[_Affine] = [_Affine()]
        self._path: list[tuple[float, float]] = []  # world μm
        # Strokes go into the main buffer (per-path, varying width).
        self._buf = StringIO()
        # Fills concatenate into one giant `d` attribute on a single <path>.
        # Each subpath = a closed ring; fill-rule="nonzero" lets adjacent
        # rings render as separate filled regions. This collapses ~5000
        # <path> elements per face into 1 element — both faster to parse and
        # ~10× smaller on disk (no per-element attribute repetition).
        self._fill_subpaths: list[str] = []
        # Circles likewise — collect as a single <path> of M/A/A/Z subpaths
        # rather than ~500 <circle> elements per face.
        self._circle_subpaths: list[str] = []
        self._fill = fill
        self._stroke = stroke

    # --- transform stack ---
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

    def _w(self, x: float, y: float) -> tuple[float, float]:
        return self._tx().apply(x, y)

    # --- path construction ---
    def move_to(self, x: float, y: float) -> None:
        self._path = [self._w(x, y)]

    def line_to(self, x: float, y: float) -> None:
        self._path.append(self._w(x, y))

    def quadratic_to(self, cx: float, cy: float, x: float, y: float) -> None:
        if not self._path:
            self._path.append(self._w(cx, cy))
        p0 = self._path[-1]
        p1 = self._w(cx, cy)
        p2 = self._w(x, y)
        # Flatten to ~8 line segments — SVG could carry Q natively, but the
        # mixed transforms here mean the control points need composing too;
        # rather than emit Q after `apply` we just polyline-approximate.
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

    # --- finishing ---
    def _path_data(self) -> str:
        if not self._path:
            return ""
        parts = []
        x0, y0 = self._path[0]
        parts.append(f"M{x0:.2f} {-y0:.2f}")
        for x, y in self._path[1:]:
            parts.append(f"L{x:.2f} {-y:.2f}")
        return " ".join(parts)

    def stroke_path(self, width: float) -> None:
        if len(self._path) < 2:
            self._path = []
            return
        world_w = max(width * self._tx().uniform_scale(), 1e-6)
        d = self._path_data()
        self._buf.write(
            f'<path d="{d}" fill="none" stroke="{self._stroke}" '
            f'stroke-width="{world_w:.2f}" stroke-linecap="butt" stroke-linejoin="miter"/>'
        )
        self._path = []

    def fill_path(self) -> None:
        if len(self._path) < 3:
            self._path = []
            return
        # Batch into a single subpath string — joined with all other fills at
        # finish() into one giant <path>. Each subpath ends with Z so the ring
        # is explicit.
        self._fill_subpaths.append(self._path_data() + " Z")
        self._path = []

    def circle(self, cx: float, cy: float, r: float, *, fill: bool = True, stroke: float = 0.0) -> None:
        center = self._w(cx, cy)
        r_world = max(r * self._tx().uniform_scale(), 1e-6)
        cx_s, cy_s = center[0], -center[1]  # flip y
        if fill:
            # SVG arc trick: M(cx-r,cy) A r,r 0 1,0 cx+r,cy A r,r 0 1,0 cx-r,cy Z
            self._circle_subpaths.append(
                f"M{cx_s - r_world:.2f} {cy_s:.2f}"
                f"A{r_world:.2f} {r_world:.2f} 0 1 0 {cx_s + r_world:.2f} {cy_s:.2f}"
                f"A{r_world:.2f} {r_world:.2f} 0 1 0 {cx_s - r_world:.2f} {cy_s:.2f} Z"
            )
        if stroke > 0 and not fill:
            stroke_world = max(stroke * self._tx().uniform_scale(), 1e-6)
            # Stroked-only circles are rare; emit as a one-off element.
            self._buf.write(
                f'<circle cx="{cx_s:.2f}" cy="{cy_s:.2f}" r="{r_world:.2f}" '
                f'fill="none" stroke="{self._stroke}" stroke-width="{stroke_world:.2f}"/>'
            )

    def finish(self) -> str:  # type: ignore[override]
        """Return the accumulated SVG fragment (no outer <svg> wrapper)."""
        out = StringIO()
        # All fills as one path
        if self._fill_subpaths:
            out.write(
                f'<path d="{" ".join(self._fill_subpaths)}" '
                f'fill="{self._fill}" stroke="none" fill-rule="nonzero"/>'
            )
        # All disks as one path
        if self._circle_subpaths:
            out.write(
                f'<path d="{" ".join(self._circle_subpaths)}" '
                f'fill="{self._fill}" stroke="none" fill-rule="nonzero"/>'
            )
        # Strokes (and stroked-only circles) — kept per-element
        out.write(self._buf.getvalue())
        return out.getvalue()
