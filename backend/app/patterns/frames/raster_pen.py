"""PIL-direct pen — paints into an Image without going through Shapely.

Shapely's Polygon constructor + validity check is ~3 ms per polygon. A dense
frame scene has 5000+ motif polygons, which costs ~25 s just in shapely. The
raster path is ~100× faster because PIL's ImageDraw is implemented in C and
takes raw vertex tuples without object construction.

Frames don't need vector output for the live preview / fab PNG, so we feed
this pen during materialize_plate. The lazy SVG export uses ``SvgPen``
(svg_pen.py); ``ShapelyPen`` remains for polygon-space consumers/tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageDraw

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


class RasterPen(Pen):
    """Pen backend that paints directly to a PIL ``L``-mode image.

    Coordinates passed to draw operations are in plate-μm space (origin at
    plate center). The pen converts to pixel coordinates against the supplied
    ``extent_um`` and ``pixel_pitch_um``. Foreground gold = 255.
    """

    def __init__(self, image: Image.Image, extent_um: tuple[float, float], pixel_pitch_um: float) -> None:
        self._img = image
        self._draw = ImageDraw.Draw(image)
        self._w_um, self._h_um = extent_um
        self._pitch = pixel_pitch_um
        self._w_px, self._h_px = image.size
        self._tx_stack: list[_Affine] = [_Affine()]
        self._path: list[tuple[float, float]] = []  # world μm

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

    def _world(self, x: float, y: float) -> tuple[float, float]:
        return self._tx().apply(x, y)

    def _to_px(self, xy: tuple[float, float]) -> tuple[float, float]:
        x, y = xy
        # Plate origin is at center; image origin is top-left, y-down.
        px = (x + self._w_um / 2.0) / self._pitch
        py = (self._h_um / 2.0 - y) / self._pitch
        return px, py

    # --- path construction ---
    def move_to(self, x: float, y: float) -> None:
        self._path = [self._world(x, y)]

    def line_to(self, x: float, y: float) -> None:
        if not self._path:
            self._path.append(self._world(x, y))
            return
        self._path.append(self._world(x, y))

    def quadratic_to(self, cx: float, cy: float, x: float, y: float) -> None:
        if not self._path:
            self._path.append(self._world(cx, cy))
        p0 = self._path[-1]
        p1 = self._world(cx, cy)
        p2 = self._world(x, y)
        for i in range(1, 9):
            tt = i / 8.0
            u = 1.0 - tt
            bx = u * u * p0[0] + 2 * u * tt * p1[0] + tt * tt * p2[0]
            by = u * u * p0[1] + 2 * u * tt * p1[1] + tt * tt * p2[1]
            self._path.append((bx, by))

    def bezier_to(self, c1x: float, c1y: float, c2x: float, c2y: float, x: float, y: float) -> None:
        if not self._path:
            self._path.append(self._world(c1x, c1y))
        p0 = self._path[-1]
        p1 = self._world(c1x, c1y)
        p2 = self._world(c2x, c2y)
        p3 = self._world(x, y)
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
    def stroke_path(self, width: float) -> None:
        if len(self._path) < 2:
            self._path = []
            return
        world_w = max(width * self._tx().uniform_scale(), 1e-6)
        px_w = max(1, int(round(world_w / self._pitch)))
        pts = [self._to_px(p) for p in self._path]
        # ImageDraw.line takes a flat (x,y,x,y,...) or list of pairs; either
        # works. Width-1 lines bypass the joint logic which is slower.
        self._draw.line(pts, fill=255, width=px_w, joint="curve")
        self._path = []

    def fill_path(self) -> None:
        if len(self._path) < 3:
            self._path = []
            return
        pts = [self._to_px(p) for p in self._path]
        try:
            self._draw.polygon(pts, fill=255)
        except Exception:  # noqa: BLE001 — drop malformed rings
            pass
        self._path = []

    def circle(self, cx: float, cy: float, r: float, *, fill: bool = True, stroke: float = 0.0) -> None:
        center_world = self._world(cx, cy)
        r_world = max(r * self._tx().uniform_scale(), 1e-6)
        px_cx, px_cy = self._to_px(center_world)
        px_r = max(0.5, r_world / self._pitch)
        bbox = (px_cx - px_r, px_cy - px_r, px_cx + px_r, px_cy + px_r)
        if fill:
            self._draw.ellipse(bbox, fill=255)
        if stroke > 0:
            stroke_world = max(stroke * self._tx().uniform_scale(), 1e-6)
            self._draw.ellipse(bbox, outline=255, width=max(1, int(round(stroke_world / self._pitch))))
