"""Pen abstraction — the drawing interface every motif uses.

Each motif (flowers/leaves) takes a Pen + the sprite's local size and emits
strokes/fills against the pen's current transform. Two backends:

- ``ShapelyPen`` (this package, in shapely_pen.py): collects strokes and fills
  into a ``MultiPolygon`` for the lithography mask.
- ``Canvas2DPen`` (frontend): paints to the preview canvas with the 3-pass
  gold look (dark base + highlight + faint fill).

This file defines only the abstract surface. Concrete pens implement it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Pen(ABC):
    """Tiny stateful drawing interface — moveTo/lineTo + transform stack.

    A motif's draw function operates entirely against this API. The motif
    knows nothing about the output format; the pen handles the conversion.

    All coordinates are in the pen's *current* local frame (origin moves with
    save/translate/rotate/scale).
    """

    # --- transform stack ---------------------------------------------------
    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def restore(self) -> None: ...

    @abstractmethod
    def translate(self, dx: float, dy: float) -> None: ...

    @abstractmethod
    def rotate(self, radians: float) -> None: ...

    @abstractmethod
    def scale(self, sx: float, sy: float | None = None) -> None: ...

    # --- path construction -------------------------------------------------
    @abstractmethod
    def move_to(self, x: float, y: float) -> None: ...

    @abstractmethod
    def line_to(self, x: float, y: float) -> None: ...

    @abstractmethod
    def quadratic_to(self, cx: float, cy: float, x: float, y: float) -> None: ...

    @abstractmethod
    def bezier_to(self, c1x: float, c1y: float, c2x: float, c2y: float, x: float, y: float) -> None: ...

    @abstractmethod
    def close_path(self) -> None: ...

    # --- finishing ---------------------------------------------------------
    @abstractmethod
    def stroke_path(self, width: float) -> None:
        """Emit the current path as a stroked polyline of the given width (μm)."""

    @abstractmethod
    def fill_path(self) -> None:
        """Emit the current path as a filled polygon (interior gold)."""

    @abstractmethod
    def circle(self, cx: float, cy: float, r: float, *, fill: bool = True, stroke: float = 0.0) -> None:
        """Convenience for petal centers, berries, cherries, etc.

        Concrete backends may bypass the path-construction protocol for this
        and emit a polygonal approximation directly.
        """
