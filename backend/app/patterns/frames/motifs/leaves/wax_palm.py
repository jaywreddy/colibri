"""Wax palm: tall pinnate frond with paired leaflets along a central rachis.

The Colombian national tree. Redrawn so the leaflets are broad, lance-shaped
blades (not hairline slivers) that stay legible at frame-raster scale, swept
forward off a gently arched rachis for a graceful palm silhouette.
"""
from __future__ import annotations

import math

from ...geometry import Mulberry32
from ...pen import Pen


def _blade(pen: Pen, length: float, width: float) -> None:
    """A lance-shaped leaflet blade along +x, tip at +length."""
    pen.move_to(0.0, 0.0)
    pen.bezier_to(length * 0.22, width * 0.55, length * 0.7, width * 0.42, length, 0.0)
    pen.bezier_to(length * 0.7, -width * 0.42, length * 0.22, -width * 0.55, 0.0, 0.0)
    pen.close_path()
    pen.fill_path()


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)
    stroke = size * 0.02

    # Arched rachis for a natural droop.
    rachis_len = size * 1.05
    arch = size * 0.18
    pen.move_to(0.0, 0.0)
    pen.quadratic_to(rachis_len * 0.5, arch * 0.5, rachis_len, arch)
    pen.stroke_path(stroke * 1.7)

    n_pairs = 7
    forward_deg = 30.0
    for i in range(1, n_pairs + 1):
        frac = i / (n_pairs + 1)
        x = rachis_len * frac
        y = arch * frac * frac
        taper = math.sin(math.pi * frac) ** 0.5
        blade_len = size * 0.40 * taper * (0.9 + 0.15 * rng.next_float())
        blade_w = blade_len * 0.28
        for side in (-1, 1):
            pen.save()
            pen.translate(x, y)
            pen.rotate(side * math.radians(58.0) - side * math.radians(forward_deg) * (1.0 - frac * 0.5))
            _blade(pen, blade_len, blade_w)
            pen.restore()
    pen.save()
    pen.translate(rachis_len, arch)
    _blade(pen, size * 0.24, size * 0.07)
    pen.restore()
