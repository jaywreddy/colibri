"""Pinnate fern frond: a rachis bearing paired leaflets (pinnae).

Redrawn for plate-raster legibility. The old version stacked pinnae-of-pinnules
(a fractal comb) whose sub-features fell below the ~33μm raster pitch and turned
into a grainy sawtooth. Here each pinna is a single broad, softly-curved leaflet
lobe — a clean silhouette that reads as a frond at frame scale while keeping the
characteristic tapered, forward-swept feather shape.
"""
from __future__ import annotations

import math

from ...geometry import Mulberry32
from ...pen import Pen


def _leaflet(pen: Pen, length: float, width: float) -> None:
    """A single softly-pointed leaflet lobe along +x, tip at +length."""
    pen.move_to(0.0, 0.0)
    pen.bezier_to(length * 0.28, width * 0.62, length * 0.72, width * 0.5, length, 0.0)
    pen.bezier_to(length * 0.72, -width * 0.5, length * 0.28, -width * 0.62, 0.0, 0.0)
    pen.close_path()
    pen.fill_path()


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)
    stroke = size * 0.02

    # Main rachis — a gentle arc gives the frond life instead of a ruler line.
    rachis_len = size * 1.0
    arch = size * 0.14
    pen.move_to(0.0, 0.0)
    pen.quadratic_to(rachis_len * 0.5, arch * 0.6, rachis_len, arch)
    pen.stroke_path(stroke * 1.6)

    n_pinnae = 6
    for i in range(1, n_pinnae + 1):
        frac = i / (n_pinnae + 1)
        x = rachis_len * frac
        y = arch * frac * frac  # follow the rachis arc
        taper = math.sin(math.pi * frac) ** 0.5
        leaflet_len = size * 0.42 * taper * (0.9 + 0.15 * rng.next_float())
        leaflet_w = leaflet_len * 0.42
        # Forward sweep: leaflets angle toward the tip like a real frond.
        for side in (-1, 1):
            pen.save()
            pen.translate(x, y)
            pen.rotate(side * math.radians(52.0) - side * math.radians(18.0) * (1.0 - frac))
            _leaflet(pen, leaflet_len, leaflet_w)
            pen.restore()
    # A terminal leaflet closes the frond tip cleanly.
    pen.save()
    pen.translate(rachis_len, arch)
    _leaflet(pen, size * 0.26, size * 0.10)
    pen.restore()
