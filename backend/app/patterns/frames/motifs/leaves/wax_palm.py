"""Wax palm: tall pinnate frond with paired thin leaflets along a central rachis."""
from __future__ import annotations

import math

from ...geometry import Mulberry32
from ...pen import Pen


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)
    stroke = size * 0.018

    # Anchor at the petiole base; rachis points along +x.
    rachis_len = size * 1.0
    pen.move_to(0.0, 0.0)
    pen.line_to(rachis_len, 0.0)
    pen.stroke_path(stroke * 1.5)

    n_pairs = 9
    for i in range(1, n_pairs + 1):
        frac = i / (n_pairs + 1)
        x = rachis_len * frac
        # Leaflet length tapers toward the tip; thinner near base.
        taper = math.sin(math.pi * frac) ** 0.6
        leaflet_len = size * 0.32 * taper * (0.9 + 0.18 * rng.next_float())
        leaflet_w = leaflet_len * 0.10
        # Leaflets angle slightly forward (away from base).
        forward_deg = 32.0
        for side in (-1, 1):
            pen.save()
            pen.translate(x, 0.0)
            pen.rotate(side * math.radians(60.0) + math.radians(-forward_deg * side))
            pen.move_to(0.0, 0.0)
            pen.bezier_to(leaflet_len * 0.4, leaflet_w, leaflet_len * 0.8, leaflet_w * 0.6, leaflet_len, 0.0)
            pen.bezier_to(leaflet_len * 0.8, -leaflet_w * 0.4, leaflet_len * 0.4, -leaflet_w * 0.6, 0.0, 0.0)
            pen.close_path()
            pen.fill_path()
            pen.restore()
