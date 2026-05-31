"""Philodendron: lobed leaf with deep cuts (an abstracted Monstera-like profile)."""
from __future__ import annotations

import math

from ...geometry import Mulberry32
from ...pen import Pen


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)
    stroke = size * 0.018

    leaf_len = size * 1.0
    leaf_w = leaf_len * 0.55

    # Construct the outline as a polyline with alternating lobes — base at origin,
    # tip along +x. We walk the top edge from base→tip, then the bottom back to
    # base. Lobes are sinusoidal cuts in y.
    n_lobes = 5
    top_pts: list[tuple[float, float]] = [(0.0, 0.0)]
    for i in range(1, 2 * n_lobes + 1):
        frac = i / (2 * n_lobes + 1)
        x = leaf_len * frac
        # Even step → outer crest; odd step → inner cut.
        if i % 2 == 1:
            y = leaf_w * 0.5 * math.sin(math.pi * frac) ** 0.7
        else:
            y = leaf_w * 0.18 * math.sin(math.pi * frac) ** 0.7
        # Pull the inner cuts toward the midrib to fake the fenestration.
        if i % 2 == 0:
            y *= 0.4
        top_pts.append((x, y * (0.9 + 0.15 * rng.next_float())))
    top_pts.append((leaf_len, 0.0))

    # Bottom mirror
    bottom_pts = [(x, -y) for x, y in reversed(top_pts[1:-1])]

    pen.move_to(top_pts[0][0], top_pts[0][1])
    for x, y in top_pts[1:]:
        pen.line_to(x, y)
    for x, y in bottom_pts:
        pen.line_to(x, y)
    pen.close_path()
    pen.fill_path()

    # Midrib
    pen.move_to(0.0, 0.0)
    pen.line_to(leaf_len, 0.0)
    pen.stroke_path(stroke * 1.6)
