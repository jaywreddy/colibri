"""Plantain: broad oval leaf with a strong central midrib and lateral veins."""
from __future__ import annotations

import math

from ...geometry import Mulberry32
from ...pen import Pen


def draw(pen: Pen, size: float, seed: int) -> None:
    _ = Mulberry32(seed)
    stroke = size * 0.018

    leaf_len = size * 1.0
    leaf_w = leaf_len * 0.45

    # Broad oval outline — anchor at base.
    pen.move_to(0.0, 0.0)
    pen.bezier_to(leaf_len * 0.25, leaf_w * 0.55, leaf_len * 0.7, leaf_w * 0.55, leaf_len, 0.0)
    pen.bezier_to(leaf_len * 0.7, -leaf_w * 0.55, leaf_len * 0.25, -leaf_w * 0.55, 0.0, 0.0)
    pen.close_path()
    pen.fill_path()

    # Midrib
    pen.move_to(0.0, 0.0)
    pen.line_to(leaf_len, 0.0)
    pen.stroke_path(stroke * 2.0)

    # Lateral veins — alternating fan from the midrib.
    n = 7
    for i in range(1, n + 1):
        frac = i / (n + 1)
        x = leaf_len * frac
        vein_len = leaf_w * 0.45 * math.sin(math.pi * frac) ** 0.7
        for side in (-1, 1):
            pen.move_to(x, 0.0)
            pen.quadratic_to(x + vein_len * 0.4, side * vein_len * 0.6, x + vein_len * 0.6, side * vein_len * 0.95)
            pen.stroke_path(stroke * 0.8)
