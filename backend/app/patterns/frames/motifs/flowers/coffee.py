"""Coffee sprig: short stem, paired oval leaves, jasmine-like blossom + cherries."""
from __future__ import annotations

import math

from ...geometry import Mulberry32
from ...pen import Pen


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)
    stroke = size * 0.02

    # Vertical stem; the sprite anchor is the bottom of the stem.
    stem_len = size * 0.55
    pen.move_to(0.0, 0.0)
    pen.line_to(0.0, stem_len)
    pen.stroke_path(stroke * 1.4)

    # Two pairs of opposing oval leaves at ~30% and ~70% of stem.
    for frac in (0.35, 0.7):
        cy = stem_len * frac
        for side in (-1, 1):
            pen.save()
            pen.translate(0.0, cy)
            pen.rotate(side * math.radians(55.0))
            leaf_len = size * 0.32 * (0.9 + 0.15 * rng.next_float())
            leaf_w = leaf_len * 0.32
            pen.move_to(0.0, 0.0)
            pen.bezier_to(leaf_len * 0.3, leaf_w, leaf_len * 0.7, leaf_w, leaf_len, 0.0)
            pen.bezier_to(leaf_len * 0.7, -leaf_w, leaf_len * 0.3, -leaf_w, 0.0, 0.0)
            pen.close_path()
            pen.fill_path()
            # Central vein
            pen.move_to(0.0, 0.0)
            pen.line_to(leaf_len, 0.0)
            pen.stroke_path(stroke * 0.7)
            pen.restore()

    # Crown blossom — 5-petal jasmine-ish star at the stem top.
    pen.save()
    pen.translate(0.0, stem_len + size * 0.08)
    for i in range(5):
        pen.save()
        pen.rotate(i * 2 * math.pi / 5)
        petal_len = size * 0.18
        petal_w = size * 0.08
        pen.move_to(0.0, 0.0)
        pen.bezier_to(petal_w, petal_len * 0.5, petal_w * 0.6, petal_len, 0.0, petal_len)
        pen.bezier_to(-petal_w * 0.6, petal_len, -petal_w, petal_len * 0.5, 0.0, 0.0)
        pen.close_path()
        pen.fill_path()
        pen.restore()
    pen.circle(0.0, 0.0, size * 0.04, fill=True)
    pen.restore()

    # Two cherries clustered near the lower leaf pair.
    cherry_r = size * 0.08
    pen.circle(-size * 0.12, stem_len * 0.4, cherry_r, fill=True, stroke=stroke * 0.5)
    pen.circle(size * 0.10, stem_len * 0.42, cherry_r * 0.95, fill=True, stroke=stroke * 0.5)
