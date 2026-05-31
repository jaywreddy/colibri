"""Cattleya-style orchid: 3 sepals + 2 petals + ruffled labellum + column."""
from __future__ import annotations

import math

from ...geometry import Mulberry32
from ...pen import Pen


def _petal(pen: Pen, length: float, width: float, stroke: float) -> None:
    """Draw a single tear-drop petal centered along +x, tip at +length."""
    pen.move_to(0.0, 0.0)
    pen.bezier_to(length * 0.3, width * 0.6, length * 0.8, width * 0.5, length, 0.0)
    pen.bezier_to(length * 0.8, -width * 0.5, length * 0.3, -width * 0.6, 0.0, 0.0)
    pen.close_path()
    pen.fill_path()
    # Re-stroke the outline so the petal has a clean edge over the fill.
    pen.move_to(0.0, 0.0)
    pen.bezier_to(length * 0.3, width * 0.6, length * 0.8, width * 0.5, length, 0.0)
    pen.bezier_to(length * 0.8, -width * 0.5, length * 0.3, -width * 0.6, 0.0, 0.0)
    pen.close_path()
    pen.stroke_path(stroke)


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)
    stroke = size * 0.022

    # 3 sepals — top, lower-left, lower-right, pointing outward at 120° spacing.
    sepal_len = size * 0.55
    sepal_w = size * 0.18
    for angle_deg in (90.0, 210.0, 330.0):
        pen.save()
        pen.rotate(math.radians(angle_deg + rng.uniform(-6.0, 6.0)))
        _petal(pen, sepal_len, sepal_w, stroke)
        pen.restore()

    # 2 lateral petals (broader, between the upper sepal and lower sepals)
    petal_len = size * 0.42
    petal_w = size * 0.28
    for angle_deg in (40.0, 140.0):
        pen.save()
        pen.rotate(math.radians(angle_deg))
        _petal(pen, petal_len, petal_w, stroke)
        pen.restore()

    # Ruffled labellum: trumpet-shaped lower lip, drawn as a wider triple-lobed petal.
    pen.save()
    pen.rotate(math.radians(270.0))
    lip_len = size * 0.50
    lip_w = size * 0.34
    pen.move_to(0.0, 0.0)
    pen.bezier_to(lip_len * 0.15, lip_w * 0.4, lip_len * 0.35, lip_w * 0.8, lip_len * 0.55, lip_w * 0.5)
    pen.bezier_to(lip_len * 0.8, lip_w * 0.7, lip_len, lip_w * 0.3, lip_len, 0.0)
    pen.bezier_to(lip_len, -lip_w * 0.3, lip_len * 0.8, -lip_w * 0.7, lip_len * 0.55, -lip_w * 0.5)
    pen.bezier_to(lip_len * 0.35, -lip_w * 0.8, lip_len * 0.15, -lip_w * 0.4, 0.0, 0.0)
    pen.close_path()
    pen.fill_path()
    pen.restore()

    # Column (center post)
    pen.circle(0.0, 0.0, size * 0.06, fill=True)
