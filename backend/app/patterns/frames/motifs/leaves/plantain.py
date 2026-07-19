"""Plantain: a broad, smooth oval leaf with a strong central midrib."""
from __future__ import annotations

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

    # Midrib only. Lateral veins were removed: on the single-color gold mask
    # internal veins are invisible (gold on gold), and their stroked tips poked
    # PAST the leaf margin, reading as spiky burrs on the silhouette edge.
    pen.move_to(0.0, 0.0)
    pen.line_to(leaf_len * 0.94, 0.0)
    pen.stroke_path(stroke * 2.0)
