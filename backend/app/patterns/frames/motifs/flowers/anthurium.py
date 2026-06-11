"""Anthurium: glossy heart-shaped spathe with a protruding spadix."""
from __future__ import annotations

from ...geometry import Mulberry32
from ...pen import Pen


def draw(pen: Pen, size: float, seed: int) -> None:
    _ = Mulberry32(seed)  # parameter retained for signature parity
    stroke = size * 0.022

    # The heart-shaped spathe — symmetric across +y axis, two lobes meeting
    # at the top, tapering to a point at the bottom.
    s = size * 0.55
    pen.move_to(0.0, -s)                                  # tip
    pen.bezier_to(s * 0.55, -s * 0.55, s * 0.9, -s * 0.1, s * 0.75, s * 0.45)  # right side up
    pen.bezier_to(s * 0.45, s * 0.85, s * 0.05, s * 0.55, 0.0, s * 0.30)        # right lobe top
    pen.bezier_to(-s * 0.05, s * 0.55, -s * 0.45, s * 0.85, -s * 0.75, s * 0.45)  # left lobe top
    pen.bezier_to(-s * 0.9, -s * 0.1, -s * 0.55, -s * 0.55, 0.0, -s)              # left side down
    pen.close_path()
    pen.fill_path()
    # Outline
    pen.move_to(0.0, -s)
    pen.bezier_to(s * 0.55, -s * 0.55, s * 0.9, -s * 0.1, s * 0.75, s * 0.45)
    pen.bezier_to(s * 0.45, s * 0.85, s * 0.05, s * 0.55, 0.0, s * 0.30)
    pen.bezier_to(-s * 0.05, s * 0.55, -s * 0.45, s * 0.85, -s * 0.75, s * 0.45)
    pen.bezier_to(-s * 0.9, -s * 0.1, -s * 0.55, -s * 0.55, 0.0, -s)
    pen.close_path()
    pen.stroke_path(stroke)

    # Spadix — yellow-orange tail sticking up out of the cleft.
    sp_x0 = -size * 0.04
    sp_x1 = size * 0.04
    sp_y0 = s * 0.10
    sp_y1 = s * 0.85
    pen.move_to(sp_x0, sp_y0)
    pen.bezier_to(sp_x0 * 0.6, sp_y1 * 0.5, sp_x1 * 0.6, sp_y1 * 0.5, sp_x1, sp_y0)
    pen.bezier_to(sp_x1 * 0.4, sp_y1 * 0.95, sp_x0 * 0.4, sp_y1 * 0.95, sp_x0, sp_y0)
    pen.close_path()
    pen.fill_path()
