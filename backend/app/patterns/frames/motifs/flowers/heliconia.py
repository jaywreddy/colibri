"""Heliconia: alternating triangular bracts climbing a central stalk."""
from __future__ import annotations

from ...geometry import Mulberry32
from ...pen import Pen


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)
    stroke = size * 0.022

    n_bracts = 6
    stalk_len = size * 0.95
    # Central stalk (zigzag — the actual heliconia is a tight zigzag of bract bases).
    pen.move_to(0.0, 0.0)
    for i in range(1, n_bracts + 1):
        x_offset = (-1 if i % 2 else 1) * size * 0.02
        pen.line_to(x_offset, stalk_len * (i / n_bracts))
    pen.stroke_path(stroke)

    # Alternating bracts — each a curved triangular boat shape.
    for i in range(n_bracts):
        side = -1 if i % 2 else 1
        y0 = stalk_len * (i / n_bracts)
        y1 = stalk_len * ((i + 1) / n_bracts)
        cy = 0.5 * (y0 + y1)
        bract_len = size * 0.42 * (0.85 + 0.20 * rng.next_float())
        bract_h = (y1 - y0) * 1.05
        pen.save()
        pen.translate(0.0, cy)
        if side < 0:
            pen.scale(-1.0, 1.0)
        # Bract shape: starts at stalk, sweeps out and back to a sharp tip.
        pen.move_to(0.0, -bract_h * 0.5)
        pen.bezier_to(bract_len * 0.55, -bract_h * 0.6, bract_len * 0.95, -bract_h * 0.1, bract_len, bract_h * 0.1)
        pen.bezier_to(bract_len * 0.7, bract_h * 0.55, bract_len * 0.25, bract_h * 0.6, 0.0, bract_h * 0.5)
        pen.close_path()
        pen.fill_path()
        # Vein down the keel
        pen.move_to(0.0, 0.0)
        pen.line_to(bract_len * 0.85, 0.0)
        pen.stroke_path(stroke * 0.7)
        pen.restore()
