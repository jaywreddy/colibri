"""Bipinnate fern frond: a rachis bearing paired pinnae, each with paired pinnules."""
from __future__ import annotations

import math

from ...geometry import Mulberry32
from ...pen import Pen


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)
    stroke = size * 0.015

    # Main rachis
    rachis_len = size * 1.0
    pen.move_to(0.0, 0.0)
    pen.line_to(rachis_len, 0.0)
    pen.stroke_path(stroke * 1.3)

    n_pinnae = 7
    for i in range(1, n_pinnae + 1):
        frac = i / (n_pinnae + 1)
        x = rachis_len * frac
        taper = math.sin(math.pi * frac) ** 0.6
        pinna_len = size * 0.30 * taper
        for side in (-1, 1):
            pen.save()
            pen.translate(x, 0.0)
            pen.rotate(side * math.radians(55.0))
            # Sub-rachis
            pen.move_to(0.0, 0.0)
            pen.line_to(pinna_len, 0.0)
            pen.stroke_path(stroke * 0.9)
            # Pinnules — short opposing teardrops along the sub-rachis.
            n_pinnules = 5
            for j in range(1, n_pinnules + 1):
                pf = j / (n_pinnules + 1)
                px = pinna_len * pf
                pt = math.sin(math.pi * pf) ** 0.5
                p_len = pinna_len * 0.32 * pt * (0.85 + 0.20 * rng.next_float())
                p_w = p_len * 0.45
                for pside in (-1, 1):
                    pen.save()
                    pen.translate(px, 0.0)
                    pen.rotate(pside * math.radians(60.0))
                    pen.move_to(0.0, 0.0)
                    pen.bezier_to(p_len * 0.3, p_w * 0.6, p_len * 0.7, p_w * 0.5, p_len, 0.0)
                    pen.bezier_to(p_len * 0.7, -p_w * 0.5, p_len * 0.3, -p_w * 0.6, 0.0, 0.0)
                    pen.close_path()
                    pen.fill_path()
                    pen.restore()
            pen.restore()
