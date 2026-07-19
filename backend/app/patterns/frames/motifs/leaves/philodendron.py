"""Philodendron: a graceful cordate (heart) leaf with a drawn-out drip tip.

Earlier versions chased a literal split-leaf *Monstera* — but at frame-raster
scale the deep sinuses (or even gentle scallops) read as an angular, stepped
gold blob, the ugliest shape in the garland. What actually reads elegant next
to the plantain's smooth oval is a SMOOTH cordate silhouette: a broad heart
shoulder at the base, a single fluid widest point, then a long tapering sweep to
a fine drip tip. It stays distinct from the plantain (heart base + acuminate
drip tip vs. symmetric ellipse) while keeping a clean, flowing margin.

Anchor at the petiole (origin), midrib along +x.
"""
from __future__ import annotations

from ...geometry import Mulberry32
from ...pen import Pen


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)
    stroke = size * 0.02

    leaf_len = size * 1.10
    hw = leaf_len * 0.40 * (0.94 + 0.12 * rng.next_float())

    # Margin control points as (x_frac, halfwidth_frac). SMOOTH and monotonic:
    # a fast heart-shoulder rise near the base, one fluid widest point at ~28 %,
    # then a long uninterrupted taper to a fine acuminate drip tip. No sinuses —
    # the flowing margin is what reads as elegant at frame scale.
    prof = [
        (0.00, 0.04),   # petiole base
        (0.05, 0.52),   # heart shoulder rises fast (cordate base)
        (0.14, 0.86),   # shoulder crown
        (0.28, 0.98),   # widest point
        (0.46, 0.90),
        (0.64, 0.70),
        (0.80, 0.44),
        (0.92, 0.20),   # ease into the drip tip
        (1.00, 0.0),    # fine acuminate tip
    ]

    def edge(sign: float, reverse: bool) -> None:
        pts = prof if not reverse else list(reversed(prof))
        for k, (fx, fw) in enumerate(pts):
            x = leaf_len * fx
            y = sign * hw * fw
            if k == 0:
                pen.line_to(x, y)
            else:
                # Curve into each control point for soft lobe crowns + rounded
                # sinus floors (a straight zigzag would look mechanical).
                px, pfw = pts[k - 1]
                mx = leaf_len * 0.5 * (px + fx)
                my = sign * hw * 0.5 * (pfw + fw)
                pen.quadratic_to(mx, my, x, y)

    pen.move_to(0.0, 0.0)
    edge(+1.0, reverse=False)   # top margin base -> tip
    edge(-1.0, reverse=True)    # bottom margin tip -> base
    pen.close_path()
    pen.fill_path()

    # Midrib.
    pen.move_to(0.0, 0.0)
    pen.line_to(leaf_len * 0.95, 0.0)
    pen.stroke_path(stroke * 1.7)
