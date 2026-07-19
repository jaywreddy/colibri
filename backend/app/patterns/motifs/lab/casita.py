from __future__ import annotations

"""Casita — a little finca (farmhouse) with a palm and mountains behind it.

"Building a life together / the Colombian countryside." The read: layered
MOUNTAIN ridges across the back, a small pitched-roof HOUSE (with door + window)
in the mid-ground, and a tall PALM leaning over it. Warm and homey. Authored in
a normalized 0..1 art box (Pillow y-down).
"""

import math

import numpy as np
from PIL import ImageDraw

from .._pillow import render_silhouette
from ._draw import Pen, bezier, chain


def _draw_casita(draw: ImageDraw.ImageDraw, n: int) -> None:
    p = Pen(draw, n)

    # ===================== MOUNTAINS (back) ================================
    # A SHALLOW ridge band across the upper third only — a thin range with dark
    # SKY above and dark GROUND below, so the house and palm read as distinct
    # gold objects in front rather than dissolving into a solid slab. The band
    # bottom sits at ~0.46 (well above the house eave), leaving open ground.
    band_bottom = 0.46
    back_ridge = [
        (0.02, 0.40), (0.16, 0.27), (0.28, 0.36), (0.42, 0.24),
        (0.58, 0.35), (0.72, 0.25), (0.86, 0.34), (0.98, 0.29),
        (0.98, band_bottom), (0.02, band_bottom),
    ]
    p.poly(back_ridge)
    # a lower front ridge for depth, drawn on top (a touch darker in real life;
    # here just a second overlapping silhouette).
    front_ridge = [
        (0.02, 0.45), (0.22, 0.36), (0.40, 0.44), (0.58, 0.37),
        (0.74, 0.45), (0.90, 0.39), (0.98, 0.44),
        (0.98, band_bottom + 0.02), (0.02, band_bottom + 0.02),
    ]
    p.poly(front_ridge)
    # a thin ground line so the house/palm have something to stand on.
    p.poly([(0.02, 0.865), (0.98, 0.865), (0.98, 0.885), (0.02, 0.885)])

    # ===================== HOUSE (mid-ground, right-of-center) =============
    # Simple finca: square body + pitched roof + a small chimney.
    hx0, hx1 = 0.52, 0.80        # house left/right walls
    hy_eave = 0.60              # where walls meet roof
    hy_base = 0.86              # ground / base of walls
    # walls (body)
    p.poly([(hx0, hy_eave), (hx1, hy_eave), (hx1, hy_base), (hx0, hy_base)])
    # pitched roof: a triangle overhanging the walls a touch.
    p.poly([(hx0 - 0.03, hy_eave + 0.005), (0.66, 0.48), (hx1 + 0.03, hy_eave + 0.005)])
    # chimney on the right slope.
    p.poly([(0.735, 0.52), (0.735, 0.44), (0.765, 0.44), (0.765, 0.55)])

    # ===================== PALM (leaning over the house) ===================
    # Trunk: a gently curved tapering column rising from the ground at the left
    # of the house and arching right over the roof.
    trunk_base = (0.40, 0.86)
    trunk_top = (0.50, 0.50)
    tc = bezier(trunk_base, (0.38, 0.72), (0.44, 0.58), trunk_top, n=24)
    lft, rgt = [], []
    for i in range(len(tc)):
        t = i / (len(tc) - 1)
        w = 0.022 * (1 - 0.55 * t)
        j = min(i + 1, len(tc) - 1)
        k = max(i - 1, 0)
        dx = tc[j][0] - tc[k][0]
        dy = tc[j][1] - tc[k][1]
        L = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / L, dx / L
        cxp, cyp = tc[i]
        lft.append((cxp + nx * w, cyp + ny * w))
        rgt.append((cxp - nx * w, cyp - ny * w))
    p.poly(lft + list(reversed(rgt)))

    # Palm fronds: long arching blades springing from the crown and DROOPING
    # under gravity — a coconut-palm crown, deliberately asymmetric (biased to
    # spread sideways, none straight up) so it never reads as a radial star.
    crown = trunk_top
    # (dir_x, dir_y_initial, droop, length) — dy_initial slightly up, droop pulls
    # the tip well down so each blade arcs over.
    fronds = [
        (-1.0, -0.35, 0.12, 0.22),   # far left, arcs down
        (-0.7, -0.7, 0.11, 0.21),
        (-0.3, -0.95, 0.09, 0.19),   # up-left
        (0.3, -0.95, 0.09, 0.19),    # up-right
        (0.7, -0.7, 0.11, 0.21),
        (1.0, -0.35, 0.12, 0.23),    # far right over the house
    ]
    for (dx, dy, droop, length) in fronds:
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        mid = (crown[0] + ux * length * 0.55, crown[1] + uy * length * 0.55 + droop * 0.5)
        tip = (crown[0] + ux * length, crown[1] + uy * length + droop)
        w = 0.024
        blade = chain(
            bezier(crown, (crown[0] + ux * 0.09 + nx * w, crown[1] + uy * 0.09 + ny * w),
                   (mid[0] + nx * w, mid[1] + ny * w), tip),
            bezier(tip, (mid[0] - nx * w, mid[1] - ny * w),
                   (crown[0] + ux * 0.09 - nx * w, crown[1] + uy * 0.09 - ny * w), crown),
        )
        p.poly(blade)
    # a couple of coconuts at the crown so it reads as a fruiting palm.
    p.ellipse(crown[0] - 0.01, crown[1] + 0.03, 0.016, fill=255)
    p.ellipse(crown[0] + 0.02, crown[1] + 0.035, 0.014, fill=255)

    # =======================================================================
    # NEGATIVE SPACE
    # =======================================================================
    # Door: a tall arch-topped opening on the house front.
    p.poly([(0.60, 0.86), (0.60, 0.72), (0.68, 0.72), (0.68, 0.86)])
    p.ellipse(0.64, 0.72, 0.04, 0.03, fill=0)  # arched top of the door
    # Window: a small square to the right of the door.
    p.poly([(0.71, 0.68), (0.77, 0.68), (0.77, 0.74), (0.71, 0.74)], fill=0)
    # Window cross-bars back in (muntins) so it reads as a window.
    p.line([(0.74, 0.68), (0.74, 0.74)], 0.006, fill=255)
    p.line([(0.71, 0.71), (0.77, 0.71)], 0.006, fill=255)
    # Roof eave line.
    p.line([(hx0 - 0.03, hy_eave + 0.005), (hx1 + 0.03, hy_eave + 0.005)], 0.005, fill=0)
    # Palm frond midribs so the crown reads as separate drooping fronds.
    for (dx, dy, droop, length) in fronds:
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        p.line([crown, (crown[0] + ux * length * 0.9, crown[1] + uy * length * 0.9 + droop * 0.85)],
               0.004, fill=0)


def casita_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — finca with palm + mountains."""
    del extent_um
    return render_silhouette(_draw_casita, n_grid)
