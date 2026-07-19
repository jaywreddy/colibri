from __future__ import annotations

"""Coffee branch — Colombian coffee: an arching stem with opposite lance-shaped
leaves and clustered round cherries nestled at the leaf axils.

Botanical read for *Coffea arabica*: leaves are OPPOSITE (paired across the
stem), glossy, lance/ovate with a pointed tip and a clear midrib; cherries grow
in tight CLUSTERS hugging the stem at the axils. Authored as a gently arching
branch rising from lower-left to upper-right, three leaf pairs, two cherry
clusters. Normalized 0..1 art box, Pillow y-down.
"""

import math

import numpy as np
from PIL import ImageDraw

from .._pillow import render_silhouette
from ._draw import Pen, bezier, chain


# main stem control points (lower-left -> upper-right arch)
_STEM = [(0.18, 0.90), (0.34, 0.72), (0.50, 0.52), (0.70, 0.30)]


def _leaf(p: Pen, base, tip, width, bow):
    """A lance/ovate leaf as two mirrored Bézier edges from base to tip.

    `bow` (+/-) bends the leaf to one side of the base->tip axis so opposite
    leaves fan apart naturally. A midrib crease is punched afterward.
    """
    bx, by = base
    tx, ty = tip
    dx, dy = tx - bx, ty - by
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L           # along-leaf unit
    nx, ny = -uy, ux                  # perpendicular unit
    # side control offsets
    w = width
    c1 = (bx + ux * 0.30 * L + nx * (w + bow), by + uy * 0.30 * L + ny * (w + bow))
    c2 = (bx + ux * 0.72 * L + nx * (0.85 * w + bow), by + uy * 0.72 * L + ny * (0.85 * w + bow))
    c3 = (bx + ux * 0.72 * L - nx * (0.85 * w - bow), by + uy * 0.72 * L - ny * (0.85 * w - bow))
    c4 = (bx + ux * 0.30 * L - nx * (w - bow), by + uy * 0.30 * L - ny * (w - bow))
    edge = chain(
        bezier(base, c1, c2, tip),
        bezier(tip, c3, c4, base),
    )
    p.poly(edge)
    # midrib: a thin negative crease down the leaf center.
    rib = chain(
        bezier(base, (bx + ux * 0.33 * L, by + uy * 0.33 * L),
               (bx + ux * 0.66 * L, by + uy * 0.66 * L), tip),
        bezier(tip, (bx + ux * 0.66 * L + nx * 0.012, by + uy * 0.66 * L + ny * 0.012),
               (bx + ux * 0.33 * L + nx * 0.012, by + uy * 0.33 * L + ny * 0.012),
               (bx + nx * 0.012, by + ny * 0.012)),
    )
    p.poly(rib, fill=0)


def _stem_point(t):
    """Point on the cubic stem at parameter t in [0,1]."""
    p0, p1, p2, p3 = _STEM
    mt = 1 - t
    x = mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
    y = mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
    return (x, y)


def _draw_coffee(draw: ImageDraw.ImageDraw, n: int) -> None:
    p = Pen(draw, n)

    # ----- STEM: a tapering woody branch (thick at base, thin at tip). ------
    stem_pts = [_stem_point(i / 40) for i in range(41)]
    for i in range(len(stem_pts) - 1):
        t = i / (len(stem_pts) - 1)
        w = 0.028 * (1 - 0.6 * t)
        p.line([stem_pts[i], stem_pts[i + 1]], w)
    p.ellipse(*_STEM[0], 0.016)  # rounded base

    # ----- LEAF PAIRS at three nodes, opposite, fanning up-and-out. ---------
    nodes = [0.20, 0.50, 0.82]
    lens = [0.30, 0.32, 0.24]
    for t, ln in zip(nodes, lens):
        bx, by = _stem_point(t)
        # slightly wider leaves (0.092 vs 0.075) read as glossy coffee ovate.
        # upper leaf (leans toward the tip / up)
        _leaf(p, (bx, by), (bx - 0.02, by - ln), 0.092 * ln / 0.30, bow=+0.02)
        # lower leaf (leans down/out on the other side)
        _leaf(p, (bx, by), (bx + ln * 0.9, by + ln * 0.35), 0.092 * ln / 0.30, bow=-0.02)

    # ----- CHERRY CLUSTERS hugging the stem at the axils (the coffee signal).
    # Bigger, tightly packed rounds — the read has to say "coffee cherries", so
    # they sit as a bold berry knot at each node rather than shy dots.
    for t in (0.34, 0.60, 0.80):
        bx, by = _stem_point(t)
        cluster = [
            (bx + 0.008, by + 0.004, 0.042),
            (bx - 0.040, by + 0.020, 0.038),
            (bx + 0.028, by + 0.048, 0.036),
            (bx - 0.014, by + 0.060, 0.032),
            (bx + 0.058, by + 0.022, 0.032),
            (bx - 0.052, by + 0.058, 0.028),
        ]
        for (cx, cy, cr) in cluster:
            p.ellipse(cx, cy, cr, fill=255)
        # navel dots on the front cherries read as the coffee-fruit blossom scar.
        for idx in (0, 2, 4):
            p.ellipse(cluster[idx][0], cluster[idx][1], 0.008, fill=0)


def coffee_branch_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — arching coffee branch with cherries."""
    del extent_um
    return render_silhouette(_draw_coffee, n_grid)
