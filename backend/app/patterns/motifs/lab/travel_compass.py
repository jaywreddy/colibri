from __future__ import annotations

"""Travel compass — a compass rose with a dotted great-circle route arc sweeping
between two endpoints (Bogota and a US city), a heart at each end.

The read: a classic 8-point COMPASS ROSE (long N/S/E/W spikes, shorter
inter-cardinal spikes) with a ring, and an arcing DOTTED route springing off to
the upper right ending in a small heart, with a second heart near the rose for
the home city. "Travel / building a life across two countries." Authored in a
normalized 0..1 art box (Pillow y-down).
"""

import math

import numpy as np
from PIL import ImageDraw

from .._pillow import render_silhouette
from ._draw import Pen, bezier, chain

CX, CY = 0.40, 0.56    # compass center (left-of-center; route flies up-right)


def _heart(p: Pen, cx, cy, r, fill=255):
    """A small heart centered near (cx, cy) with lobe radius ~r."""
    pts = []
    for i in range(41):
        t = math.pi * (i / 40) * 2  # 0..2pi
        # classic heart parametric curve
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + (x / 16) * r, cy - (y / 16) * r))
    p.poly(pts, fill=fill)


def _draw_compass(draw: ImageDraw.ImageDraw, n: int) -> None:
    p = Pen(draw, n)

    # ----- COMPASS ROSE: 8 kite-shaped points radiating from center. --------
    # Cardinal points are long; inter-cardinal points are shorter. Each point is
    # a slim kite (two triangles) so the star reads crisp.
    R_long = 0.30
    R_short = 0.17
    hub = 0.03  # half-width of each kite at the hub

    def kite(ang_deg, length):
        a = math.radians(ang_deg)
        ux, uy = math.sin(a), -math.cos(a)     # radial (0deg = up = N)
        nx, ny = uy, -ux
        tip = (CX + ux * length, CY + uy * length)
        return [
            (CX + nx * hub, CY + ny * hub),
            tip,
            (CX - nx * hub, CY - ny * hub),
        ]

    for k in range(8):
        ang = k * 45
        length = R_long if (k % 2 == 0) else R_short
        p.poly(kite(ang, length))
    # center hub disk
    p.ellipse(CX, CY, 0.045, fill=255)

    # ----- OUTER RING: a thin gold circle around the rose. ------------------
    ring_r = 0.335
    rw = 0.012
    for pts in ([], ):
        pass
    # draw ring as an annulus: outer circle gold, inner punched later
    p.ellipse(CX, CY, ring_r, fill=255)

    # ----- ROUTE ARC: a dotted great-circle springing up-and-right. ---------
    # Start just outside the ring (home city near the rose), arc to the far
    # endpoint in the upper-right. Dots grow subtly toward the destination.
    # first dot sits clearly OUTSIDE the ring so the home heart is free of it.
    start = (CX + 0.40, CY - 0.08)
    c1 = (CX + 0.48, CY - 0.30)
    c2 = (CX + 0.56, CY - 0.36)
    end = (0.85, 0.19)
    arc = bezier(start, c1, c2, end, n=12)
    for i, (x, y) in enumerate(arc):
        r = 0.008 + 0.004 * (i / len(arc))
        p.ellipse(x, y, r, fill=255)

    # ----- ENDPOINT HEARTS: home (just outside the rose ring, SW of the arc
    # start) + destination (arc end, upper right). Both clearly free-standing.
    _heart(p, CX + 0.355, CY + 0.02, 0.034)   # home city heart
    _heart(p, end[0] + 0.01, end[1] - 0.015, 0.040)   # destination heart

    # =======================================================================
    # NEGATIVE SPACE
    # =======================================================================
    # Punch the ring interior so only a thin annulus of gold remains (the rose
    # points overpaint through it, tying rose + ring together).
    p.ellipse(CX, CY, ring_r - rw, fill=0)
    # Re-lay the rose + hub OVER the punched ring so they stay solid.
    for k in range(8):
        ang = k * 45
        length = R_long if (k % 2 == 0) else R_short
        p.poly(kite(ang, length), fill=255)
    p.ellipse(CX, CY, 0.045, fill=255)
    # A small compass-needle diamond in the hub (N half gold already; punch a
    # thin outline slit so the needle reads).
    p.ellipse(CX, CY, 0.014, fill=0)
    # Tiny center dot back in.
    p.ellipse(CX, CY, 0.006, fill=255)


def travel_compass_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — compass rose + dotted route arc."""
    del extent_um
    return render_silhouette(_draw_compass, n_grid)
