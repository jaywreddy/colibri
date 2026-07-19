from __future__ import annotations

"""Cattleya trianae — Colombia's national flower ("Flor de Mayo"), as a big
CENTERPIECE bloom, face-on.

Botanical read for a Cattleya: three narrow SEPALS (one top, two lower-side)
behind two broad, wavy PETALS (upper left/right), and in front a large frilly
trumpet LIP (labellum) that flares open at the bottom with a ruffled margin and
a paler throat. Radially arranged about the flower center. Authored in a
normalized 0..1 art box (Pillow y-down); the bloom fills most of the frame the
way a showpiece centerpiece should.
"""

import math

import numpy as np
from PIL import ImageDraw

from .._pillow import render_silhouette
from ._draw import Pen, bezier, chain

CX, CY = 0.5, 0.46   # flower center (slightly high; lip hangs below)


def _petal(p: Pen, ang_deg, length, width, curl=0.0, wavy=0.0, fill=255):
    """A tapered petal/sepal radiating from center at `ang_deg` (0=up, CW).

    `wavy` adds a gentle S to the two side edges (ruffled Cattleya petals).
    Built as two mirrored Bézier edges tip-to-base.
    """
    a = math.radians(ang_deg)
    ux, uy = math.sin(a), -math.cos(a)      # radial (0deg = up)
    nx, ny = uy, -ux                        # perpendicular
    tip = (CX + ux * length, CY + uy * length)
    base = (CX + ux * 0.06, CY + uy * 0.06)

    def at(fr, off):
        bx = CX + ux * fr * length
        by = CY + uy * fr * length
        return (bx + nx * off, by + ny * off)

    edge = chain(
        bezier(base, at(0.30, width + wavy), at(0.72, 0.92 * width - wavy + curl), tip),
        bezier(tip, at(0.72, -(0.92 * width) - wavy + curl), at(0.30, -(width) + wavy), base),
    )
    p.poly(edge, fill=fill)


def _draw_orchid(draw: ImageDraw.ImageDraw, n: int) -> None:
    p = Pen(draw, n)

    # ----- SEPALS (behind): narrow, one up, two lower-diagonal. -------------
    _petal(p, 0, 0.40, 0.055)          # dorsal sepal (top)
    _petal(p, 143, 0.40, 0.048)        # lower-right lateral sepal (steeper down)
    _petal(p, -143, 0.40, 0.048)       # lower-left lateral sepal (steeper down)

    # ----- PETALS (broad, wavy, upper-left & upper-right). ------------------
    _petal(p, 62, 0.36, 0.115, curl=0.015, wavy=0.012)   # right petal
    _petal(p, -62, 0.36, 0.115, curl=-0.015, wavy=0.012) # left petal

    # ----- LIP / labellum: a big frilled trumpet flaring down. --------------
    # Column tube at the throat, opening to a broad ruffled skirt. Built as a
    # scalloped fan below center; the ruffle is a series of arcs along the rim.
    lip_top = (CX, CY + 0.02)
    # tube sides
    tube = chain(
        bezier(lip_top, (CX - 0.06, CY + 0.10), (CX - 0.10, CY + 0.20), (CX - 0.13, CY + 0.30)),
        # ruffled bottom rim: sweep left->right in scallops
        bezier((CX - 0.13, CY + 0.30), (CX - 0.11, CY + 0.40), (CX - 0.05, CY + 0.43), (CX - 0.02, CY + 0.40)),
        bezier((CX - 0.02, CY + 0.40), (CX, CY + 0.45), (CX + 0.02, CY + 0.45), (CX + 0.04, CY + 0.40)),
        bezier((CX + 0.04, CY + 0.40), (CX + 0.07, CY + 0.44), (CX + 0.12, CY + 0.41), (CX + 0.13, CY + 0.30)),
        bezier((CX + 0.13, CY + 0.30), (CX + 0.10, CY + 0.20), (CX + 0.06, CY + 0.10), lip_top),
    )
    p.poly(tube)
    # widen the skirt with two side lobes so the lip reads BROAD, not a stripe.
    p.poly(chain(
        bezier((CX - 0.05, CY + 0.15), (CX - 0.17, CY + 0.20), (CX - 0.20, CY + 0.30), (CX - 0.13, CY + 0.33)),
        bezier((CX - 0.13, CY + 0.33), (CX - 0.12, CY + 0.26), (CX - 0.08, CY + 0.20), (CX - 0.05, CY + 0.15)),
    ))
    p.poly(chain(
        bezier((CX + 0.05, CY + 0.15), (CX + 0.17, CY + 0.20), (CX + 0.20, CY + 0.30), (CX + 0.13, CY + 0.33)),
        bezier((CX + 0.13, CY + 0.33), (CX + 0.12, CY + 0.26), (CX + 0.08, CY + 0.20), (CX + 0.05, CY + 0.15)),
    ))

    # ----- CENTER column knot so all parts fuse at the middle. --------------
    p.ellipse(CX, CY, 0.07, 0.075, fill=255)

    # =======================================================================
    # NEGATIVE SPACE
    # =======================================================================
    # Throat shadow inside the lip tube — an open funnel: wide at the mouth
    # (top, under the column) narrowing as it recedes, so it reads as the paler
    # OPEN throat of a Cattleya rather than a keyhole slot.
    p.poly(chain(
        bezier((CX - 0.055, CY + 0.095), (CX - 0.075, CY + 0.17), (CX - 0.055, CY + 0.24), (CX - 0.018, CY + 0.27)),
        bezier((CX - 0.018, CY + 0.27), (CX, CY + 0.285), (CX + 0.018, CY + 0.285), (CX + 0.018, CY + 0.27)),
        bezier((CX + 0.018, CY + 0.27), (CX + 0.055, CY + 0.24), (CX + 0.075, CY + 0.17), (CX + 0.055, CY + 0.095)),
        bezier((CX + 0.055, CY + 0.095), (CX + 0.03, CY + 0.085), (CX - 0.03, CY + 0.085), (CX - 0.055, CY + 0.095)),
    ), fill=0)
    # A small round anther cap dot at the very center of the column.
    p.ellipse(CX, CY + 0.005, 0.018, fill=0)
    # Faint midrib creases on the two broad petals so they read as petals.
    for ang in (62, -62):
        a = math.radians(ang)
        ux, uy = math.sin(a), -math.cos(a)
        p.line([(CX + ux * 0.10, CY + uy * 0.10), (CX + ux * 0.33, CY + uy * 0.33)], 0.006, fill=0)


def orchid_cattleya_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — Cattleya trianae centerpiece bloom."""
    del extent_um
    return render_silhouette(_draw_orchid, n_grid)
