from __future__ import annotations

"""Andean condor — Colombia's national bird — soaring head-on, wings spread.

The read for a soaring condor (vs. an eagle) is: enormous BROAD wings held
nearly flat/level, deeply slotted PRIMARY "fingers" upturned at each tip, a
short wedge TAIL, and a small head on a distinct pale neck RUFF. Symmetric,
front-on, as if seen from below riding a thermal. Authored in a normalized 0..1
art box (Pillow y-down); mirror-symmetric about x=0.5.
"""

import numpy as np
from PIL import ImageDraw

from .._pillow import render_silhouette
from ._draw import Pen, bezier, chain

MID = 0.5


def _mirror(pts):
    """Reflect a vertex list about x=MID (right half -> left half)."""
    return [(2 * MID - x, y) for (x, y) in pts]


def _draw_condor(draw: ImageDraw.ImageDraw, n: int) -> None:
    p = Pen(draw, n)

    # ----- BODY: a stout, short spindle down the centerline (a condor is mostly
    # wing — the body must NOT read long). Blends into a short wedge tail.
    body = chain(
        bezier((MID - 0.058, 0.36), (MID - 0.062, 0.42), (MID - 0.06, 0.48), (MID - 0.055, 0.52)),
        bezier((MID - 0.055, 0.52), (MID - 0.058, 0.58), (MID - 0.05, 0.64), (MID, 0.665)),   # short wedge tail L
        bezier((MID, 0.665), (MID + 0.05, 0.64), (MID + 0.058, 0.58), (MID + 0.055, 0.52)),   # short wedge tail R
        bezier((MID + 0.055, 0.52), (MID + 0.06, 0.48), (MID + 0.062, 0.42), (MID + 0.058, 0.36)),
        bezier((MID + 0.058, 0.36), (MID + 0.02, 0.33), (MID - 0.02, 0.33), (MID - 0.058, 0.36)),
    )
    p.poly(body)

    # ----- RIGHT WING: broad, held level, trailing edge swept back. Span pulled
    # IN to ~0.38 so the primary fingers have room to fan without leaving frame.
    wing_r = chain(
        bezier((MID + 0.04, 0.375), (MID + 0.15, 0.36), (MID + 0.27, 0.365), (MID + 0.37, 0.39)),  # leading edge out
        bezier((MID + 0.37, 0.39), (MID + 0.395, 0.405), (MID + 0.395, 0.43), (MID + 0.37, 0.45)),  # squared tip
        bezier((MID + 0.37, 0.45), (MID + 0.25, 0.50), (MID + 0.14, 0.51), (MID + 0.05, 0.50)),     # trailing edge back
        bezier((MID + 0.05, 0.50), (MID + 0.045, 0.45), (MID + 0.042, 0.41), (MID + 0.04, 0.375)),  # to shoulder
    )
    p.poly(wing_r)
    p.poly(_mirror(wing_r))

    # ----- PRIMARY FINGERS: 5 broad upturned feathers off each squared tip.
    # Fatter (halfw 0.024) and shorter so they read as distinct fingers, fanned
    # so the outer finger points most up-and-out — the classic condor hand.
    def fingers(sign):
        anchors = [
            (MID + sign * 0.375, 0.395),
            (MID + sign * 0.372, 0.415),
            (MID + sign * 0.362, 0.432),
            (MID + sign * 0.345, 0.448),
            (MID + sign * 0.322, 0.462),
        ]
        angles = [22, 38, 54, 70, 86]     # degrees above +x, fanning up
        length = [0.115, 0.125, 0.12, 0.105, 0.085]
        halfw = 0.024
        for (ax, ay), ang, ln in zip(anchors, angles, length):
            import math
            a = math.radians(ang)
            ux, uy = sign * math.cos(a), -math.sin(a)   # up-and-out
            nx, ny = -uy, ux                             # perpendicular (width)
            tipx, tipy = ax + ux * ln, ay + uy * ln
            blade = [
                (ax + nx * halfw, ay + ny * halfw),
                (ax - nx * halfw, ay - ny * halfw),
                (tipx - nx * halfw * 0.4, tipy - ny * halfw * 0.4),
                (tipx + nx * halfw * 0.4, tipy + ny * halfw * 0.4),
            ]
            p.poly(blade)
    fingers(+1)
    fingers(-1)

    # ----- HEAD + NECK RUFF: small head on a distinct collar. ---------------
    p.ellipse(MID, 0.32, 0.075, 0.045, fill=255)   # the pale neck ruff (wide collar)
    p.ellipse(MID, 0.275, 0.032, 0.036, fill=255)  # small head above the ruff
    # caruncle/comb hint: a tiny bump on the crown (male condor).
    p.poly(chain(
        bezier((MID - 0.012, 0.245), (MID - 0.01, 0.225), (MID + 0.01, 0.225), (MID + 0.012, 0.245)),
        bezier((MID + 0.012, 0.245), (MID + 0.006, 0.25), (MID - 0.006, 0.25), (MID - 0.012, 0.245)),
    ))

    # =======================================================================
    # NEGATIVE SPACE
    # =======================================================================
    # Primary slots: thin gaps radiating between the finger anchors so the hand
    # reads as separated feathers, not a solid paddle. Cut along the mid-angle
    # between each adjacent pair of fingers.
    def slots(sign):
        import math
        anchors = [
            (MID + sign * 0.375, 0.395),
            (MID + sign * 0.372, 0.415),
            (MID + sign * 0.362, 0.432),
            (MID + sign * 0.345, 0.448),
            (MID + sign * 0.322, 0.462),
        ]
        angles = [22, 38, 54, 70, 86]
        for i in range(len(angles) - 1):
            mang = math.radians((angles[i] + angles[i + 1]) / 2)
            ax = (anchors[i][0] + anchors[i + 1][0]) / 2
            ay = (anchors[i][1] + anchors[i + 1][1]) / 2
            ux, uy = sign * math.cos(mang), -math.sin(mang)
            p.line([(ax - ux * 0.01, ay - uy * 0.01), (ax + ux * 0.10, ay + uy * 0.10)], 0.007, fill=0)
    slots(+1)
    slots(-1)
    # Ruff separation: a thin dark line under the head so the collar reads.
    p.line([(MID - 0.06, 0.305), (MID + 0.06, 0.305)], 0.006, fill=0)


def condor_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — Andean condor soaring, wings spread."""
    del extent_um
    return render_silhouette(_draw_condor, n_grid)
