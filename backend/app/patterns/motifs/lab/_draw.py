from __future__ import annotations

"""Shared authoring helpers for the motif LAB.

Same spirit as the local helpers in ``colibri.py``: everything is authored in a
normalized 0..1 art box (Pillow convention, y grows DOWN) and scaled to the
raster grid at draw time. A cubic-Bézier sampler plus a couple of convenience
wrappers keep the individual motif files focused on *shape*, not plumbing.
"""

from typing import Sequence

from PIL import ImageDraw


def bezier(p0, p1, p2, p3, n: int = 48):
    """Sample a cubic Bézier into ``n + 1`` points."""
    out = []
    for i in range(n + 1):
        t = i / n
        mt = 1.0 - t
        x = (mt**3) * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
        y = (mt**3) * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
        out.append((x, y))
    return out


def chain(*bez_chains: Sequence):
    """Concatenate several Bézier point chains into one polygon vertex list."""
    pts: list[tuple[float, float]] = []
    for c in bez_chains:
        pts.extend(c)
    return pts


def bezier_pt(p0, p1, p2, p3, t):
    """The single point on a cubic Bézier at parameter ``t`` (0..1)."""
    mt = 1.0 - t
    x = (mt**3) * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
    y = (mt**3) * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
    return (x, y)


def bezier_tan(p0, p1, p2, p3, t):
    """The *unit* tangent (direction) of a cubic Bézier at parameter ``t``."""
    import math

    mt = 1.0 - t
    dx = 3 * mt * mt * (p1[0] - p0[0]) + 6 * mt * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0])
    dy = 3 * mt * mt * (p1[1] - p0[1]) + 6 * mt * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1])
    L = math.hypot(dx, dy) or 1e-6
    return (dx / L, dy / L)


class Pen:
    """A drawing pen bound to a Pillow ImageDraw + raster size.

    Coordinates passed to its methods are in the normalized 0..1 art box; the
    pen scales them to pixels. ``fill=255`` paints gold, ``fill=0`` punches
    negative space.
    """

    def __init__(self, draw: ImageDraw.ImageDraw, n: int):
        self.d = draw
        self.s = float(n)

    def P(self, x, y):
        return (x * self.s, y * self.s)

    def poly(self, pts, fill=255):
        self.d.polygon([self.P(x, y) for x, y in pts], fill=fill)

    def ellipse(self, cx, cy, rx, ry=None, fill=255):
        ry = rx if ry is None else ry
        self.d.ellipse(
            [self.P(cx - rx, cy - ry), self.P(cx + rx, cy + ry)], fill=fill
        )

    def line(self, pts, width_unit, fill=255):
        w = max(1, int(round(width_unit * self.s)))
        self.d.line([self.P(x, y) for x, y in pts], fill=fill, width=w, joint="curve")

    def stroke(self, pts, width_unit, fill=255):
        """A thick polyline with rounded caps (line + dots at every vertex)."""
        self.line(pts, width_unit, fill=fill)
        r = max(0.5, 0.5 * width_unit)
        for x, y in pts:
            self.ellipse(x, y, r, fill=fill)
