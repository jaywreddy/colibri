from __future__ import annotations

import math

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .._helpers import crop, ensure_multipolygon, ring_grating


def goldwork_spiral(
    extent_um: tuple[float, float] | float,
    arm_width_um: float,
    pitch_um: float,
    n_turns: float = 4.0,
) -> MultiPolygon:
    """Archimedean-spiral band — Tairona goldwork concentric coil motif."""
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    r_max = math.hypot(*extent_um) / 2
    samples = max(400, int(n_turns * 200))
    pts_outer = []
    pts_inner = []
    half = arm_width_um / 2
    for s in range(samples + 1):
        t = s / samples
        theta = t * n_turns * 2 * math.pi
        r = pitch_um * theta / (2 * math.pi)
        if r > r_max:
            break
        nx = math.cos(theta + math.pi / 2)
        ny = math.sin(theta + math.pi / 2)
        cx = r * math.cos(theta)
        cy = r * math.sin(theta)
        pts_outer.append((cx + half * nx, cy + half * ny))
        pts_inner.append((cx - half * nx, cy - half * ny))
    if len(pts_outer) < 3:
        return MultiPolygon()
    poly = Polygon(pts_outer + list(reversed(pts_inner)))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return crop(ensure_multipolygon(poly), extent_um)


def concentric_goldwork(
    extent_um: tuple[float, float] | float,
    period_um: float,
    duty: float = 0.5,
) -> MultiPolygon:
    """Concentric rings for use as a circular grating / Talbot carrier."""
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    return ring_grating(period_um, duty, extent_um)
