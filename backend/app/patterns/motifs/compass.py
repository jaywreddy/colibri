from __future__ import annotations

import math

from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .._helpers import crop, ensure_multipolygon


def compass_rose(
    extent_um: tuple[float, float] | float,
    radius_um: float,
    n_points: int = 8,
    sharp_ratio: float = 0.25,
) -> MultiPolygon:
    """Mariner's compass rose — 8-pointed star of alternating long/short rays.

    Each 'ray' is a thin isoceles triangle from center to radius_um; between
    each long point is a short point of length `radius_um * sharp_ratio`.
    """
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    n = n_points * 2
    r_long = radius_um
    r_short = radius_um * sharp_ratio
    pts = []
    for k in range(n):
        theta = 2 * math.pi * k / n - math.pi / 2
        r = r_long if k % 2 == 0 else r_short
        pts.append((r * math.cos(theta), r * math.sin(theta)))
    outer = Polygon(pts)
    # Add a central circle for the compass hub
    from shapely.geometry import Point as _Point
    hub = _Point(0, 0).buffer(radius_um * 0.1, quad_segs=48)
    mp = unary_union([outer, hub])
    return crop(ensure_multipolygon(mp), extent_um)


def spiral_arms(
    extent_um: tuple[float, float] | float,
    n_arms: int,
    arm_width_um: float,
    twist_turns: float = 0.5,
) -> MultiPolygon:
    """Logarithmic-spiral arms radiating from center — used as a rotation-coupled grating."""
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    r_max = math.hypot(*extent_um) / 2
    arms: list[Polygon] = []
    for k in range(n_arms):
        theta0 = 2 * math.pi * k / n_arms
        samples = 80
        pts_outer = []
        pts_inner = []
        for s in range(samples + 1):
            t = s / samples
            r = t * r_max
            theta = theta0 + twist_turns * 2 * math.pi * t
            nx = -math.sin(theta)
            ny = math.cos(theta)
            cx = r * math.cos(theta)
            cy = r * math.sin(theta)
            half = arm_width_um / 2
            pts_outer.append((cx + half * nx, cy + half * ny))
            pts_inner.append((cx - half * nx, cy - half * ny))
        poly = Polygon(pts_outer + list(reversed(pts_inner)))
        if not poly.is_valid:
            poly = poly.buffer(0)
        arms.append(poly)
    mp = unary_union(arms)
    return crop(ensure_multipolygon(mp), extent_um)
