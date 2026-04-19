from __future__ import annotations

import math

from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .._helpers import crop, ensure_multipolygon


def concentric_bands(
    extent_um: tuple[float, float] | float,
    band_period_um: float,
    duty: float = 0.5,
) -> MultiPolygon:
    """Concentric black/white bands — the Sombrero Vueltiao silhouette seen from above.

    Opaque bands (annuli) of width `band_period_um * duty`, spaced `band_period_um`.
    """
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    r_max = math.hypot(*extent_um) / 2
    rings = []
    r = band_period_um
    w = band_period_um * duty
    while r - w / 2 < r_max:
        outer = Point(0, 0).buffer(r + w / 2, quad_segs=96)
        inner = Point(0, 0).buffer(max(0.0, r - w / 2), quad_segs=96)
        rings.append(outer.difference(inner))
        r += band_period_um
    mp = unary_union(rings) if rings else MultiPolygon()
    return crop(ensure_multipolygon(mp), extent_um)


def pinta_triangles(
    extent_um: tuple[float, float] | float,
    band_period_um: float,
    triangle_size_um: float,
) -> MultiPolygon:
    """Triangular 'pinta' motifs — the woven tooth pattern on the Sombrero Vueltiao brim.

    A ring of upward/downward triangles arranged along each concentric band.
    """
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    r_max = math.hypot(*extent_um) / 2
    tris: list[Polygon] = []
    r = band_period_um
    s = triangle_size_um
    while r < r_max:
        circumference = 2 * math.pi * r
        n = max(8, int(circumference / (s * 1.4)))
        for k in range(n):
            theta = 2 * math.pi * k / n
            cx = r * math.cos(theta)
            cy = r * math.sin(theta)
            pointing_out = (k % 2 == 0)
            if pointing_out:
                p0 = (cx + s * math.cos(theta), cy + s * math.sin(theta))
                p1 = (cx + 0.4 * s * math.cos(theta + 2.4),
                      cy + 0.4 * s * math.sin(theta + 2.4))
                p2 = (cx + 0.4 * s * math.cos(theta - 2.4),
                      cy + 0.4 * s * math.sin(theta - 2.4))
            else:
                p0 = (cx - s * math.cos(theta), cy - s * math.sin(theta))
                p1 = (cx + 0.4 * s * math.cos(theta + 2.4),
                      cy + 0.4 * s * math.sin(theta + 2.4))
                p2 = (cx + 0.4 * s * math.cos(theta - 2.4),
                      cy + 0.4 * s * math.sin(theta - 2.4))
            tris.append(Polygon([p0, p1, p2]))
        r += band_period_um
    mp = unary_union(tris) if tris else MultiPolygon()
    return crop(ensure_multipolygon(mp), extent_um)
