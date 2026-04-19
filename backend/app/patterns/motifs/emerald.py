from __future__ import annotations

import math

from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .._helpers import crop


def _hexagon(cx: float, cy: float, r: float) -> Polygon:
    pts = [
        (cx + r * math.cos(math.radians(60 * k + 30)),
         cy + r * math.sin(math.radians(60 * k + 30)))
        for k in range(6)
    ]
    return Polygon(pts)


def hex_facets(
    extent_um: tuple[float, float] | float,
    period_um: float,
    duty: float = 0.7,
    rotation_deg: float = 0.0,
) -> MultiPolygon:
    """Hexagonal close-packed lattice of facets — Muzo emerald crystal faces.

    `period_um` is center-to-center spacing. `duty` controls hexagon radius
    relative to the half-period (duty=1.0 → hexagons touch).
    """
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    w, h = extent_um
    diag = math.hypot(w, h) * 1.1
    dx = period_um
    dy = period_um * math.sqrt(3) / 2
    r = (period_um / 2) * duty
    nx = int(diag / dx) + 3
    ny = int(diag / dy) + 3
    hexes: list[Polygon] = []
    for j in range(-ny, ny + 1):
        for i in range(-nx, nx + 1):
            cx = i * dx + (0.5 * dx if j % 2 else 0.0)
            cy = j * dy
            hexes.append(_hexagon(cx, cy, r))
    mp = unary_union(hexes)
    if rotation_deg:
        mp = affinity.rotate(mp, rotation_deg, origin=(0, 0))
    if isinstance(mp, Polygon):
        mp = MultiPolygon([mp])
    return crop(mp, extent_um)
