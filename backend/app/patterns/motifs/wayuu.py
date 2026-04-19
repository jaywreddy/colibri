from __future__ import annotations

import math

from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .._helpers import crop


def kanasu_diamonds(
    extent_um: tuple[float, float] | float,
    period_um: float,
    duty: float = 0.5,
    rotation_deg: float = 0.0,
) -> MultiPolygon:
    """Wayuu kanasü weave: diamond (rhombus) lattice, evocative of mochila diamonds.

    The lattice is a square grid of 45°-rotated squares. `duty` controls the
    fraction of each cell that is opaque gold; `period_um` is the center-to-center
    spacing on one diagonal.
    """
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    w, h = extent_um
    diag = math.hypot(w, h) * 1.1
    half = period_um * duty / math.sqrt(2)
    n = int(diag / period_um) + 3
    diamonds: list[Polygon] = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            cx = (i + 0.5 * (j % 2)) * period_um
            cy = j * period_um * 0.75
            diamonds.append(
                Polygon([
                    (cx, cy - half),
                    (cx + half, cy),
                    (cx, cy + half),
                    (cx - half, cy),
                ])
            )
    mp = unary_union(diamonds)
    if rotation_deg:
        mp = affinity.rotate(mp, rotation_deg, origin=(0, 0))
    if isinstance(mp, Polygon):
        mp = MultiPolygon([mp])
    return crop(mp, extent_um)
