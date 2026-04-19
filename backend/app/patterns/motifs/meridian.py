from __future__ import annotations

import math

from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from .._helpers import crop, ensure_multipolygon


def meridian_grid(
    extent_um: tuple[float, float] | float,
    n_meridians: int = 12,
    n_parallels: int = 6,
    line_width_um: float = 4.0,
) -> MultiPolygon:
    """Meridian/parallel grid — globe seen from the pole.

    Straight 'meridians' radiate from center; 'parallels' are concentric rings.
    Stylized for a flat plate: lines are drawn on the plate, not the sphere.
    """
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    w, h = extent_um
    r_max = min(w, h) / 2 * 0.95
    elements: list[Polygon] = []

    # Meridians: thin rectangles rotated around origin
    for k in range(n_meridians):
        theta = math.pi * k / n_meridians
        bar = box(-line_width_um / 2, -r_max, line_width_um / 2, r_max)
        from shapely import affinity
        elements.append(affinity.rotate(bar, math.degrees(theta), origin=(0, 0)))

    # Parallels: concentric rings
    for k in range(1, n_parallels + 1):
        r = r_max * k / n_parallels
        outer = Point(0, 0).buffer(r + line_width_um / 2, quad_segs=96)
        inner = Point(0, 0).buffer(max(0.0, r - line_width_um / 2), quad_segs=96)
        elements.append(outer.difference(inner))

    mp = unary_union(elements)
    return crop(ensure_multipolygon(mp), extent_um)
