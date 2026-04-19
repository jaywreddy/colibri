from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from .base import ensure_multipolygon


def crop_box(extent_um: tuple[float, float]) -> Polygon:
    w, h = extent_um
    return box(-w / 2, -h / 2, w / 2, h / 2)


def crop(polys, extent_um: tuple[float, float]) -> MultiPolygon:
    return ensure_multipolygon(polys.intersection(crop_box(extent_um)))


def linear_grating(
    period_um: float,
    duty: float,
    extent_um: tuple[float, float],
    rotation_deg: float = 0.0,
) -> MultiPolygon:
    """Return alternating stripes of opaque gold of the given period."""
    w, h = extent_um
    diag = math.hypot(w, h) * 1.05
    stripe_w = period_um * duty
    n = int(diag / period_um) + 2
    stripes = []
    for i in range(-n, n + 1):
        cx = i * period_um
        stripes.append(box(cx - stripe_w / 2, -diag / 2, cx + stripe_w / 2, diag / 2))
    mp = MultiPolygon(stripes)
    if rotation_deg:
        mp = affinity.rotate(mp, rotation_deg, origin=(0, 0), use_radians=False)
    return crop(mp, extent_um)


def dot_array(
    period_um: float,
    diameter_um: float,
    extent_um: tuple[float, float],
    rotation_deg: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> MultiPolygon:
    """Square lattice of circular dots."""
    w, h = extent_um
    diag = math.hypot(w, h) * 1.05
    nx = int(diag / period_um) + 2
    r = diameter_um / 2
    disks = []
    for i in range(-nx, nx + 1):
        for j in range(-nx, nx + 1):
            cx = i * period_um + offset[0]
            cy = j * period_um + offset[1]
            disks.append(Point(cx, cy).buffer(r, quad_segs=12))
    mp = unary_union(disks)
    mp = ensure_multipolygon(mp)
    if rotation_deg:
        mp = affinity.rotate(mp, rotation_deg, origin=(0, 0))
    return crop(mp, extent_um)


def annulus(r_inner: float, r_outer: float, quad_segs: int = 96) -> Polygon:
    outer = Point(0, 0).buffer(r_outer, quad_segs=quad_segs)
    inner = Point(0, 0).buffer(r_inner, quad_segs=quad_segs)
    return outer.difference(inner)


def zone_plate(
    focal_length_um: float,
    wavelength_um: float,
    extent_um: tuple[float, float],
    min_feature_um: float = 2.0,
) -> MultiPolygon:
    """Binary amplitude Fresnel zone plate — odd zones opaque, even transparent."""
    w, h = extent_um
    r_max = min(w, h) / 2
    rings = []
    n = 1
    while True:
        r_n = math.sqrt(n * focal_length_um * wavelength_um)
        if r_n > r_max:
            break
        r_prev = math.sqrt((n - 1) * focal_length_um * wavelength_um) if n > 1 else 0.0
        if n % 2 == 1:  # opaque ring
            if r_n - r_prev < min_feature_um:
                break
            rings.append(annulus(r_prev, r_n))
        n += 1
    mp = unary_union(rings) if rings else MultiPolygon()
    return crop(ensure_multipolygon(mp), extent_um)


def raster_to_polygons(
    grid: np.ndarray,
    cell_um: float,
    extent_um: tuple[float, float],
) -> MultiPolygon:
    """Convert a 2D binary numpy array (1=gold) into a MultiPolygon of cells.

    `cell_um` is the size of each pixel in μm. This is O(n_cells) but we merge
    adjacent cells via unary_union to keep output compact.
    """
    h, w = grid.shape
    hx = w * cell_um / 2.0
    hy = h * cell_um / 2.0
    cells: list[Polygon] = []
    ys, xs = np.where(grid > 0)
    for px, py in zip(xs, ys):
        x0 = px * cell_um - hx
        y0 = hy - (py + 1) * cell_um
        cells.append(box(x0, y0, x0 + cell_um, y0 + cell_um))
    if not cells:
        return MultiPolygon()
    merged = unary_union(cells)
    return crop(ensure_multipolygon(merged), extent_um)


def empty_layer() -> MultiPolygon:
    return MultiPolygon()


def wavelength_from_color(color: str) -> float:
    return {"red": 0.65, "green": 0.532, "blue": 0.405}.get(color, 0.55)


def ring_grating(
    radial_period_um: float,
    duty: float,
    extent_um: tuple[float, float],
    center: tuple[float, float] = (0.0, 0.0),
) -> MultiPolygon:
    """Concentric rings of opaque gold at the given radial period."""
    r_max = math.hypot(*extent_um) / 2 + math.hypot(*center)
    cx, cy = center
    rings = []
    r = radial_period_um
    w = radial_period_um * duty
    while r - w / 2 < r_max:
        outer = Point(cx, cy).buffer(r + w / 2, quad_segs=128)
        inner = Point(cx, cy).buffer(max(0.0, r - w / 2), quad_segs=128)
        rings.append(outer.difference(inner))
        r += radial_period_um
    mp = unary_union(rings) if rings else MultiPolygon()
    return crop(ensure_multipolygon(mp), extent_um)


def chevron_stripes(
    extent_um: tuple[float, float],
    period_um: float,
    amp_um: float,
) -> MultiPolygon:
    """V-shaped (chevron) stripes filling the extent, period_um tall, with width amp_um."""
    from shapely.geometry import Polygon as _Polygon
    w, h = extent_um
    polys = []
    n = int(h / period_um) + 2
    half = w / 2
    for i in range(-n, n + 1):
        cy = i * period_um
        left = _Polygon([
            (-half, cy - amp_um / 2),
            (0.0, cy + amp_um / 2),
            (0.0, cy + amp_um / 2 + period_um * 0.25),
            (-half, cy - amp_um / 2 + period_um * 0.25),
        ])
        right = _Polygon([
            (0.0, cy + amp_um / 2),
            (half, cy - amp_um / 2),
            (half, cy - amp_um / 2 + period_um * 0.25),
            (0.0, cy + amp_um / 2 + period_um * 0.25),
        ])
        polys.extend([left, right])
    merged = unary_union(polys)
    return crop(ensure_multipolygon(merged), extent_um)
