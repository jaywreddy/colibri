"""Rasterize real Natural Earth geography under an orthographic projection.

The public entry point is :func:`render_globe`, which paints a binary gold-on-
dark cartographic globe: ocean is negative space (the hole), land + graticule +
limb are gold (True). Colombia can be emphasized with an extra solid fill and/or
a star marker. Geometry is loaded from vendored 110m GeoJSON (see
``data/LICENSE_NOTE.md``), simplified adaptively for the target raster size, and
projected/clipped in :mod:`.ortho`.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .ortho import (
    Orthographic,
    clamp_to_disk,
    close_on_limb,
    project_ring,
    ring_area,
    simplify_ring,
)

_DATA = Path(__file__).parent / "data"
_LAND = _DATA / "ne_110m_land.geojson"
_COUNTRIES = _DATA / "ne_110m_admin_0_countries.geojson"


# --------------------------------------------------------------------------
# GeoJSON loading -> flat lists of (N, 2) lon/lat ring arrays.
# --------------------------------------------------------------------------
def _geom_rings(geom: dict):
    """Yield exterior + hole rings of a Polygon/MultiPolygon as (lon, lat)."""
    t = geom["type"]
    coords = geom["coordinates"]
    if t == "Polygon":
        polys = [coords]
    elif t == "MultiPolygon":
        polys = coords
    else:
        return
    for poly in polys:
        for ring in poly:  # ring[0] = exterior, rest = holes (rare at 110m)
            yield np.asarray(ring, dtype=float)


@lru_cache(maxsize=1)
def load_land_rings() -> tuple:
    """All 110m land rings as a tuple of (N, 2) lon/lat arrays (cached)."""
    data = json.loads(_LAND.read_text(encoding="utf-8"))
    rings = []
    for feat in data["features"]:
        rings.extend(_geom_rings(feat["geometry"]))
    return tuple(rings)


@lru_cache(maxsize=8)
def load_country_rings(name: str) -> tuple:
    """Rings for one country, matched case-insensitively on NAME/ADMIN/NAME_LONG."""
    data = json.loads(_COUNTRIES.read_text(encoding="utf-8"))
    key = name.strip().lower()
    rings = []
    for feat in data["features"]:
        props = feat["properties"]
        names = {
            str(props.get(k, "")).strip().lower()
            for k in ("NAME", "ADMIN", "NAME_LONG", "NAME_EN", "SOVEREIGNT")
        }
        if key in names:
            rings.extend(_geom_rings(feat["geometry"]))
    return tuple(rings)


# --------------------------------------------------------------------------
# Rasterization
# --------------------------------------------------------------------------
def _fill_rings(draw, rings, proj, cx, cy, r, tol_ll, min_area_px, fill):
    """Project, simplify, clip, and fill a batch of lon/lat rings.

    ``tol_ll`` is Douglas–Peucker tolerance in *degrees* (lon/lat units),
    ``min_area_px`` drops islands whose projected area is under a few pixels so
    coarse grids stay crisp.
    """
    def W(xy):
        # projected unit xy (y up) -> pixel coords (y down)
        px = cx + xy[:, 0] * r
        py = cy - xy[:, 1] * r
        return list(zip(px.tolist(), py.tolist()))

    for ring in rings:
        if tol_ll > 0:
            ring = simplify_ring(ring, tol_ll)
        if len(ring) < 3:
            continue
        runs = project_ring(proj, ring)
        poly = close_on_limb(runs)
        if poly is None or len(poly) < 3:
            continue
        poly = clamp_to_disk(poly)
        if ring_area(poly) * r * r < min_area_px:
            continue
        draw.polygon(W(poly), fill=fill)


def _draw_graticule(draw, proj, cx, cy, r, width, fill):
    """Latitude parallels (every 15 deg) and meridians (every 30 deg)."""
    def W(xy):
        return list(zip((cx + xy[:, 0] * r).tolist(), (cy - xy[:, 1] * r).tolist()))

    def polyline(lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x, y, cosc = proj.project(lon, lat)
        vis = proj.visible(cosc)
        xy = np.column_stack([x, y])
        seg = []
        for i in range(len(xy)):
            if vis[i]:
                seg.append(xy[i])
            else:
                if len(seg) >= 2:
                    draw.line(W(clamp_to_disk(np.asarray(seg))), fill=fill,
                              width=width, joint="curve")
                seg = []
        if len(seg) >= 2:
            draw.line(W(clamp_to_disk(np.asarray(seg))), fill=fill,
                      width=width, joint="curve")

    lon_sweep = np.arange(-180, 181, 2.0)
    for lat in (-60, -45, -30, -15, 0, 15, 30, 45, 60, 75):
        polyline(lon_sweep, np.full_like(lon_sweep, lat))
    # Meridians stop short of the pole (|lat| <= 78) so they don't collapse into
    # a cluttered starburst where they all converge at the visible pole.
    lat_sweep = np.arange(-78, 78.1, 2.0)
    for lon in range(-180, 181, 30):
        polyline(np.full_like(lat_sweep, lon), lat_sweep)


def _draw_star(draw, proj, cx, cy, r, lon, lat, star_r, fill, halo=True):
    """Gold star at (lon, lat). When ``halo``, a dark keep-out disk is punched
    first so the star reads as a distinct emblem even sitting on gold land."""
    x, y, cosc = proj.project(np.array([lon]), np.array([lat]))
    if not proj.visible(cosc)[0]:
        return
    px = cx + float(x[0]) * r
    py = cy - float(y[0]) * r
    if halo:
        h = star_r * 1.30
        draw.ellipse([(px - h, py - h), (px + h, py + h)], fill=0)
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = star_r if (i % 2 == 0) else star_r * 0.42
        pts.append((px + rad * math.cos(ang), py + rad * math.sin(ang)))
    draw.polygon(pts, fill=fill)


def render_globe(
    n: int,
    lat0: float = 30.0,
    lon0: float = -50.0,
    disk_frac: float = 0.44,
    highlight_country: str | None = "Colombia",
    highlight_star: bool = True,
    star_lonlat: tuple[float, float] = (-73.5, 4.6),
    threshold: int = 127,
) -> np.ndarray:
    """Render the cartographic globe to a bool (n, n) array (True = gold).

    Adaptive detail: Douglas–Peucker tolerance and the min-island area both
    scale with 1/n, so a 256-px composed raster gets crisp, de-noised coastlines
    while a 1024-px preview keeps fine detail. Feature floor is kept >= ~2 px.
    """
    # Every caller derives ``n`` from request params that nothing validates
    # against their ParamSpec bounds, so gate the raster before allocating it.
    # Function-local import keeps app.patterns.motifs (which imports THIS
    # module via motifs.globe) out of geo's import-time graph, and keeps geo
    # shapely-free as its package docstring promises.
    from ..motifs._pillow import check_silhouette_budget

    check_silhouette_budget(n, "Cartographic globe raster")
    img = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(img)
    s = float(n)
    cx = cy = 0.5 * s
    r = disk_frac * s
    proj = Orthographic(lat0=lat0, lon0=lon0)

    # Adaptive simplification. At the composed scale (~256) 1 px ~= 0.5 deg on
    # the disk face, so a ~0.6 px DP tolerance in lon/lat de-noises stair-steps
    # without eroding capes. Detail-rich previews (n large) barely simplify.
    px_per_deg = (2.0 * r) / 180.0  # rough: disk diameter spans ~180 deg
    tol_ll = 0.6 / max(px_per_deg, 1e-6)      # ~0.6 px worth of lon/lat
    min_area_px = 2.0                          # drop sub-2px islands

    # --- ocean disk stays dark (0) ---
    grat_w = max(1, int(round(0.0045 * s)))
    _draw_graticule(draw, proj, cx, cy, r, grat_w, fill=150)

    # --- land (solid gold, overpaints graticule) ---
    _fill_rings(draw, load_land_rings(), proj, cx, cy, r,
                tol_ll, min_area_px, fill=255)

    # --- Colombia emphasis: solid re-fill (crisper edge) + star marker ---
    if highlight_country:
        col = load_country_rings(highlight_country)
        if col:
            _fill_rings(draw, col, proj, cx, cy, r, tol_ll * 0.5,
                        0.0, fill=255)
    if highlight_star:
        star_r = max(2.0, 0.028 * s)
        _draw_star(draw, proj, cx, cy, r, star_lonlat[0], star_lonlat[1],
                   star_r, fill=255)

    # --- crisp gold limb ring ---
    limb_w = max(2, int(round(0.014 * s)))
    for i in range(limb_w):
        draw.ellipse([(cx - r + i, cy - r + i), (cx + r - i, cy + r - i)],
                     outline=255)

    arr = np.asarray(img, dtype=np.uint8)
    return arr > threshold
