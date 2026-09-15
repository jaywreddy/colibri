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


@lru_cache(maxsize=1)
def _countries_doc() -> tuple:
    """The 110m country features, parsed ONCE.

    ``load_country_rings`` used to re-parse the multi-MB GeoJSON per name behind
    an 8-entry cache; the region map below asks for a whole continent's worth of
    names in one pass, which would have thrashed that cache into dozens of JSON
    parses. Every name lookup now walks this single parsed tuple.
    """
    data = json.loads(_COUNTRIES.read_text(encoding="utf-8"))
    return tuple(data["features"])


_NAME_KEYS = ("NAME", "ADMIN", "NAME_LONG", "NAME_EN", "SOVEREIGNT")


def _feat_names(feat: dict) -> set:
    props = feat["properties"]
    return {str(props.get(k, "")).strip().lower() for k in _NAME_KEYS}


@lru_cache(maxsize=16)
def load_country_rings(name: str) -> tuple:
    """Rings for one country, matched case-insensitively on NAME/ADMIN/NAME_LONG."""
    key = name.strip().lower()
    rings = []
    for feat in _countries_doc():
        if key in _feat_names(feat):
            rings.extend(_geom_rings(feat["geometry"]))
    return tuple(rings)


@lru_cache(maxsize=8)
def load_country_group_rings(names: tuple) -> tuple:
    """Rings for a GROUP of countries as one flat ring list (one file pass)."""
    keys = {str(n).strip().lower() for n in names}
    rings = []
    for feat in _countries_doc():
        if keys & _feat_names(feat):
            rings.extend(_geom_rings(feat["geometry"]))
    return tuple(rings)


@lru_cache(maxsize=4)
def load_continent_rings(
    continent: str, exclude: tuple = (), bbox: tuple | None = None
) -> tuple:
    """Rings of every country whose ``CONTINENT`` is ``continent``.

    ``exclude`` drops countries by name — the region map uses it to keep Russia
    out of "Europe", which at 110m is one polygon spanning to the Pacific and
    would flood the whole of northern Asia with Europe's colour.

    ``bbox`` is ``(lon_min, lat_min, lon_max, lat_max)`` and drops individual
    RINGS whose centre falls outside it. Metropolitan-vs-overseas is the reason:
    at 110m the France feature carries French Guiana in the same MultiPolygon,
    so without a box "Europe" paints a patch on the shoulder of South America,
    right next to Colombia, where it reads as a mistake rather than as a fact
    about French departments.
    """
    key = continent.strip().lower()
    drop = {str(n).strip().lower() for n in exclude}
    rings = []
    for feat in _countries_doc():
        if str(feat["properties"].get("CONTINENT", "")).strip().lower() != key:
            continue
        if _feat_names(feat) & drop:
            continue
        for ring in _geom_rings(feat["geometry"]):
            if bbox is not None:
                lon, lat = float(ring[:, 0].mean()), float(ring[:, 1].mean())
                if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                    continue
            rings.append(ring)
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


_GRATICULE_LATS = (-60, -45, -30, -15, 0, 15, 30, 45, 60, 75)


def _draw_graticule(draw, proj, cx, cy, r, width, fill, lats=_GRATICULE_LATS,
                    lon_step: int = 30):
    """Latitude parallels (``lats``, degrees) and meridians (every ``lon_step``).

    The region map dials both COARSER: there every line is solid gold at a
    ≥3-cell width, and the default 15 deg parallel spacing turns the ocean into
    a net that competes with the continents instead of framing them.
    """
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
    for lat in lats:
        polyline(lon_sweep, np.full_like(lon_sweep, float(lat)))
    # Meridians stop short of the pole (|lat| <= 78) so they don't collapse into
    # a cluttered starburst where they all converge at the visible pole.
    lat_sweep = np.arange(-78, 78.1, 2.0)
    for lon in range(-180, 181, int(lon_step)):
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


# --------------------------------------------------------------------------
# SINGLE-LAYER DIFFRACTION region map (see app/region_art.py).
#
# The same projection, the same rings, the same limb/graticule/star furniture as
# render_globe — but the output is a LABEL raster instead of a bool one: each
# geographic class gets its own id, and the motif layer above turns those ids
# into grating PERIODS (colour) or solid gold. Sharing this module means the
# preview globe and the written globe are the same map, drawn once.
# --------------------------------------------------------------------------
GLOBE_REGION_OCEAN = 0
"""Bare glass — no gold. The dark sea the continents sit in."""
GLOBE_REGION_LAND = 1
"""Every land mass that is not one of the three named below."""
GLOBE_REGION_USA = 2
GLOBE_REGION_EUROPE = 3
GLOBE_REGION_COLOMBIA = 4
GLOBE_REGION_FURNITURE = 5
"""Graticule + limb ring — the cartographic furniture.

It is a GRATING region, not solid gold, and that is a hard constraint rather
than a taste call. The fine writer heals the merged layer with
``drc_clean_region``, which returns EXTERIOR rings only (holes dropped), so any
closed ring of SOLID gold comes back as the filled disk it bounds: a solid limb
ring turns the whole globe into a gold coin, and a solid graticule turns every
cell its meridians and parallels enclose into a gold pane. Grating regions
cannot do that — they are disjoint vertical lines and enclose nothing — so the
furniture is written at its own period instead. It can afford one: at the
shipping art box these lines are ~66 µm wide, a dozen-plus periods, far above
region_art.py's "narrower than ~2 periods has no spectrum" floor. Flip it back
to solid (period 0) the day the heal keeps holes.
"""
GLOBE_REGION_STAR = 6
"""The two emphasis stars: SOLID gold, plain specular metal against the coloured
region each one sits on. A star is simply connected — it encloses nothing — so
it is safe where the ring and the graticule are not."""

USA_NAME = "United States of America"
COLOMBIA_NAME = "Colombia"
EUROPE_EXCLUDE = ("Russia",)
"""Excluded from the "Europe" region: at 110m Russia is ONE polygon reaching the
Pacific, so including it would colour half of Asia as Europe."""
EUROPE_BBOX = (-32.0, 30.0, 45.0, 75.0)
"""Ring box for the "Europe" region — Azores to the Urals, Crete to Tromsø.
Keeps overseas departments (French Guiana, which rides in France's own 110m
MultiPolygon) from colouring a patch of South America as Europe."""


def globe_region_labels(
    n: int,
    lat0: float = 30.0,
    lon0: float = -50.0,
    disk_frac: float = 0.46,
    star_lonlats: tuple = ((-73.5, 4.6), (-119.5, 36.5)),
    graticule: bool = True,
) -> np.ndarray:
    """int32 (n, n) label raster of the orthographic globe (y DOWN).

    Values are the ``GLOBE_REGION_*`` ids above. Paint order mirrors
    :func:`render_globe`: graticule first (so land overpaints it and only the
    ocean is threaded by meridians), then all land, then the three named
    regions, then the limb ring, then the stars with their bare-glass halo.

    Feature widths are floored at **4 px** rather than render_globe's 2: the
    emitter insets every region by one cell to open the inter-region gutter, so
    a 2-px graticule would be left 0 px wide and vanish from the written plate,
    and the furniture carries a grating that wants a few periods of width left
    after the inset.
    """
    from ..motifs._pillow import check_silhouette_budget

    check_silhouette_budget(n, "Cartographic globe region map")
    n = int(n)
    img = Image.new("L", (n, n), GLOBE_REGION_OCEAN)
    draw = ImageDraw.Draw(img)
    s = float(n)
    cx = cy = 0.5 * s
    r = disk_frac * s
    proj = Orthographic(lat0=lat0, lon0=lon0)

    # Adaptive simplification, as in render_globe but a touch stronger: a region
    # boundary is a COLOUR boundary (two gratings and a glass gutter), so a
    # single-pixel coastal stair-step costs more here than it does in a
    # monochrome silhouette. min_area drops islands under ~3x3 cells, which
    # would be eaten by the gutter and read as speckle.
    px_per_deg = (2.0 * r) / 180.0
    tol_ll = 0.8 / max(px_per_deg, 1e-6)
    min_area_px = 12.0

    if graticule:
        grat_w = max(4, int(round(0.0045 * s)))
        _draw_graticule(draw, proj, cx, cy, r, grat_w, fill=GLOBE_REGION_FURNITURE,
                        lats=(-60, -30, 0, 30, 60), lon_step=30)

    _fill_rings(draw, load_land_rings(), proj, cx, cy, r,
                tol_ll, min_area_px, fill=GLOBE_REGION_LAND)

    # The three emphasis regions, each re-filled at half the land tolerance so
    # their (colour-bearing) borders stay crisper than the generic coastline.
    for rings, label in (
        (load_country_group_rings((USA_NAME,)), GLOBE_REGION_USA),
        (load_continent_rings("Europe", EUROPE_EXCLUDE, EUROPE_BBOX),
         GLOBE_REGION_EUROPE),
        (load_country_group_rings((COLOMBIA_NAME,)), GLOBE_REGION_COLOMBIA),
    ):
        if rings:
            _fill_rings(draw, rings, proj, cx, cy, r, tol_ll * 0.5,
                        min_area_px, fill=label)

    # ONE thick stroke, not render_globe's stack of concentric 1-px ellipses:
    # on a bool silhouette the hairline gaps that stack leaves where the curve
    # runs shallow simply fill in at threshold, but here every gap is a strip of
    # bare glass inside the ring, and the 1-px ribs either side of it are thinner
    # than the emitter's gutter — the limb would come out of the writer as a
    # band of speckle instead of a ring.
    limb_w = max(4, int(round(0.014 * s)))
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                 outline=GLOBE_REGION_FURNITURE, width=limb_w)

    # Stars: SOLID gold and NO dark halo. render_globe needs the halo because
    # there a gold star on gold land is invisible; here the star is plain metal
    # sitting on a COLOURED grating (its own region), so the emitter's one-cell
    # glass gutter already draws the outline for free — and a halo the size of
    # the Colombia star would punch out most of Colombia.
    star_r = max(3.0, 0.014 * s)
    for lon, lat in star_lonlats:
        _draw_star(draw, proj, cx, cy, r, lon, lat, star_r,
                   fill=GLOBE_REGION_STAR, halo=False)

    return np.asarray(img, dtype=np.uint8).astype(np.int32)
