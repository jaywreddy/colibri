from __future__ import annotations

import math

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from .base import ensure_multipolygon


# Hard ceiling on the lattice cells a single generator call may BUILD
# (pre-crop). GEOS polygon memory is ~4-5 kB per cell, and cell count grows
# quadratically with extent/period — extent 5000 um at period 4 um (both
# inside the UI slider ranges) is ~15M cells ≈ tens of GB of commit, which
# froze and bugchecked the dev machine three times on 2026-06-10. 400k cells
# is ~1.8 GB transient and a few seconds of build: generous for every
# legitimate moire use (defaults build ~100k), tight enough that no request
# can take down the host.
MAX_LATTICE_CELLS = 400_000


def check_lattice_budget(n_cells: int, what: str, **dials: float) -> None:
    """Refuse pattern builds whose lattice would not fit in memory.

    Raises ValueError with an actionable message (the /patterns and /boxes
    routes surface it verbatim as an HTTP 400 in the UI).
    """
    if n_cells <= MAX_LATTICE_CELLS:
        return
    knobs = ", ".join(f"{k}={v:g}" for k, v in dials.items())
    raise ValueError(
        f"{what} would build {n_cells:,} lattice cells ({knobs}); the cap is "
        f"{MAX_LATTICE_CELLS:,}. Increase period_um or shrink extent_um — the "
        "plate compositor upscales the pattern raster to fill the aperture, "
        "so patterns never need multi-mm extents at micron periods."
    )


def crop_box(extent_um: tuple[float, float]) -> Polygon:
    w, h = extent_um
    return box(-w / 2, -h / 2, w / 2, h / 2)


def crop(polys, extent_um: tuple[float, float]) -> MultiPolygon:
    """Whole-geometry boolean crop. Requires OGC-VALID input — a concatenated
    MultiPolygon with overlapping members makes GEOS throw ("side location
    conflict"). For bulk lattice output prefer :func:`crop_parts`."""
    return ensure_multipolygon(polys.intersection(crop_box(extent_um)))


def crop_parts(parts, extent_um: tuple[float, float]) -> MultiPolygon:
    """Clip polygon parts to the extent box WITHOUT a whole-geometry set-op.

    Strictly-inside parts (the overwhelming majority of a lattice) pass
    through untouched; only boundary-crossing parts are clipped, one geometry
    at a time via the vectorized array API; parts fully outside are dropped.

    The result is a plain CONCATENATION — members may overlap or share edges
    (OGC-invalid as a MultiPolygon, e.g. kanasu diamonds at duty > 1/√2),
    which is fine for the fill-only consumers (``rasterize`` / ``to_svg``;
    see ``plates._concat_polygons`` for the precedent) but must never be fed
    to a GEOS boolean op afterwards.
    """
    parts = np.asarray(parts, dtype=object)
    if parts.size == 0:
        return MultiPolygon()
    bbox = crop_box(extent_um)
    inside = shapely.contains_properly(bbox, parts)
    kept = list(parts[inside])
    crossing = parts[~inside]
    if crossing.size:
        clipped = shapely.intersection(crossing, bbox)
        for geom in clipped:
            if geom.is_empty:
                continue
            if isinstance(geom, Polygon):
                kept.append(geom)
            else:  # MultiPolygon / GeometryCollection — keep area parts only
                kept.extend(
                    g
                    for g in getattr(geom, "geoms", [])
                    if isinstance(g, Polygon) and not g.is_empty
                )
    return MultiPolygon(kept)


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
    """Square lattice of circular dots.

    NOTE: currently caller-less. If revived on a hot path: disks overlap
    whenever ``diameter > period`` — needs ``crop_parts`` (per-part clip),
    not plain concat + ``crop``.
    """
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
    """Binary amplitude Fresnel zone plate — odd zones opaque, even transparent.

    NOTE: currently caller-less. Rings are concentric and radially disjoint —
    SAFE to concat if revived (the annuli have holes, but no member overlaps
    another's hole, so rasterize's hole-punch stays correct).
    """
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
    """Convert a 2D binary numpy array (1=gold) into a MultiPolygon of
    axis-aligned rectangles — one per horizontal run of consecutive gold cells.

    `cell_um` is the size of each pixel in μm. The old implementation built
    one shapely box per cell and compacted them with ``unary_union`` (~4 s for
    a 1M-cell grid, ~75% of some patterns' generate time); this pure-numpy
    run-length merge is milliseconds and keeps SVG path counts ~5-10× below
    per-cell concatenation. Output rectangles share edges row-to-row
    (OGC-invalid as a MultiPolygon) — fine for the fill-only consumers
    (``rasterize`` / ``to_svg``), but NOT for GEOS set-ops. Run coordinates
    reproduce the old per-cell float math exactly so rasterization rounds
    identically.

    In every in-repo caller the cell grid exactly spans ``extent_um``, making
    the old post-union crop a geometric no-op; if a future caller passes a
    grid larger than the extent we fall back to a per-part clip.
    """
    h, w = grid.shape
    hx = w * cell_um / 2.0
    hy = h * cell_um / 2.0
    g = np.asarray(grid) > 0
    # Pad each row with a zero column on both sides; diff finds run edges.
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = g
    d = np.diff(padded, axis=1)
    rows, starts = np.nonzero(d == 1)  # run start columns (inclusive)
    _, ends = np.nonzero(d == -1)      # run end columns (exclusive)
    # np.nonzero scans row-major, so the k-th start and k-th end pair up.
    if starts.size == 0:
        return MultiPolygon()
    x0 = starts * cell_um - hx
    x1 = ((ends - 1) * cell_um - hx) + cell_um   # == last cell's old x0+cell
    y0 = hy - (rows + 1) * cell_um
    y1 = (hy - (rows + 1) * cell_um) + cell_um   # == old cell's y0+cell
    verts = np.empty((starts.size, 4, 2), dtype=np.float64)
    verts[:, 0, 0] = x0
    verts[:, 0, 1] = y0
    verts[:, 1, 0] = x1
    verts[:, 1, 1] = y0
    verts[:, 2, 0] = x1
    verts[:, 2, 1] = y1
    verts[:, 3, 0] = x0
    verts[:, 3, 1] = y1
    rects = shapely.polygons(verts)
    if 2.0 * hx > extent_um[0] + 1e-9 or 2.0 * hy > extent_um[1] + 1e-9:
        return crop_parts(rects, extent_um)
    return MultiPolygon(list(rects))


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
    """Concentric rings of opaque gold at the given radial period.

    NOTE: currently caller-less. Same concat-safety class as ``zone_plate``
    (concentric, radially disjoint annuli — SAFE to concat if revived).
    """
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
    """V-shaped (chevron) stripes filling the extent, period_um tall, with width amp_um.

    NOTE: currently caller-less. Left/right halves share the x=0 edge —
    plain concat is OGC-invalid; use ``crop_parts`` if revived.
    """
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
