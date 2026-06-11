from __future__ import annotations

import math

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon

from .._helpers import check_lattice_budget, crop_parts


def hex_facets(
    extent_um: tuple[float, float] | float,
    period_um: float,
    duty: float = 0.7,
    rotation_deg: float = 0.0,
) -> MultiPolygon:
    """Hexagonal close-packed lattice of facets — Muzo emerald crystal faces.

    `period_um` is center-to-center spacing. `duty` controls hexagon radius
    relative to the half-period (duty=1.0 → hexagons touch).

    The ~50k hexes are built via shapely's bulk array API and combined by
    CONCATENATION, not ``unary_union`` — the hexes are pairwise disjoint over
    the entire parameter range (overlap would only start above duty ≈ 1.15,
    beyond the ParamSpec max of 0.95), so the union was a pure ~11 s/layer
    no-op. :func:`crop_parts` clips boundary-crossing members individually;
    downstream consumers are fill-only (rasterize / to_svg).
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
    check_lattice_budget(
        (2 * nx + 1) * (2 * ny + 1), "Emerald hex lattice",
        extent_um=max(w, h), period_um=period_um,
    )
    # Same lattice math as the original per-cell loops (j-major, i-minor),
    # kept operation-for-operation identical so coordinates match bitwise.
    jdx = np.arange(-ny, ny + 1, dtype=np.float64)
    idx = np.arange(-nx, nx + 1, dtype=np.float64)
    jj, ii = np.meshgrid(jdx, idx, indexing="ij")
    cx = (ii * dx + (0.5 * dx) * (jj % 2.0)).ravel()
    cy = (jj * dy).ravel()
    # Hexagon vertex offsets — the exact per-vertex floats of _hexagon().
    offs = [
        (r * math.cos(math.radians(60 * k + 30)), r * math.sin(math.radians(60 * k + 30)))
        for k in range(6)
    ]
    verts = np.empty((cx.size, 6, 2), dtype=np.float64)
    for k, (ox, oy) in enumerate(offs):
        verts[:, k, 0] = cx + ox
        verts[:, k, 1] = cy + oy
    mp = shapely.multipolygons(shapely.polygons(verts))
    if rotation_deg:
        mp = affinity.rotate(mp, rotation_deg, origin=(0, 0))
    return crop_parts(shapely.get_parts(mp), extent_um)
