from __future__ import annotations

import math

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon

from .._helpers import check_lattice_budget, crop_parts


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

    The ~100k diamonds are built in one shot via shapely's bulk array API and
    combined by CONCATENATION, not ``unary_union`` — the union of mutually
    disjoint diamonds was a ~25 s/layer no-op. Above duty = 1/√2 neighboring
    diamonds overlap (an OGC-invalid MultiPolygon), so the final crop uses
    :func:`crop_parts`, which clips boundary-crossing members one at a time
    and never runs a whole-geometry GEOS boolean. Downstream consumers are
    fill-only (rasterize / to_svg) and overlap-tolerant.
    """
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    w, h = extent_um
    diag = math.hypot(w, h) * 1.1
    half = period_um * duty / math.sqrt(2)
    n = int(diag / period_um) + 3
    check_lattice_budget(
        (2 * n + 1) ** 2, "Wayuu kanasü weave",
        extent_um=max(w, h), period_um=period_um,
    )
    # Same lattice math as the original per-cell loops (i-major, j-minor),
    # kept operation-for-operation identical so coordinates match bitwise.
    idx = np.arange(-n, n + 1, dtype=np.float64)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    cx = ((ii + 0.5 * (jj % 2.0)) * period_um).ravel()
    cy = (jj.ravel() * period_um) * 0.75
    verts = np.empty((cx.size, 4, 2), dtype=np.float64)
    verts[:, 0, 0] = cx
    verts[:, 0, 1] = cy - half
    verts[:, 1, 0] = cx + half
    verts[:, 1, 1] = cy
    verts[:, 2, 0] = cx
    verts[:, 2, 1] = cy + half
    verts[:, 3, 0] = cx - half
    verts[:, 3, 1] = cy
    mp = shapely.multipolygons(shapely.polygons(verts))
    if rotation_deg:
        mp = affinity.rotate(mp, rotation_deg, origin=(0, 0))
    return crop_parts(shapely.get_parts(mp), extent_um)
