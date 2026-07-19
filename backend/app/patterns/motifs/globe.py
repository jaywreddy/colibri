from __future__ import annotations

import numpy as np

from ..geo.render import render_globe

# ---------------------------------------------------------------------------
# North-Atlantic orthographic globe, built on REAL geography.
#
# The disk is a dark ocean (negative space). Land, a foreshortened graticule,
# and the limb ring are gold. Unlike the earlier hand-authored coastlines, the
# land here is Natural Earth 110m vector data (public domain; vendored under
# ``app/patterns/geo/data/`` with a license note) projected with a true
# orthographic projection and rasterized in :mod:`app.patterns.geo`.
#
# Center (LAT0, LON0) = (32 N, 52 W) — the mid North Atlantic, tilted north so
# the "Colombia -> US -> Europe" triangle the client asked for all reads at once:
#
#     upper-left   : continental US / Atlantic seaboard, Great Lakes, Hudson Bay
#     lower-left   : Central America + Colombia (marked) at the top of S. America
#     top-center   : Greenland (Iceland to its east in open ocean)
#     upper-right  : western Europe (Iberia, British Isles, France, Scandinavia)
#     lower-right  : NW Africa bulge
#     center       : open North Atlantic (dark, threaded by the graticule)
#
# Colombia is emphasized: its country polygon is re-filled crisply and a gold
# star with a dark keep-out halo marks it so it reads as a distinct emblem even
# though it sits on gold land. See ``geo/render.py`` for the projection,
# adaptive Douglas-Peucker simplification, limb clipping, and highlight logic.
# ---------------------------------------------------------------------------

LAT0 = 32.0    # projection center latitude (~32 N — pulls the view north)
LON0 = -52.0   # projection center longitude (mid North Atlantic)


def globe_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — North-Atlantic cartographic globe.

    True where gold (land + graticule + limb + Colombia marker); False on the
    ocean and outside the disk. Built from real Natural Earth 110m geography via
    :func:`app.patterns.geo.render.render_globe`, centered on ~32 N / 52 W so
    Colombia, the continental US, and western Europe all read at once, with
    Colombia emphasized. Signature and output semantics are unchanged from the
    prior versions, so callers (colibri_globe_phase, plates composition) keep
    working untouched.

    ``rotation_deg`` spins the earth about its polar axis by offsetting the
    orthographic projection's center longitude — with real geography this is a
    TRUE rotation (continents wheel past the limb), used by
    ``globe-rotation-stereo`` to interlace two rotation phases behind a slit
    barrier. ``rotation_deg=0`` is exactly the default view.
    """
    del extent_um  # silhouette is scale-free; caller controls cell_um to hit extent
    return render_globe(
        n_grid, lat0=LAT0, lon0=LON0 + rotation_deg, highlight_country="Colombia"
    )
