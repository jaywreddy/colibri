"""Real-geography support for the cartographic globe motif.

Zero external geo deps (no geopandas/shapely-heavy paths): Natural Earth 110m
GeoJSON is parsed with the stdlib :mod:`json` module into flat numpy ring
arrays, projected orthographically in :mod:`.ortho`, and rasterized with Pillow
in :mod:`.render`. See ``data/LICENSE_NOTE.md`` (public-domain source).
"""

from .ortho import Orthographic, project_ring
from .render import render_globe, load_land_rings, load_country_rings

__all__ = [
    "Orthographic",
    "project_ring",
    "render_globe",
    "load_land_rings",
    "load_country_rings",
]
