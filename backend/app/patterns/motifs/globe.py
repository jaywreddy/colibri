from __future__ import annotations

import numpy as np

from ...region_art import Region, RegionArt
from ..geo.render import (
    GLOBE_REGION_COLOMBIA,
    GLOBE_REGION_EUROPE,
    GLOBE_REGION_FURNITURE,
    GLOBE_REGION_LAND,
    GLOBE_REGION_STAR,
    GLOBE_REGION_USA,
    globe_region_labels,
    render_globe,
)

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
    prior versions, so callers (colibri_globe_lenticular, colibri_globe_moire,
    plates composition) keep working untouched.

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


# ---------------------------------------------------------------------------
# SINGLE-LAYER DIFFRACTION view: ONE globe, colour by REGION.
#
# Centre (ATLANTIC_LAT0, ATLANTIC_LON0) = (36 N, 56 W). The old two-view switch
# could afford a per-view centre; a static single-ply face gets ONE, and it has
# to hold California, Colombia and western Europe inside the limb at once. The
# angular distances from 36 N / 56 W are
#
#     Los Angeles 50°   Sacramento 51°   Bogotá 36°
#     Lisbon      37°   Madrid     41°   London 42°   Paris 44°   Rome 52°
#
# so nothing the design cares about sits past ~52° — radius 0.79 of the disk,
# 0.62 of full scale radially — while the old 32 N / 52 W put California at 55-57°
# (radius 0.84, badly smeared along the limb) and, going the other way, the pure
# minimax centre 40 N / 56 W (max 50.5°) buys one degree by cutting the shoulder
# of South America off at the bottom limb. 36 N / 56 W is the compromise that
# keeps the whole Colombia→US→Europe triangle comfortably on the near
# hemisphere with Brazil's bulge and the NW Africa bulge still closing the frame.
# ---------------------------------------------------------------------------
ATLANTIC_LAT0 = 36.0
ATLANTIC_LON0 = -56.0
ATLANTIC_DISK_FRAC = 0.46

COLOMBIA_STAR_LONLAT = (-73.5, 4.6)      # the historian's home
CALIFORNIA_STAR_LONLAT = (-119.5, 36.5)  # the engineer's

# Periods from plates.SINGLE_PLY_LEAF_HUE_PERIODS_UM (4.15 … 6.02 µm, blue→red)
# — the ladder region_art.py names as the safe choice, every rung clearing the
# 2 µm litho floor AND the 1 µm die-finish open at 50 % duty.
#
# The assignment is driven by ADJACENCY, not by taste: a colour boundary only
# reads if the two periods either side of it are far apart. Generic land takes
# the bluest rung because it is what the three named regions actually touch —
# USA|Canada and USA|Mexico are 2 rungs (16 %), Colombia|Venezuela/Ecuador/Peru
# is 5 rungs (45 %), and Europe (which touches only the excluded east, out at
# the limb) is 4. The graticule and limb take 5.19, 3 rungs off the land they
# cross at the bottom of the disk. The one close pair, Europe|Colombia at 1
# rung, never touch and sit on opposite sides of the globe.
ATLANTIC_LAND_PERIOD_UM = 4.15
ATLANTIC_USA_PERIOD_UM = 4.82
ATLANTIC_FURNITURE_PERIOD_UM = 5.19
ATLANTIC_EUROPE_PERIOD_UM = 5.59
ATLANTIC_COLOMBIA_PERIOD_UM = 6.02


def globe_regions(
    n_px: int,
    lat0: float = ATLANTIC_LAT0,
    lon0: float = ATLANTIC_LON0,
    land_period_um: float = ATLANTIC_LAND_PERIOD_UM,
    usa_period_um: float = ATLANTIC_USA_PERIOD_UM,
    europe_period_um: float = ATLANTIC_EUROPE_PERIOD_UM,
    colombia_period_um: float = ATLANTIC_COLOMBIA_PERIOD_UM,
    furniture_period_um: float = ATLANTIC_FURNITURE_PERIOD_UM,
    disk_frac: float = ATLANTIC_DISK_FRAC,
) -> RegionArt:
    """The Atlantic globe as a :class:`~app.region_art.RegionArt` label map.

    ``n_px`` square, y DOWN, over the centrepiece art box. The ocean is region 0
    — BARE GLASS, so the sea is the dark negative space the gold continents sit
    in (the same figure/ground render_globe uses, and the only "colour" on the
    ladder that costs no gold). Land is written as a vertical 50 % grating at
    its region's period, so each of generic land / USA / Europe / Colombia
    flashes its own hue under a lamp, as do the graticule and limb ring at a
    period of their own (they are closed curves, and a closed curve of SOLID
    gold does not survive the writer's heal — see ``GLOBE_REGION_FURNITURE``).
    The two stars ARE solid: plain bright metal, and simply connected, so they
    are safe where the ring is not.
    """
    labels = globe_region_labels(
        int(n_px), lat0=float(lat0), lon0=float(lon0), disk_frac=float(disk_frac),
        star_lonlats=(COLOMBIA_STAR_LONLAT, CALIFORNIA_STAR_LONLAT),
    )
    return RegionArt(
        np.asarray(labels, dtype=np.int32),
        {
            GLOBE_REGION_LAND: Region("land", float(land_period_um)),
            GLOBE_REGION_USA: Region("usa", float(usa_period_um)),
            GLOBE_REGION_EUROPE: Region("europe", float(europe_period_um)),
            GLOBE_REGION_COLOMBIA: Region("colombia", float(colombia_period_um)),
            GLOBE_REGION_FURNITURE: Region("graticule-limb", float(furniture_period_um)),
            GLOBE_REGION_STAR: Region("stars", 0.0),
        },
    )
