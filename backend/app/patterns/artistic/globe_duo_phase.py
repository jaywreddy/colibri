"""California ↔ Colombia duo-globe parallax-barrier switch.

History: this slug originally shipped a "phase-shift overlay" (California
globe on the FRONT face, Colombia globe on the BACK at anti-phase), then a
HALF rebuild whose front slit comb was clipped to the union of both discs.
Both constructions are structurally incapable of a clean switch under honest
parallax: a front-face image never moves with tilt, and a union-gated comb IS
a front image (its envelope is the two-disc union — measured 0.748/0.037 vs
0.314/0.568 channel visibility, the front view never dropping below ~0.31).
Rebuilt 2026-07 as a true parallax barrier: BOTH globes live in the BACK
layer as interlaced half-period column channels and the FRONT is a pure
full-field slit mask, exactly the jp-monogram-phase / colibri-globe-lenticular
architecture (audited ~0.90/0.00 channel separation at ±p/4 back shift).
The module filename and slug are kept for cache/registry continuity.

Both views are true orthographic globes (Natural Earth 110m, via
app.patterns.geo), centered on the couple's two homes. Same disc size and
position for both, so the tilt switch reads as the SAME globe ROTATING
between California and Colombia:

  View A (channel A, -tilt) — orthographic centered on CALIFORNIA
      (~37 N / 119 W). US West Coast prominent, Pacific to the left, Baja
      hinted below, Rockies→Midwest→eastern seaboard fading to the right limb.
      A gold star marks California (~36.5 N / 119.5 W). No country fill: the
      state is part of the USA polygon, so a country re-fill would gold-flood
      all of North America; the star alone carries the emphasis.

  View B (channel B, +tilt)  — orthographic centered on COLOMBIA
      (~4.5 N / 74 W). Colombia central with its emphasis re-fill + star,
      Caribbean and Central America arcing up-left, the body of South America
      below, Africa hinted on the right limb. Reuses the existing Colombia
      highlight logic in geo.render.
"""
from __future__ import annotations

import math

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon

from .._helpers import (
    check_lattice_budget,
    crop,
    exterior_tilt_deg,
    linear_grating,
    raster_to_polygons,
)
from ..base import (
    GeneratedPattern,
    ParamSpec,
    Pattern,
    ensure_multipolygon,
    register,
)
from ..geo.render import render_globe

# Projection centers (lat, lon) and star markers (lon, lat) — see geo/render.py.
# Imported by plates._centerpiece_masks so the plate centerpiece renders the
# identical two views.
CAL_LAT0, CAL_LON0 = 37.0, -119.0
CAL_STAR_LONLAT = (-119.5, 36.5)

COL_LAT0, COL_LON0 = 4.5, -74.0
COL_STAR_LONLAT = (-73.5, 4.6)


@register
class GlobeDuoPhase(Pattern):
    slug = "globe-duo-phase"
    name = "Globe duo switch — California ↔ Colombia"
    description = (
        "A parallax-barrier (lenticular) tile. The back layer holds BOTH "
        "orthographic globes interlaced in alternating half-period columns — "
        "the California-centered view in one channel, the Colombia-centered "
        "view (emphasis re-fill + star) in the other — and the front layer is "
        "a pure slit mask whose open slits straddle the channel boundaries. "
        "Snell-refracted parallax through the substrate gates which channel "
        "the eye sees: both discs share size and position, so tilting makes "
        "the globe appear to ROTATE between the couple's two homes, with the "
        "switch completing at a quarter-period parallax shift. Head-on, the "
        "two hemispheres blend 50/50 through the slits. Geography is Natural "
        "Earth 110m vector data projected and rasterized in app.patterns.geo."
    )
    tags = ["moire", "lenticular", "tilt-reveal", "globe", "Colombia", "Global Travel"]
    tier = 1
    theme = "Colombia"
    render_recipe = "stereo_lenticular"
    params = [
        ParamSpec("slit_period_um", "Slit period", "float", 60.0, 8.0, 200.0, 0.5, "μm"),
        ParamSpec("slit_duty", "Slit duty", "float", 0.4, 0.1, 0.9, 0.05),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        slit_period_um: float = 60.0,
        slit_duty: float = 0.4,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Interlace rect-count estimate: raster_to_polygons emits one rect per
        # horizontal run, and the channel masks chop every silhouette row into
        # ~extent/period runs per channel. Worst case (silhouettes covering the
        # full grid) is n_grid rows × stripes-per-extent — gate before building.
        n_grid = max(192, int(extent_um / max(2.0, slit_period_um / 8)))
        n_stripes = int(math.ceil(extent_um / slit_period_um))
        check_lattice_budget(
            n_grid * n_stripes,
            "globe-duo-phase interlace",
            slit_period_um=slit_period_um,
            extent_um=extent_um,
        )

        # Front: FULL-FIELD vertical slit barrier — never clipped to any
        # silhouette (a union-gated comb is itself a static front image; that
        # was the measured ~0.31 front residual of the half rebuild). 1 -
        # slit_duty so the "duty" parameter reads naturally as "fraction of
        # the surface that is open slit". The half-period translate places
        # each open slit STRADDLING the boundary between the two back
        # channels — that is what makes the switch symmetric in tilt sign.
        barrier = linear_grating(slit_period_um, 1 - slit_duty, extent)
        barrier = affinity.translate(barrier, xoff=slit_period_um / 2)
        front = ensure_multipolygon(crop(barrier, extent))

        # Rasterize the two globe views onto a shared pixel grid so each can
        # be sliced into vertical strips that interlace one to one with the
        # slit lattice.
        view_a_raster = render_globe(
            n_grid,
            lat0=CAL_LAT0,
            lon0=CAL_LON0,
            highlight_country=None,
            highlight_star=True,
            star_lonlat=CAL_STAR_LONLAT,
        )
        view_b_raster = render_globe(
            n_grid,
            lat0=COL_LAT0,
            lon0=COL_LON0,
            highlight_country="Colombia",
            highlight_star=True,
            star_lonlat=COL_STAR_LONLAT,
        )

        cell_um = extent_um / n_grid
        # Number of pixel columns per slit period. Each slit period gets one
        # California half and one Colombia half: left half = California
        # (view_a), right half = Colombia (view_b).
        cols_per_period = max(2, int(round(slit_period_um / cell_um)))
        half = cols_per_period // 2

        cols = np.arange(n_grid)
        phase = (cols % cols_per_period) < half
        a_strips = view_a_raster & phase[None, :]
        b_strips = view_b_raster & (~phase[None, :])

        # Scene A = California interlace (the -tilt side sees this);
        # Scene B = Colombia interlace (the +tilt side sees this).
        # Sign convention per the parallax audit: +tilt/+shift reveals view_b.
        view_a = raster_to_polygons(a_strips.astype(np.uint8), cell_um, extent)
        view_b = raster_to_polygons(b_strips.astype(np.uint8), cell_um, extent)

        # Back layer = BOTH interlace channels, per the barrier architecture
        # (all image content behind the slits). The channels occupy
        # complementary column sets (they can share edges but never overlap)
        # and everything downstream is fill-only, so plain concatenation is
        # safe — never feed this MultiPolygon to a GEOS boolean.
        back = MultiPolygon([*view_a.geoms, *view_b.geoms])

        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=max(0.5, slit_period_um / 20),
            min_feature_um=slit_period_um * min(slit_duty, 1 - slit_duty),
            extra={
                # Switch completes at a quarter-period shift (slit straddles
                # the channel boundary); the clean first zone ends at p/2.
                "switch_half_angle_deg": exterior_tilt_deg(slit_period_um / 4),
                "zone_half_angle_deg": exterior_tilt_deg(slit_period_um / 2),
                "view_a_center_latlon": (CAL_LAT0, CAL_LON0),
                "view_b_center_latlon": (COL_LAT0, COL_LON0),
            },
            extra_layers={
                "view_a": ensure_multipolygon(view_a),
                "view_b": ensure_multipolygon(view_b),
            },
            recipe_data={
                "slit_axis_deg": 0.0,
                "slit_period_um": slit_period_um,
            },
        )
