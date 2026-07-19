from __future__ import annotations

import numpy as np

from .._helpers import raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..geo.render import render_globe

# ---------------------------------------------------------------------------
# EXPERIMENTAL "globe duo" phase-switch (Track D).
#
# Same mechanism as colibri-globe-phase: two silhouettes carved into one
# high-frequency vertical stripe carrier with a half-period phase offset, so
# Snell-refracted parallax through the substrate biases which carrier phase the
# eye samples — one view emerges with tilt while the other recedes.
#
# Here BOTH silhouettes are true orthographic globes (Natural Earth 110m, via
# app.patterns.geo), centered on the couple's two homes. Same disc size and
# position for both, so the tilt switch reads as the SAME globe ROTATING between
# California and Colombia:
#
#   View A (front, carrier phase 0)  — orthographic centered on CALIFORNIA
#       (~37 N / 119 W). US West Coast prominent, Pacific to the left, Baja
#       hinted below, Rockies→Midwest→eastern seaboard fading to the right limb.
#       A gold star marks California (~36.5 N / 119.5 W). No country fill: the
#       state is part of the USA polygon, so a country re-fill would gold-flood
#       all of North America; the star alone carries the emphasis.
#
#   View B (back, carrier phase π)   — orthographic centered on COLOMBIA
#       (~4.5 N / 74 W). Colombia central with its emphasis re-fill + star,
#       Caribbean and Central America arcing up-left, the body of South America
#       below, Africa hinted on the right limb. Reuses the existing Colombia
#       highlight logic in geo.render.
# ---------------------------------------------------------------------------

# Projection centers (lat, lon) and star markers (lon, lat) — see geo/render.py.
CAL_LAT0, CAL_LON0 = 37.0, -119.0
CAL_STAR_LONLAT = (-119.5, 36.5)

COL_LAT0, COL_LON0 = 4.5, -74.0
COL_STAR_LONLAT = (-73.5, 4.6)


@register
class GlobeDuoPhase(Pattern):
    slug = "globe-duo-phase"
    name = "Globe duo phase-shift — California ↔ Colombia"
    description = (
        "Two true orthographic globes carved into a single high-frequency "
        "stripe carrier with a half-period phase offset. Front holds the "
        "California-centered view (carrier phase 0); back holds the "
        "Colombia-centered view (carrier phase π — physically shifted by Λ/2). "
        "Both discs are the same size and position, so Snell-refracted parallax "
        "through the substrate makes the globe appear to ROTATE between the "
        "couple's two homes as the box tilts: the US West Coast (star on "
        "California) resolves one way, Colombia (star, emphasized) the other. "
        "Head-on, the two interlace into a fine line pattern with both "
        "hemispheres half-visible at once. Geography is Natural Earth 110m "
        "vector data projected and rasterized in app.patterns.geo."
    )
    tags = ["moire", "phase-shift", "tilt-reveal", "globe", "Colombia", "Global Travel"]
    tier = 1
    theme = "Colombia"
    render_recipe = "phase_shift_overlay"
    params = [
        ParamSpec("period_um", "Carrier period", "float", 20.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Resolve the carrier with ≥8 samples per period so the half-period
        # phase offset between front and back is geometrically clean. Mirrors
        # colibri_globe_phase so both discs land on an identical grid.
        n_grid = max(384, int(extent_um / max(1.0, period_um / 10)))
        cell_um = extent_um / n_grid

        # View A: California. No country re-fill (star carries the emphasis).
        view_a = render_globe(
            n_grid,
            lat0=CAL_LAT0,
            lon0=CAL_LON0,
            highlight_country=None,
            highlight_star=True,
            star_lonlat=CAL_STAR_LONLAT,
        )
        # View B: Colombia, emphasized (re-fill + star) — same disc geometry.
        view_b = render_globe(
            n_grid,
            lat0=COL_LAT0,
            lon0=COL_LON0,
            highlight_country="Colombia",
            highlight_star=True,
            star_lonlat=COL_STAR_LONLAT,
        )

        # BARRIER INTERLACE (Task 3). Both globes live on the BACK layer,
        # interleaved in alternating lanes (lane pitch = half the barrier period):
        # California (view A) in the even lanes, Colombia (view B) in the odd. The
        # FRONT layer is a NEUTRAL slit barrier (open duty 0.5 = one lane) over the
        # union of both discs — no image, just the gate. Snell parallax slides the
        # barrier across the lanes as the piece tilts, so one tilt shows ONLY
        # California and the other ONLY Colombia (a hard swap, not a redistribution
        # of a shared carrier). Barrier phase −0.25 straddles an A|B lane boundary
        # head-on so ± tilt reveals A/B symmetrically.
        period_pix = period_um / cell_um
        cols = np.arange(n_grid, dtype=np.float32)
        lane = np.floor(cols / (period_pix / 2.0)).astype(int)
        even_lane = (lane % 2 == 0)[None, :]
        union = view_a | view_b
        interleave_back = (view_a & even_lane) | (view_b & ~even_lane)
        frac = (cols / period_pix) % 1.0
        barrier_bar = ((frac >= 0.25) & (frac < 0.75))[None, :]

        front_mask = union & barrier_bar
        back_mask = interleave_back

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        return GeneratedPattern(
            front=front_poly,
            back=back_poly,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=period_um * 0.5,
            extra={
                "carrier_period_um": period_um,
                "phase_offset_pix": period_pix / 2.0,
                "view_a_center_latlon": (CAL_LAT0, CAL_LON0),
                "view_b_center_latlon": (COL_LAT0, COL_LON0),
            },
            recipe_data={
                # Slit-normal axis for the parallax-shift projection in the
                # shader. Vertical stripes → horizontal switch axis (+X).
                "switch_axis_deg": 0.0,
                "carrier_period_um": period_um,
            },
        )
