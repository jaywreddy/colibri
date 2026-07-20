"""J+P monogram ↔ heart parallax-barrier switch.

History: this slug originally shipped a "phase-shift overlay" — monogram
carrier-halftone on the FRONT face, heart on the BACK at anti-phase. That
construction is structurally incapable of switching under honest parallax:
the front-face image never moves, so the monogram stayed visible at every
tilt (measured region contrast 0.52-0.54 across the whole ±1.5-period sweep)
and the 3D "switch" existed only via the phase_shift_overlay shader's
view-sign bias cheat. Rebuilt 2026-07 as a parallax barrier: BOTH images
live in the BACK layer as interlaced half-period column channels and the
FRONT is a pure slit mask, exactly the colibri-globe-lenticular architecture
(measured 0.819/0.000 channel separation at ±p/4 shift on that geometry).
The module filename and slug are kept for cache/registry continuity.
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
from ..motifs import monogram


@register
class JPMonogramPhase(Pattern):
    slug = "jp-monogram-phase"
    name = "J+P ↔ heart switch"
    description = (
        "A parallax-barrier (lenticular) keepsake tile. The back layer holds "
        "BOTH images interlaced in alternating half-period columns — the "
        "J + P monogram in one channel, a heart-with-globe in the other — "
        "and the front layer is a pure slit mask whose open slits straddle "
        "the channel boundaries. Snell-refracted parallax through the "
        "substrate gates which channel the eye sees: tilting one way shows "
        "the monogram, the other reveals the heart, with the switch "
        "completing at a quarter-period parallax shift. Head-on, both "
        "images blend 50/50 through the slits."
    )
    tags = ["moire", "lenticular", "tilt-reveal", "monogram", "Global Travel"]
    tier = 1
    theme = "Global Travel"
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
            "jp-monogram-phase interlace",
            slit_period_um=slit_period_um,
            extent_um=extent_um,
        )

        # Front: vertical slit barrier. 1 - slit_duty so the "duty" parameter
        # reads naturally as "fraction of the surface that is open slit". The
        # half-period translate places each open slit STRADDLING the boundary
        # between the two back channels — that is what makes the switch
        # symmetric in tilt sign (audited: 0.819/0.000 separation at ±p/4).
        barrier = linear_grating(slit_period_um, 1 - slit_duty, extent)
        barrier = affinity.translate(barrier, xoff=slit_period_um / 2)
        front = ensure_multipolygon(crop(barrier, extent))

        # Rasterize the two keepsake images onto a shared pixel grid so each
        # can be sliced into vertical strips that interlace one to one with
        # the slit lattice.
        mono = monogram.jp_monogram_silhouette(extent, n_grid=n_grid)
        heart = monogram.heart_globe_silhouette(extent, n_grid=n_grid)

        cell_um = extent_um / n_grid
        # Number of pixel columns per slit period. Each slit period gets one
        # monogram half and one heart half: left half = monogram (view_a),
        # right half = heart (view_b).
        cols_per_period = max(2, int(round(slit_period_um / cell_um)))
        half = cols_per_period // 2

        cols = np.arange(n_grid)
        phase = (cols % cols_per_period) < half
        mono_strips = mono & phase[None, :]
        heart_strips = heart & (~phase[None, :])

        # Scene A = J+P monogram interlace (the -tilt side sees this);
        # Scene B = heart-globe interlace (the +tilt side sees this).
        # Sign convention per the parallax audit: +tilt/+shift reveals view_b.
        view_a = raster_to_polygons(mono_strips.astype(np.uint8), cell_um, extent)
        view_b = raster_to_polygons(heart_strips.astype(np.uint8), cell_um, extent)

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
