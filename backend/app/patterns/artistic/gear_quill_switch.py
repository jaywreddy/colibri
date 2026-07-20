"""Gear ↔ quill+book parallax-barrier switch — "the engineer and the historian".

History: this slug originally shipped a "phase-shift overlay" (gear on the
FRONT face, quill+book on the BACK at anti-phase), then a HALF rebuild whose
front slit comb was clipped to the union of both silhouettes. Both
constructions are structurally incapable of a clean switch under honest
parallax: a front-face image never moves with tilt, and a union-gated comb IS
a front image (its envelope is the union — measured 0.732/0.010 vs
0.276/0.564 channel visibility, the front motif never fully vanishing).
Rebuilt 2026-07 as a true parallax barrier: BOTH silhouettes live in the BACK
layer as interlaced half-period column channels and the FRONT is a pure
full-field slit mask, exactly the jp-monogram-phase / colibri-globe-lenticular
architecture (audited ~0.90/0.00 channel separation at ±p/4 back shift).
The module filename and slug are kept for cache/registry continuity.

In a box the plate compositor dispatches the paired silhouettes through
``plates._centerpiece_masks`` and bakes the same barrier interlace via
``SWITCH_INTERLACE_SLUGS``; this standalone ``generate`` mirrors that
architecture so the pattern registers in the catalog and the ``/patterns``
endpoint can preview it.
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
from ..motifs.lab.gear_quill import gear_silhouette, quill_book_silhouette


@register
class GearQuillSwitch(Pattern):
    slug = "gear-quill-switch"
    name = "Gear ↔ quill barrier switch"
    description = (
        "A parallax-barrier (lenticular) tile. The back layer holds BOTH "
        "silhouettes interlaced in alternating half-period columns — a "
        "toothed gear (the engineer) in one channel, an open book with a "
        "quill pen (the historian) in the other — and the front layer is a "
        "pure slit mask whose open slits straddle the channel boundaries. "
        "Snell-refracted parallax through the substrate gates which channel "
        "the eye sees: tilting one way shows the gear, the other reveals the "
        "quill+book, with the switch completing at a quarter-period parallax "
        "shift. Head-on, both crafts blend 50/50 through the slits — a "
        "tilt-switch portrait of the couple's two callings."
    )
    tags = ["moire", "lenticular", "tilt-reveal", "engineer", "historian"]
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
            "gear-quill-switch interlace",
            slit_period_um=slit_period_um,
            extent_um=extent_um,
        )

        # Front: FULL-FIELD vertical slit barrier — never clipped to any
        # silhouette (a union-gated comb is itself a static front image; that
        # was the measured front residual of the half rebuild). 1 - slit_duty
        # so the "duty" parameter reads naturally as "fraction of the surface
        # that is open slit". The half-period translate places each open slit
        # STRADDLING the boundary between the two back channels — that is what
        # makes the switch symmetric in tilt sign.
        barrier = linear_grating(slit_period_um, 1 - slit_duty, extent)
        barrier = affinity.translate(barrier, xoff=slit_period_um / 2)
        front = ensure_multipolygon(crop(barrier, extent))

        # Rasterize the two silhouettes onto a shared pixel grid so each can
        # be sliced into vertical strips that interlace one to one with the
        # slit lattice.
        gear = gear_silhouette(extent, n_grid=n_grid)
        quill = quill_book_silhouette(extent, n_grid=n_grid)

        cell_um = extent_um / n_grid
        # Number of pixel columns per slit period. Each slit period gets one
        # gear half and one quill half: left half = gear (view_a), right
        # half = quill+book (view_b).
        cols_per_period = max(2, int(round(slit_period_um / cell_um)))
        half = cols_per_period // 2

        cols = np.arange(n_grid)
        phase = (cols % cols_per_period) < half
        gear_strips = gear & phase[None, :]
        quill_strips = quill & (~phase[None, :])

        # Scene A = gear interlace (the -tilt side sees this);
        # Scene B = quill+book interlace (the +tilt side sees this).
        # Sign convention per the parallax audit: +tilt/+shift reveals view_b.
        view_a = raster_to_polygons(gear_strips.astype(np.uint8), cell_um, extent)
        view_b = raster_to_polygons(quill_strips.astype(np.uint8), cell_um, extent)

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
