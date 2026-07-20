"""Colibrí wing-beat parallax-barrier switch.

History: this slug originally shipped the banned "phase-shift overlay"
construction — wing pose A carved into a stripe carrier on the FRONT face,
pose B at anti-phase on the BACK. That construction is structurally incapable
of switching under honest parallax: the front-face image never moves with
tilt, so pose A stayed visible at every angle (measured channel visibility
1.000/0.559 with 0.924 lit overlap — both wing frames visible at once).
Rebuilt 2026-07 as a true parallax barrier: BOTH poses live in the BACK layer
as interlaced half-period column channels and the FRONT is a pure full-field
slit mask, exactly the jp-monogram-phase / colibri-globe-lenticular
architecture (audited ~0.90/0.00 channel separation at ±p/4 back shift).
The module filename and slug are kept for cache/registry continuity.

The two channels hold the SAME hummingbird in two wing poses — body, head,
beak and tail pixel-registered — so the channel swap reads as a wing BEAT:
the bird itself never changes, only its wings move as the box rocks.
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
from ..motifs import colibri


@register
class ColibriFlapPhase(Pattern):
    slug = "colibri-flap-phase"
    name = "Colibrí wing-beat barrier switch"
    description = (
        "A parallax-barrier (lenticular) tile holding the SAME hummingbird in "
        "two wing poses, interlaced in alternating half-period columns of the "
        "back layer — wings swept up in the hover V in one channel, caught "
        "mid-downstroke with speed slivers trailing the tips in the other — "
        "under a pure slit mask on the front whose open slits straddle the "
        "channel boundaries. Body, head, beak and tail are pixel-registered "
        "between the two poses, so as Snell-refracted parallax gates one "
        "channel then the other, the wings appear to BEAT while the bird "
        "holds still — the switch completing at a quarter-period parallax "
        "shift each way as the box rocks left ↔ right."
    )
    tags = ["moire", "lenticular", "tilt-reveal", "animation", "Colombia"]
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
            "colibri-flap-phase interlace",
            slit_period_um=slit_period_um,
            extent_um=extent_um,
        )

        # Front: FULL-FIELD vertical slit barrier — never clipped to any
        # silhouette (a silhouette-gated comb is itself a static front image;
        # the banned construction class). 1 - slit_duty so the "duty"
        # parameter reads naturally as "fraction of the surface that is open
        # slit". The half-period translate places each open slit STRADDLING
        # the boundary between the two back channels — that is what makes the
        # switch symmetric in tilt sign.
        barrier = linear_grating(slit_period_um, 1 - slit_duty, extent)
        barrier = affinity.translate(barrier, xoff=slit_period_um / 2)
        front = ensure_multipolygon(crop(barrier, extent))

        # Rasterize the two poses of the SAME bird onto a shared pixel grid —
        # body/head/beak/tail pixel-registered (colibri._draw_colibri only
        # swaps the wing set) — so each pose can be sliced into vertical
        # strips that interlace one to one with the slit lattice.
        pose_a = colibri.colibri_silhouette(extent, n_grid=n_grid, pose="up")
        pose_b = colibri.colibri_silhouette(extent, n_grid=n_grid, pose="down")

        cell_um = extent_um / n_grid
        # Number of pixel columns per slit period. Each slit period gets one
        # pose half: left half = wings-up (view_a), right half = wings-down
        # (view_b).
        cols_per_period = max(2, int(round(slit_period_um / cell_um)))
        half = cols_per_period // 2

        cols = np.arange(n_grid)
        phase = (cols % cols_per_period) < half
        a_strips = pose_a & phase[None, :]
        b_strips = pose_b & (~phase[None, :])

        # Scene A = wings-up interlace (the -tilt side sees this);
        # Scene B = wings-down interlace (the +tilt side sees this).
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
                "view_a_pose": "wings-up",
                "view_b_pose": "wings-down",
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
