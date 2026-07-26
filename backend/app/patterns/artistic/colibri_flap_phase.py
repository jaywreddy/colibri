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
architecture.
The module filename and slug are kept for cache/registry continuity.

Re-registered 2026-07 (audit): the rebuild inherited a raster derived from the
extent instead of from the slit lattice, so at this module's 60 µm default the
back interlace period (60.15 µm) did not match the front comb (60.0 µm) and the
slit sat ~p/8 off the channel boundary — the ±p/4 extinction the metadata
promised never happened and the off-channel pose stayed ~19 % visible at every
tilt, smearing the beat. The ~0.90/0.00 separation quoted for this construction
was measured on the p=40 family, whose lattice happened to come out exact. Both
the raster and the barrier phase are now solved from the slit period and the
extent, so the straddle registration holds for every legal parameter
combination.

Converged onto the shared ``_helpers.barrier_lattice`` solver afterwards, which
also moved this slug's phase from the mod-p/2 residue to the mod-p one (a p/2
shift of the comb, same straddle class). Only mod p pins WHICH channel sits on
the +x side of every slit; under mod p/2 the "+tilt shows the downstroke"
convention below silently flipped with the extent slider.

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
    barrier_lattice,
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

# Silhouette quality floor: below ~192 raster rows the beak and the wing-tip
# slivers break up. Handed to barrier_lattice, which trades it off against the
# litho floor — that floor is a DESIGN constraint here rather than a heal: the
# ParamSpec bounds plus the helper's cols_per_period cap keep the front slit,
# the front bar and the back interlace cell all at or above it, so the
# GeneratedPattern floor check never has to fire.
MIN_GRID_ROWS = 192


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
    # Bounds are set by the litho floor, not by taste: the front comb's narrowest
    # feature is period·min(duty, 1-duty), so period ≥ 20 µm with duty in
    # [0.2, 0.8] keeps the front slit AND the front bar at ≥ 4 µm, while the back
    # interlace cell (period/cols_per_period) is held at or above the floor by
    # barrier_lattice's cap. The old 8 µm / 0.1 corner advertised a 0.8 µm slit
    # on a 2 µm process.
    params = [
        ParamSpec("slit_period_um", "Slit period", "float", 60.0, 20.0, 200.0, 0.5, "μm"),
        ParamSpec("slit_duty", "Slit duty", "float", 0.4, 0.2, 0.8, 0.05),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------
    # Pure arithmetic on the params, reusing the same ``barrier_lattice`` solve
    # ``generate`` runs, so the plate compositor can read the manifest numbers
    # without building the interlace.

    @classmethod
    def pixel_pitch_um(
        cls,
        slit_period_um: float = 60.0,
        slit_duty: float = 0.4,
        extent_um: float = 2000.0,
    ) -> float:
        return max(0.5, slit_period_um / 20)

    @classmethod
    def min_feature_um(
        cls,
        slit_period_um: float = 60.0,
        slit_duty: float = 0.4,
        extent_um: float = 2000.0,
    ) -> float:
        # BOTH layers, not just the front comb: the front's narrowest
        # slit-or-bar is p·min(duty, 1-duty), and the back channel masks chop
        # silhouette rows into runs as short as ONE cell, so cell_um is the
        # back minimum. The old value described the front alone and could
        # advertise 0.8 µm while the back really emitted 2.0 µm.
        _cols, cell_um, _n_grid, _xoff = barrier_lattice(
            slit_period_um, extent_um, min_rows=MIN_GRID_ROWS
        )
        return min(slit_period_um * min(slit_duty, 1 - slit_duty), cell_um)

    @classmethod
    def extra_metadata(
        cls,
        slit_period_um: float = 60.0,
        slit_duty: float = 0.4,
        extent_um: float = 2000.0,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        cols_per_period, cell_um, _n_grid, xoff = barrier_lattice(
            slit_period_um, extent_um, min_rows=MIN_GRID_ROWS
        )
        return (
            {
                # Honest because the registration is enforced: the open slit
                # straddles a channel boundary at every extent, so the switch
                # completes at a quarter-period shift and the clean first zone
                # ends at p/2.
                "switch_half_angle_deg": exterior_tilt_deg(slit_period_um / 4),
                "zone_half_angle_deg": exterior_tilt_deg(slit_period_um / 2),
                # Registration witnesses — interlace_period_um must equal
                # slit_period_um exactly, and barrier_phase_um is the solved
                # offset that lands a slit centre on a channel boundary.
                "interlace_period_um": cols_per_period * cell_um,
                "interlace_cell_um": cell_um,
                "barrier_phase_um": xoff,
                "view_a_pose": "wings-up",
                "view_b_pose": "wings-down",
            },
            {
                "slit_axis_deg": 0.0,
                "slit_period_um": slit_period_um,
            },
            ("view_a", "view_b"),
        )

    @classmethod
    def generate(
        cls,
        slit_period_um: float = 60.0,
        slit_duty: float = 0.4,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Interlace raster + front-comb phase — the whole registration, solved
        # once for all six barrier slugs in _helpers.barrier_lattice (why the
        # raster comes from the slit lattice rather than the extent, why n_grid
        # is floored, and why the phase is solved mod p all live in its
        # docstring).
        cols_per_period, cell_um, n_grid, xoff = barrier_lattice(
            slit_period_um, extent_um, min_rows=MIN_GRID_ROWS
        )
        half = cols_per_period // 2

        # Interlace rect-count estimate: raster_to_polygons emits one rect per
        # horizontal run, and the channel masks chop every silhouette row into
        # ~extent/period runs per channel. Worst case (silhouettes covering the
        # full grid) is n_grid rows × stripes-per-extent — gate before building.
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
        # slit". The solved xoff above is what puts each open slit ON a channel
        # boundary (straddle class) with channel A on its +x side — the whole
        # beat lives there.
        #
        # Grating built a period wider than the tile because linear_grating crops
        # to the extent it is handed — cropping first and translating after would
        # leave a bare-glass band xoff wide at the left edge.
        barrier = linear_grating(
            slit_period_um, 1 - slit_duty, (extent_um + 2 * slit_period_um, extent_um)
        )
        barrier = affinity.translate(barrier, xoff=xoff)
        front = ensure_multipolygon(crop(barrier, extent))

        # Rasterize the two poses of the SAME bird onto the slit-derived pixel
        # grid — body/head/beak/tail pixel-registered (colibri._draw_colibri only
        # swaps the wing set) — so each pose can be sliced into vertical strips
        # that interlace one to one with the slit lattice.
        pose_a = colibri.colibri_silhouette(extent, n_grid=n_grid, pose="up")
        pose_b = colibri.colibri_silhouette(extent, n_grid=n_grid, pose="down")

        # Each slit period gets one pose half: left half = wings-up (view_a),
        # right half = wings-down (view_b).
        cols = np.arange(n_grid)
        phase = (cols % cols_per_period) < half
        a_strips = pose_a & phase[None, :]
        b_strips = pose_b & (~phase[None, :])

        # Tilt-sign convention, PINNED by barrier_lattice's mod-p phase: every
        # slit centre lands on a B→A boundary, so channel B occupies the -x half
        # of each slit and channel A the +x half. A +x back shift (+tilt) samples
        # the B half, a -x shift the A half.
        #   Scene A = wings-up interlace    → -tilt
        #   Scene B = wings-down interlace  → +tilt
        view_a = raster_to_polygons(a_strips.astype(np.uint8), cell_um, extent)
        view_b = raster_to_polygons(b_strips.astype(np.uint8), cell_um, extent)

        # Back layer = BOTH interlace channels, per the barrier architecture
        # (all image content behind the slits). The channels occupy
        # complementary column sets (they can share edges but never overlap)
        # and everything downstream is fill-only, so plain concatenation is
        # safe — never feed this MultiPolygon to a GEOS boolean.
        back = MultiPolygon([*view_a.geoms, *view_b.geoms])

        # Metadata comes from the accessors above so the numbers a plate reads
        # without generating are the ones a full generate publishes.
        kw = dict(
            slit_period_um=slit_period_um, slit_duty=slit_duty, extent_um=extent_um
        )
        extra, recipe_data, _layer_names = cls.extra_metadata(**kw)
        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=cls.pixel_pitch_um(**kw),
            min_feature_um=cls.min_feature_um(**kw),
            extra=extra,
            extra_layers={
                "view_a": ensure_multipolygon(view_a),
                "view_b": ensure_multipolygon(view_b),
            },
            recipe_data=recipe_data,
        )
