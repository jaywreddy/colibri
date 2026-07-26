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
from ..motifs import colibri, globe

# Silhouette quality floor: below ~192 rows the colibrí's beak and the globe's
# graticule break up. Handed to barrier_lattice, which trades it off against the
# litho floor.
MIN_GRID_ROWS = 192


@register
class ColibriGlobeLenticular(Pattern):
    slug = "colibri-globe-lenticular"
    name = "Colibrí ↔ globe lenticular"
    description = (
        "A parallax-barrier (lenticular) Moiré tile. The front layer is a "
        "vertical slit grating; the back is two interlaced silhouettes — "
        "hummingbird stripes occupying the odd columns of the slit lattice, "
        "globe stripes occupying the even columns. Snell-refracted parallax "
        "through the substrate gates which interlace channel the eye sees, "
        "so tilting the plate left ↔ right hard-switches the visible image. "
        "The switch completes at a quarter-period parallax shift "
        "(Snell-corrected exterior angle in extra.switch_half_angle_deg)."
    )
    tags = ["moire", "lenticular", "tilt-reveal", "Colombia", "Global Travel"]
    tier = 1
    theme = "Colombia"
    render_recipe = "stereo_lenticular"
    # Bounds are set by the litho floor, not by taste: the front comb's
    # narrowest feature is period·min(duty, 1-duty) and the back interlace cell
    # is period/cols_per_period (≤ period/8 for any extent in range), so
    # period ≥ 20 µm with duty in [0.2, 0.8] keeps every emitted feature — line
    # AND gap, on BOTH layers — at or above the litho floor. The old 8 µm / 0.1
    # corner advertised a 0.8 µm slit on a 2 µm process.
    params = [
        ParamSpec("slit_period_um", "Slit period", "float", 40.0, 20.0, 200.0, 0.5, "μm"),
        ParamSpec("slit_duty", "Slit duty", "float", 0.5, 0.2, 0.8, 0.05),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------
    # Pure arithmetic on the params, reusing the same ``barrier_lattice`` solve
    # ``generate`` runs, so the plate compositor can read the manifest numbers
    # without building the interlace.

    @classmethod
    def pixel_pitch_um(
        cls,
        slit_period_um: float = 40.0,
        slit_duty: float = 0.5,
        extent_um: float = 2000.0,
    ) -> float:
        return max(0.5, slit_period_um / 20)

    @classmethod
    def min_feature_um(
        cls,
        slit_period_um: float = 40.0,
        slit_duty: float = 0.5,
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
        slit_period_um: float = 40.0,
        slit_duty: float = 0.5,
        extent_um: float = 2000.0,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        cols_per_period, cell_um, _n_grid, xoff = barrier_lattice(
            slit_period_um, extent_um, min_rows=MIN_GRID_ROWS
        )
        return (
            {
                # Exterior tilt where the image switch completes. Honest because
                # the registration is enforced: the slit straddles a channel
                # boundary at every extent, so the switch peaks at a back shift
                # of p/4 and the clean first zone ends at p/2. Shared Snell
                # helper (audited 2026-07-19 — the old atan(p/2t) overstated it
                # 4x and omitted the index factor).
                "switch_half_angle_deg": exterior_tilt_deg(slit_period_um / 4),
                "zone_half_angle_deg": exterior_tilt_deg(slit_period_um / 2),
                # Registration witnesses — interlace_period_um must equal
                # slit_period_um exactly, and barrier_phase_um is the solved
                # offset that lands a slit centre on a channel boundary.
                "interlace_period_um": cols_per_period * cell_um,
                "interlace_cell_um": cell_um,
                "barrier_phase_um": xoff,
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
        slit_period_um: float = 40.0,
        slit_duty: float = 0.5,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Interlace raster + front-comb phase — the whole registration, solved
        # once for all six barrier slugs in _helpers.barrier_lattice (why the
        # raster comes from the slit lattice rather than the extent, why n_grid
        # is floored, and why the phase is solved mod p all live in its
        # docstring). This slug's p=40/extent=2000 default is where the audited
        # 0.819/0.000 separation was measured — back then only because that one
        # combination came out accidentally exact.
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
            "colibri-globe-lenticular interlace",
            slit_period_um=slit_period_um,
            extent_um=extent_um,
        )

        # Front: vertical slit barrier. 1 - slit_duty so the "duty" parameter
        # reads naturally as "fraction of the surface that is open slit". The
        # solved xoff above is what puts each open slit ON a channel boundary
        # (straddle class) with channel A (bird) on its +x side — the whole
        # effect lives there.
        #
        # Grating built a period wider than the tile on each side because
        # linear_grating crops to the extent it is handed — cropping first and
        # translating after would leave a bare-glass band xoff wide at one edge.
        barrier = linear_grating(
            slit_period_um, 1 - slit_duty, (extent_um + 2 * slit_period_um, extent_um)
        )
        barrier = affinity.translate(barrier, xoff=xoff)
        front = ensure_multipolygon(crop(barrier, extent))

        # Rasterize the two silhouettes onto the slit-derived pixel grid so each
        # silhouette can be sliced into vertical strips that interlace one to
        # one with the slit lattice.
        bird = colibri.colibri_silhouette(extent, n_grid=n_grid)
        gl = globe.globe_silhouette(extent, n_grid=n_grid)

        # Each slit period gets one "bird" half and one "globe" half: left
        # half = bird, right half = globe. cols_per_period is even by
        # construction, so the two channels are exactly p/2 wide.
        cols = np.arange(n_grid)
        phase = (cols % cols_per_period) < half
        bird_strips = bird & phase[None, :]
        globe_strips = gl & (~phase[None, :])

        # Tilt-sign convention, PINNED by barrier_lattice's mod-p phase: every
        # slit centre lands on a B→A boundary, so channel B occupies the -x half
        # of each slit and channel A the +x half. A +x back shift (+tilt) samples
        # the B half, a -x shift the A half. (The pre-convergence comment here
        # had the two signs the wrong way round.)
        #   Scene A = hummingbird interlace → -tilt
        #   Scene B = globe interlace       → +tilt
        view_a = raster_to_polygons(bird_strips.astype(np.uint8), cell_um, extent)
        view_b = raster_to_polygons(globe_strips.astype(np.uint8), cell_um, extent)

        # Legacy back layer combines the two interlace channels — kept so the
        # stylized fallback (in case a recipe binding drops to flat) still has
        # something visible behind the slits. The channels occupy complementary
        # column sets (they can share edges but never overlap) and everything
        # downstream is fill-only, so plain concatenation replaces the old
        # unary_union + no-op crop (the strips already live inside the extent).
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
