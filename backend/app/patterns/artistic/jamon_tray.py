from __future__ import annotations

import math

import numpy as np

from .._helpers import check_lattice_budget, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..effects.gratings import beat_delta_um, shimmer_moire_layers
from ..motifs.lab.jamon import jamon_silhouette


def _resolve_grid(period_um: float, extent_um: float) -> tuple[int, float]:
    """``(n_grid, cell_um)`` for the carrier raster.

    ≥8 samples/period so the carrier's half-period phase offset is clean —
    identical grid rule to monogram-jp / food-pair-chirp.
    """
    n_grid = max(384, int(extent_um / max(1.0, period_um / 10)))
    return n_grid, extent_um / n_grid


@register
class JamonTray(Pattern):
    """Jamón ibérico on a jamonero — front-only gold-stripe glimmer.

    A cured ham leg clamped hoof-up on its wooden carving stand, with a couple
    of carved slices on the board — the Spanish table the couple shares. Built
    exactly like the J+P monogram and the coffee+arepa (``food-pair-chirp``): the
    whole scene is carved into a uniform fine gold STRIPE CARRIER on the FRONT
    face while the BACK face is left plain glass, so the piece *glimmers*
    front-only with a moiré highlight as it tilts (rather than switching to a
    second image). No steam here — the jamón is served cured, not hot — so unlike
    the coffee scene there is no chirped-carrier band; a single uniform carrier
    fills the entire silhouette.

    LITHO / FAB (2 µm process, 4 µm min period):
      * The carrier sits at ≥ ~6 µm period (default 20 µm) — pure shimmer, no
        colour; ``linear_grating_mask`` auto-coarsens against the 400k-lattice
        budget and holds the 4 µm floor.

    INTEGRATOR NOTE (plates.py / shader): this bakes a uniform front carrier into
    the front polygons and leaves the back empty — identical wiring to
    ``'monogram-jp'`` / ``'food-pair-chirp'``. When wiring the box FACE, add a
    ``'jamon-tray'`` branch to ``plates._centerpiece_masks`` returning the jamón
    silhouette as the front and an EMPTY back (front-only glimmer). Face defaults
    are the integrator's to set — this pattern only registers the design.
    """

    slug = "jamon-tray"
    name = "Jamón ibérico on a jamonero"
    description = (
        "A whole cured ham leg clamped hoof-up on its wooden carving stand, with "
        "a few sliced pieces fanned on the board. The scene is carved into a fine "
        "uniform gold stripe carrier on the front face so it shimmers with a "
        "moiré highlight as the piece tilts; the back face is left plain glass, "
        "so the jamón reads as a front-only glimmer rather than a two-image "
        "switch. A gold-on-quartz nod to the Spanish table."
    )
    tags = ["tilt-shimmer", "food", "Spain", "jamon", "front-only"]
    tier = 1
    theme = "Colombia"
    render_recipe = "moire_interactive"
    params = [
        ParamSpec("period_um", "Carrier period", "float", 22.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
        # Shading-moiré band spacing; see monogram_jp / shimmer_moire_layers.
        ParamSpec("beat_um", "Moiré band spacing", "float", 1635.0, 200.0, 8000.0, 5.0, "μm"),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------

    @classmethod
    def pixel_pitch_um(
        cls,
        period_um: float = 22.0,
        beat_um: float = 1635.0,
        extent_um: float = 2000.0,
    ) -> float:
        return _resolve_grid(period_um, extent_um)[1]

    @classmethod
    def min_feature_um(
        cls,
        period_um: float = 22.0,
        beat_um: float = 1635.0,
        extent_um: float = 2000.0,
    ) -> float:
        # Both gratings are 50% duty; the FINER of the two sets the limit.
        # NOT _carrier_grating's effective period: linear_grating_mask(coarsen=
        # True) snaps to its own 4-samples-per-period floor and returns 12.78 um
        # for ANY request at this extent, which would silently make the moire
        # delta 0.10 um instead of 0.30 — ten times finer than the other three
        # shimmer faces, and at the edge of what the writer can hold.
        return min(period_um, period_um + beat_delta_um(period_um, beat_um)) * 0.5

    @classmethod
    def extra_metadata(
        cls,
        period_um: float = 22.0,
        beat_um: float = 1635.0,
        extent_um: float = 2000.0,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        front_period = period_um + beat_delta_um(period_um, beat_um)
        return (
            {
                "carrier_period_um": period_um,
                "switch_axis_deg": 0.0,
                "beat_period_um": beat_um,
            },
            {
                "fab_back_period_um": period_um,
                "fab_front_period_um": front_period,
                "carrier_angle_deg": 0.0,
                "slit_axis_deg": 0.0,
                "grating_duty": 0.5,
                "beat_period_um": beat_um,
            },
            (),
        )

    @classmethod
    def generate(
        cls,
        period_um: float = 22.0,
        beat_um: float = 1635.0,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Same grid rule as the other three shimmer faces, and the REQUESTED
        # period — see min_feature_um for why _carrier_grating's effective
        # period must not drive the moire pair.
        n_grid, cell_um = _resolve_grid(period_um, extent_um)

        delta = beat_delta_um(period_um, beat_um)
        n_stripes = int(math.ceil(extent_um / min(period_um, period_um + delta)))
        check_lattice_budget(
            2 * n_grid * n_stripes,
            "jamon-tray carrier pair",
            period_um=period_um,
            beat_um=beat_um,
            extent_um=extent_um,
        )

        jamon = jamon_silhouette(extent, n_grid=n_grid)

        # SHADING MOIRE: the ham filled at a mismatched pitch over a full-field
        # back carrier. See monogram_jp for why the beat comes from pitch and
        # not from a crossing angle on a build with no backside alignment.
        front_mask, back_mask = shimmer_moire_layers(
            jamon,
            back_period_um=period_um,
            delta_um=delta,
            cell_um=cell_um,
        )

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        # Metadata comes from the accessors above so the numbers a plate reads
        # without generating are the ones a full generate publishes.
        kw = dict(period_um=period_um, extent_um=extent_um, beat_um=beat_um)
        extra, recipe_data, _layer_names = cls.extra_metadata(**kw)
        return GeneratedPattern(
            front=front_poly,
            back=back_poly,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=cls.min_feature_um(**kw),
            extra=extra,
            recipe_data=recipe_data,
        )
