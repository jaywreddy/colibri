from __future__ import annotations

import numpy as np

from .._helpers import raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..effects.gratings import clip_mask, linear_grating_mask
from ..motifs.lab.jamon import jamon_silhouette


def _resolve_grid(period_um: float, extent_um: float) -> tuple[int, float]:
    """``(n_grid, cell_um)`` for the carrier raster.

    ≥8 samples/period so the carrier's half-period phase offset is clean —
    identical grid rule to monogram-jp / food-pair-chirp.
    """
    n_grid = max(384, int(extent_um / max(1.0, period_um / 10)))
    return n_grid, extent_um / n_grid


def _carrier_grating(period_um: float, extent_um: float):
    """The uniform front carrier for these params.

    Cheap — one ≤400k-cell numpy mask, budget-guarded by ``linear_grating_mask``
    — and the ONLY place the EFFECTIVE (possibly coarsened) period comes from, so
    ``generate`` and the metadata accessors cannot report different periods.
    """
    _n_grid, cell_um = _resolve_grid(period_um, extent_um)
    return linear_grating_mask(
        period_um,
        (extent_um, extent_um),
        angle_deg=0.0,
        pitch_um=cell_um,
        coarsen=True,
    )


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
        ParamSpec("period_um", "Carrier period", "float", 20.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------

    @classmethod
    def pixel_pitch_um(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
    ) -> float:
        return _resolve_grid(period_um, extent_um)[1]

    @classmethod
    def min_feature_um(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
    ) -> float:
        return _carrier_grating(period_um, extent_um).period_um * 0.5

    @classmethod
    def extra_metadata(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        carrier_g = _carrier_grating(period_um, extent_um)
        # Informational only — kept out of recipe_data on purpose: the
        # Pattern Lab zone UI offers tilt quick-sets whenever recipe_data
        # carries a period key, and with an empty back layer there is no
        # mask-level tilt effect to quick-set to.
        return (
            {
                "carrier_period_um": carrier_g.period_um,
                "switch_axis_deg": 0.0,
                "coarsened": bool(carrier_g.coarsened),
            },
            {},
            (),
        )

    @classmethod
    def generate(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # The linear grating picks the pitch (budget-aware) and we build the
        # silhouette at the SAME pitch so the masks compose 1:1 with no
        # resampling.
        n_grid, cell_um = _resolve_grid(period_um, extent_um)

        carrier_g = _carrier_grating(period_um, extent_um)
        carrier = carrier_g.mask
        h_px, w_px = carrier.shape
        n_grid = w_px  # square extent → square grid

        jamon = jamon_silhouette(extent, n_grid=n_grid)

        front_mask = clip_mask(carrier, jamon)
        # Back is plain glass — front-only glimmer (matches monogram-jp).
        back_mask = np.zeros_like(front_mask)

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        # Metadata comes from the accessors above so the numbers a plate reads
        # without generating are the ones a full generate publishes.
        kw = dict(period_um=period_um, extent_um=extent_um)
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
