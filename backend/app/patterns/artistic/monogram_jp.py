from __future__ import annotations

import math

import numpy as np

from .._helpers import check_lattice_budget, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import monogram


def _resolve_grid(period_um: float, extent_um: float) -> tuple[int, float]:
    """``(n_grid, cell_um)`` for the carrier raster.

    ≥8 samples/period so the carrier's half-period phase offset is clean.
    Module-level because both ``generate`` and ``pixel_pitch_um`` need it and the
    plate compositor reads the pitch WITHOUT generating — one expression, so they
    cannot disagree.
    """
    n_grid = max(384, int(extent_um / max(1.0, period_um / 10)))
    return n_grid, extent_um / n_grid


@register
class MonogramJP(Pattern):
    """Interlocked cursive J+P wedding monogram, front-only shimmer.

    The FRONT face carries the entwined J+P silhouette carved into a fine
    stripe carrier; the BACK face is empty (all glass), so inside a box the
    monogram *glimmers* against the plain back carrier as the piece tilts
    (front-only glimmer) rather than switching to a second figure. This is the
    lid centerpiece — a classic engagement engraving in gold on quartz.

    In a box the plate compositor dispatches the paired silhouettes through
    ``plates._centerpiece_masks`` (front = monogram, back = empty) and draws the
    grating procedurally in the shader; this standalone ``generate`` exists so
    the pattern registers in the catalog and the ``/patterns`` endpoint can
    preview it. It is a front-only stripe-carrier shimmer (moire_interactive):
    single-layer gold art whose motion, inside a box, comes from the
    plate-level foliage carrier behind it.
    """

    slug = "monogram-jp"
    name = "J + P monogram (engagement engraving)"
    description = (
        "An interlocked cursive J and P — the couple's initials entwined the "
        "way an engraver would set them on a signet or a wedding invitation "
        "(Great Vibes copperplate swashes). Carved into a fine gold stripe "
        "carrier so the letters shimmer with a moiré highlight as the lid tilts, "
        "the back face left empty (plain glass; inside a box the plate "
        "compositor puts the uniform back carrier behind it) so the monogram "
        "reads as a front-only glimmer rather than a two-image switch."
    )
    tags = ["monogram", "engagement", "cursive", "tilt-shimmer", "lid"]
    tier = 1
    theme = "Colombia"
    render_recipe = "moire_interactive"
    params = [
        ParamSpec("period_um", "Carrier period", "float", 20.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
        ParamSpec("overlap", "Glyph interlock", "float", 0.68, 0.4, 0.85, 0.01),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------

    @classmethod
    def pixel_pitch_um(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
        overlap: float = 0.68,
    ) -> float:
        return _resolve_grid(period_um, extent_um)[1]

    @classmethod
    def min_feature_um(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
        overlap: float = 0.68,
    ) -> float:
        return period_um * 0.5

    @classmethod
    def extra_metadata(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
        overlap: float = 0.68,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        # Informational only — kept out of recipe_data on purpose: the
        # Pattern Lab zone UI offers tilt quick-sets whenever recipe_data
        # carries a period key, and with an empty back layer there is no
        # mask-level tilt effect to quick-set to.
        return ({"carrier_period_um": period_um, "switch_axis_deg": 0.0}, {}, ())

    @classmethod
    def generate(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
        overlap: float = 0.68,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        n_grid, cell_um = _resolve_grid(period_um, extent_um)

        # Rect-count estimate: raster_to_polygons emits one rect per horizontal
        # run, and the carrier chops every glyph row into ~extent/period runs.
        # Worst case (silhouette covering the full grid) is n_grid rows ×
        # stripes-per-extent — gate before the silhouette raster.
        n_stripes = int(math.ceil(extent_um / period_um))
        check_lattice_budget(
            n_grid * n_stripes,
            "monogram-jp front carrier",
            period_um=period_um,
            extent_um=extent_um,
        )

        mono = monogram.monogram_silhouette(extent, n_grid=n_grid, overlap=overlap)

        period_pix = period_um / cell_um
        cols = np.arange(n_grid, dtype=np.float32)
        # Vertical stripe carrier at phase 0 on the front.
        phase_front = (cols % period_pix) < (period_pix / 2.0)

        front_mask = mono & phase_front[None, :]
        # Back is plain glass — the monogram glimmers front-only.
        back_mask = np.zeros_like(front_mask)

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        # Metadata comes from the accessors above so the numbers a plate reads
        # without generating are the ones a full generate publishes.
        kw = dict(period_um=period_um, extent_um=extent_um, overlap=overlap)
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
