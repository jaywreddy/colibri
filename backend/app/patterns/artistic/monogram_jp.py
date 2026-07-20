from __future__ import annotations

import numpy as np

from .._helpers import raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import monogram


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

    @classmethod
    def generate(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
        overlap: float = 0.68,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # >=8 samples/period so the carrier's half-period phase offset is clean.
        n_grid = max(384, int(extent_um / max(1.0, period_um / 10)))
        cell_um = extent_um / n_grid

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

        return GeneratedPattern(
            front=front_poly,
            back=back_poly,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=period_um * 0.5,
            # Informational only — kept out of recipe_data on purpose: the
            # Pattern Lab zone UI offers tilt quick-sets whenever recipe_data
            # carries a period key, and with an empty back layer there is no
            # mask-level tilt effect to quick-set to.
            extra={"carrier_period_um": period_um, "switch_axis_deg": 0.0},
        )
