from __future__ import annotations

import numpy as np

from .._helpers import raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import colibri, globe


@register
class ColibriGlobePhase(Pattern):
    slug = "colibri-globe-phase"
    name = "Colibrí ↔ globe phase-shift overlay"
    description = (
        "Both silhouettes carved into a single high-frequency stripe carrier "
        "with a half-period phase offset between them. Front holds the "
        "hummingbird stripes (carrier phase 0); back holds the globe stripes "
        "(carrier phase π — physically shifted by Λ/2). Snell-refracted "
        "parallax through the substrate slides the back layer by a fraction "
        "of the stripe period; the sign of the shift biases which carrier "
        "phase the eye samples, so one image cleanly emerges with tilt while "
        "the other recedes. Head-on, the two interlace into a fine line "
        "pattern with both images half-visible at once."
    )
    tags = ["moire", "phase-shift", "tilt-reveal", "Colombia", "Global Travel"]
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
        # phase offset between front and back is geometrically clean.
        n_grid = max(384, int(extent_um / max(1.0, period_um / 10)))
        cell_um = extent_um / n_grid

        bird = colibri.colibri_silhouette(extent, n_grid=n_grid)
        gl = globe.globe_silhouette(extent, n_grid=n_grid)

        period_pix = period_um / cell_um
        cols = np.arange(n_grid, dtype=np.float32)
        # Front carrier: vertical stripes at phase 0 (gold on even-period half).
        phase_front = (cols % period_pix) < (period_pix / 2.0)
        # Back carrier: same stripes shifted by half a period.
        phase_back = ((cols + period_pix / 2.0) % period_pix) < (period_pix / 2.0)

        front_mask = bird & phase_front[None, :]
        back_mask = gl & phase_back[None, :]

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
            },
            recipe_data={
                # Slit-normal axis for the parallax-shift projection in the
                # shader. Vertical stripes → horizontal switch axis (+X).
                "switch_axis_deg": 0.0,
                "carrier_period_um": period_um,
            },
        )
