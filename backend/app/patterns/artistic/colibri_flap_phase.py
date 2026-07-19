from __future__ import annotations

import numpy as np

from .._helpers import raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import colibri


@register
class ColibriFlapPhase(Pattern):
    slug = "colibri-flap-phase"
    name = "Colibrí wing-beat phase-shift"
    description = (
        "The SAME hummingbird in two wing poses, carved into one high-frequency "
        "stripe carrier with a half-period phase offset between them. Front "
        "holds pose A — wings swept up in the hover V (carrier phase 0). Back "
        "holds pose B — wings caught mid-downstroke, rotated ~60° about the "
        "shoulder and slightly extended, with a few speed slivers trailing the "
        "tips (carrier phase π, physically shifted by Λ/2). Body, head, beak and "
        "tail are pixel-registered between the two poses, so Snell-refracted "
        "parallax through the substrate biases which phase the eye samples and "
        "the wings appear to BEAT as the box rocks left↔right — the bird itself "
        "never changes, only its wings move. Head-on, the two poses interlace "
        "into a fine line pattern."
    )
    tags = ["moire", "phase-shift", "tilt-reveal", "animation", "Colombia"]
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

        # Two poses of the SAME bird — body/head/beak/tail pixel-registered.
        pose_a = colibri.colibri_silhouette(extent, n_grid=n_grid, pose="up")
        pose_b = colibri.colibri_silhouette(extent, n_grid=n_grid, pose="down")

        period_pix = period_um / cell_um
        cols = np.arange(n_grid, dtype=np.float32)
        # Front carrier: vertical stripes at phase 0.
        phase_front = (cols % period_pix) < (period_pix / 2.0)
        # Back carrier: same stripes shifted by half a period.
        phase_back = ((cols + period_pix / 2.0) % period_pix) < (period_pix / 2.0)

        front_mask = pose_a & phase_front[None, :]
        back_mask = pose_b & phase_back[None, :]

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
                "pose_front": "wings-up",
                "pose_back": "wings-down",
            },
            recipe_data={
                # Slit-normal axis for the parallax-shift projection in the
                # shader. Vertical stripes → horizontal switch axis (+X), so the
                # beat is triggered by rocking the box left↔right.
                "switch_axis_deg": 0.0,
                "carrier_period_um": period_um,
            },
        )
