from __future__ import annotations

import numpy as np

from .._helpers import chevron_stripes, linear_grating, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import caravel


@register
class CaravelLatent(Pattern):
    slug = "caravel-latent"
    name = "Caravel latent image"
    description = (
        "Front is a nearly-opaque slit field. Back carries a caravel silhouette "
        "fused with a chevron carrier that evokes the bow-wave of a ship under "
        "sail. At normal incidence the slits don't align with the chevron arms "
        "and the hull stays hidden; at ~15° tilt, parallax slides the caravel "
        "under the slits and she sails into view."
    )
    tags = ["parallax", "tilt-reveal", "latent", "caravel"]
    tier = 1
    theme = "Global Travel"
    params = [
        ParamSpec("slit_period_um", "Slit period", "float", 40.0, 10.0, 200.0, 1.0, "μm"),
        ParamSpec("slit_width_um", "Slit width", "float", 4.0, 2.0, 20.0, 0.5, "μm"),
        ParamSpec("chevron_period_um", "Chevron period", "float", 40.0, 10.0, 200.0, 1.0, "μm"),
        ParamSpec("chevron_amp_um", "Chevron amplitude", "float", 20.0, 4.0, 100.0, 1.0, "μm"),
        ParamSpec("caravel_grid", "Caravel grid", "int", 192, 96, 384, 32),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        slit_period_um: float = 40.0,
        slit_width_um: float = 4.0,
        chevron_period_um: float = 40.0,
        chevron_amp_um: float = 20.0,
        caravel_grid: int = 192,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)
        duty = 1 - slit_width_um / slit_period_um
        front = linear_grating(slit_period_um, duty, extent)

        carrier = chevron_stripes(extent, chevron_period_um, chevron_amp_um)
        ship_mask_grid = caravel.caravel_silhouette(extent_um, n_grid=int(caravel_grid))
        cell = extent_um / ship_mask_grid.shape[0]
        ship = raster_to_polygons(ship_mask_grid.astype("uint8"), cell, extent)
        # The 'image' under the slits is the chevron carrier ∪ the caravel silhouette.
        from shapely.ops import unary_union
        from .._helpers import crop, ensure_multipolygon
        back = crop(ensure_multipolygon(unary_union([carrier, ship])), extent)

        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=max(0.25, slit_width_um / 4),
            min_feature_um=min(slit_width_um, chevron_amp_um),
            extra={
                "reveal_angle_deg": float(np.degrees(np.arctan(chevron_amp_um / 500.0))),
            },
        )
