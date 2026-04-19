from __future__ import annotations

import math

from shapely.ops import unary_union

from .._helpers import crop, empty_layer, linear_grating
from ..base import GeneratedPattern, ParamSpec, Pattern, ensure_multipolygon, register
from ..motifs import cafetero


@register
class CafeteroIridescence(Pattern):
    slug = "cafetero-iridescence"
    name = "Eje Cafetero iridescent terraces"
    description = (
        "A fine linear grating (period near the visible wavelength scale) "
        "masked into horizontal 'terrace' bands evoking the coffee-country "
        "hills of the Eje Cafetero. Only the terraces iridesce; under "
        "broadband light a rainbow fan rolls along each band. First-order "
        "angle is asin(λ/Λ)."
    )
    tags = ["diffraction", "ambient", "iridescence", "Eje Cafetero"]
    tier = 2
    theme = "Colombia"
    params = [
        ParamSpec("period_um", "Grating period", "float", 4.0, 2.0, 10.0, 0.1, "μm"),
        ParamSpec("duty", "Grating duty", "float", 0.5, 0.1, 0.9, 0.05),
        ParamSpec("terrace_height_um", "Terrace height", "float", 120.0, 40.0, 400.0, 10.0, "μm"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
        ParamSpec("layer", "Layer", "choice", "front", choices=["front", "back"]),
    ]

    @classmethod
    def generate(
        cls,
        period_um: float = 4.0,
        duty: float = 0.5,
        terrace_height_um: float = 120.0,
        extent_um: float = 2000.0,
        layer: str = "front",
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)
        grating = linear_grating(period_um, duty, extent)
        terraces = cafetero.terrace_envelope(extent, terrace_height_um, duty=0.55)
        # Intersect grating with terrace envelope — fine grating appears only on terraces.
        masked = grating.intersection(terraces)
        iridescent = ensure_multipolygon(masked)
        iridescent = crop(iridescent, extent)

        front = iridescent if layer == "front" else empty_layer()
        back = iridescent if layer == "back" else empty_layer()
        m1 = math.degrees(math.asin(min(1.0, 0.55 / period_um)))
        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=max(1.0, period_um * duty / 4),
            min_feature_um=period_um * duty,
            extra={"first_order_green_deg": m1},
        )
