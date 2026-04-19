from __future__ import annotations

from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import emerald


@register
class EmeraldFacetMoire(Pattern):
    slug = "emerald-facet-moire"
    name = "Muzo emerald facet moiré"
    description = (
        "Hexagonal close-packed facet lattice evoking Muzo emerald crystal faces "
        "on both front and back, with a small mismatch in pitch. The moiré beat "
        "produces a floating 'ghost gem' magnified above the plate with "
        "magnification Λ_front / (Λ_front − Λ_back)."
    )
    tags = ["moire", "parallax", "magnifier", "emerald"]
    tier = 1
    theme = "Colombia"
    params = [
        ParamSpec("period_front_um", "Front facet pitch", "float", 30.0, 6.0, 100.0, 0.5, "μm"),
        ParamSpec("period_back_um", "Back facet pitch", "float", 31.0, 6.0, 100.0, 0.5, "μm"),
        ParamSpec("facet_duty", "Facet size", "float", 0.7, 0.3, 0.95, 0.05),
        ParamSpec("rotation_deg", "Back rotation", "float", 0.0, -10.0, 10.0, 0.1, "°"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        period_front_um: float = 30.0,
        period_back_um: float = 31.0,
        facet_duty: float = 0.7,
        rotation_deg: float = 0.0,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)
        front = emerald.hex_facets(extent, period_front_um, facet_duty)
        back = emerald.hex_facets(extent, period_back_um, facet_duty,
                                  rotation_deg=rotation_deg)
        dp = period_front_um - period_back_um
        mag = period_front_um / dp if dp != 0 else float("inf")
        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=max(0.25, min(period_front_um, period_back_um) / 16),
            min_feature_um=min(period_front_um, period_back_um) * facet_duty / 2,
            extra={"expected_magnification": mag},
        )
