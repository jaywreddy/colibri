from __future__ import annotations

import math

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
    # Hex-lattice pitch mismatch produces a floating "ghost gem" via moiré —
    # the parallax-shifted back sample makes the beat walk with the camera.
    render_recipe = "moire_interactive"
    # Pitch and facet_duty both feed the reported min_feature_um (see generate),
    # which GeneratedPattern checks against the 2 µm litho floor, so the bounds
    # keep every legal combination printable: the binding corner is 12 µm × 0.85
    # → a 3.2 µm trench between facets, and the small-facet corner is
    # 12 µm × 0.4 → a 4.2 µm facet. The old 6 µm × 0.3 corner advertised 0.9 µm
    # facets and the old 0.95 ceiling left a 0.26 µm trench (near-touching facets
    # have no ghost-gem beat left anyway).
    params = [
        ParamSpec("period_front_um", "Front facet pitch", "float", 30.0, 12.0, 100.0, 0.5, "μm"),
        ParamSpec("period_back_um", "Back facet pitch", "float", 31.0, 12.0, 100.0, 0.5, "μm"),
        ParamSpec("facet_duty", "Facet size", "float", 0.7, 0.4, 0.85, 0.05),
        ParamSpec("rotation_deg", "Back rotation", "float", 0.0, -10.0, 10.0, 0.1, "°"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------

    @classmethod
    def pixel_pitch_um(
        cls,
        period_front_um: float = 30.0,
        period_back_um: float = 31.0,
        facet_duty: float = 0.7,
        rotation_deg: float = 0.0,
        extent_um: float = 2000.0,
    ) -> float:
        return max(0.25, min(period_front_um, period_back_um) / 16)

    @classmethod
    def min_feature_um(
        cls,
        period_front_um: float = 30.0,
        period_back_um: float = 31.0,
        facet_duty: float = 0.7,
        rotation_deg: float = 0.0,
        extent_um: float = 2000.0,
    ) -> float:
        # Narrower of the GOLD facet and the TRENCH between facets — both matter
        # on a 2 µm line / 2 µm gap process. hex_facets builds hexagons of
        # circumradius r = pitch·duty/2, so a facet's flat-to-flat width is
        # r·√3 = pitch·duty·√3/2 and its six nearest neighbours (all one pitch
        # away) leave pitch·(1 − duty·√3/2). The old report was the circumradius
        # alone: conservative about the gold, blind to the trench.
        s = math.sqrt(3) / 2
        pitch_min = min(period_front_um, period_back_um)
        return pitch_min * min(facet_duty * s, 1.0 - facet_duty * s)

    @classmethod
    def extra_metadata(
        cls,
        period_front_um: float = 30.0,
        period_back_um: float = 31.0,
        facet_duty: float = 0.7,
        rotation_deg: float = 0.0,
        extent_um: float = 2000.0,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        dp = period_front_um - period_back_um
        mag = period_front_um / dp if dp != 0 else float("inf")
        return ({"expected_magnification": mag}, {}, ())

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
        # Metadata comes from the accessors above so the numbers a plate reads
        # without generating are the ones a full generate publishes.
        kw = dict(
            period_front_um=period_front_um,
            period_back_um=period_back_um,
            facet_duty=facet_duty,
            rotation_deg=rotation_deg,
            extent_um=extent_um,
        )
        extra, recipe_data, _layer_names = cls.extra_metadata(**kw)
        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=cls.pixel_pitch_um(**kw),
            min_feature_um=cls.min_feature_um(**kw),
            extra=extra,
            recipe_data=recipe_data,
        )
