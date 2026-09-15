"""A face that is SOLID GOLD: a base plate.

The production box's bottom is an opaque gold floor (2026-09-15, Jay's ask):
the ring stands on gold and nothing shows through from below. That needs a
slug so it can be SAID — like ``blank`` says "bare glass" — and every writer
tests it once:

* ``export_fine.build_plate_fine`` emits ONE rectangle, the whole outer ply,
  on the front layer (no frame, no centrepiece, no rim: the gold runs under
  the foil, which is where a base plate's edge belongs).
* ``plates._raster_compose_plate`` paints the front mask ART_LEVEL everywhere
  (``recipe_data["art_solid"]`` makes the shader fill it as plain gold);
  ``plates._bake_plate_svg`` writes the same rectangle.
* On the CLEAR-polarity production plate (``witness_dies``) a solid die is
  simply a die with NO openings — the chrome stays — so it costs the mask
  nothing but its dicing ticks.

Single-ply like every other face: the inner ply is bare quartz.
"""
from __future__ import annotations

from typing import Any

from shapely.geometry import MultiPolygon, box

from ._helpers import empty_layer
from .base import LITHO_FLOOR_UM, GeneratedPattern, ParamSpec, Pattern, register


@register
class SolidGold(Pattern):
    slug = "solid-gold"
    name = "Solid gold (base plate)"
    description = (
        "One unbroken sheet of gold over the whole face — the box's base "
        "plate. No frame, no centrepiece, no clear rim: composed as a plate it "
        "emits a single rectangle on the outer ply and nothing on the inner."
    )
    tags = ["solid", "base", "production"]
    tier = 1
    theme = "Colombia"
    render_recipe = "foliage_moire"
    params = [
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def pixel_pitch_um(cls, extent_um: float = 2000.0) -> float:
        return extent_um / 1024.0

    @classmethod
    def min_feature_um(cls, extent_um: float = 2000.0) -> float:
        # The one feature is the face itself; the floor is the finite stand-in
        # the manifest can carry (see blank.Blank.min_feature_um).
        return LITHO_FLOOR_UM

    @classmethod
    def extra_metadata(
        cls, extent_um: float = 2000.0
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
        return ({"solid": True}, {"solid": True, "art_solid": True}, ())

    @classmethod
    def generate(cls, extent_um: float = 2000.0) -> GeneratedPattern:
        h = extent_um / 2.0
        return GeneratedPattern(
            front=MultiPolygon([box(-h, -h, h, h)]),
            back=empty_layer(),
            extent_um=(extent_um, extent_um),
            pixel_pitch_um=cls.pixel_pitch_um(extent_um=extent_um),
            min_feature_um=cls.min_feature_um(extent_um=extent_um),
            extra={"solid": True},
            recipe_data={"solid": True, "art_solid": True},
        )
