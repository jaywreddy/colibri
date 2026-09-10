"""A face that is deliberately BARE GLASS.

The production box does not decorate every side. Its back and its bottom are
plain quartz: nothing is written on either ply, so the box reads through them
and the four decorated faces are what the eye lands on. That is a design
decision, not an omission, and it needs a slug so it can be SAID — a face with
no pattern would otherwise have to be expressed as "some pattern, but skipped
everywhere", which is exactly the sort of implicit special case that leaks.

So ``blank`` is a real registered pattern that generates two empty layers, and
the three writers that compose a face (``plates._raster_compose_plate``,
``plates.ensure_plate_svg`` and ``export_fine.build_plate_fine``) each test the
slug once and emit NO frame, NO carrier and NO centerpiece. The plate manifest
still exists — the box needs its cut dims, its glass and its assembly entry —
it just describes a rectangle of glass. ``recipe_data['blank']`` is True so the
preview shader can skip the face's gold passes rather than draw an empty mask.

``render_recipe`` is ``foliage_moire`` like every other composed face
(CLAUDE.md): the plate manifest forces that recipe regardless, and declaring
anything else here would only invite a reader to think a blank face renders
through some other path.
"""
from __future__ import annotations

from typing import Any

from ._helpers import empty_layer
from .base import LITHO_FLOOR_UM, GeneratedPattern, ParamSpec, Pattern, register


@register
class Blank(Pattern):
    slug = "blank"
    name = "Blank glass"
    description = (
        "No gold at all — a face of bare polished quartz. The production box "
        "leaves its back and bottom undecorated so the four written faces "
        "carry the piece; composed as a plate it emits no frame, no carrier "
        "and no centerpiece on either layer."
    )
    tags = ["blank", "glass", "production"]
    tier = 1
    theme = "Colombia"
    render_recipe = "foliage_moire"
    params = [
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    # --- cheap metadata (no geometry) — see Pattern.metadata ----------------

    @classmethod
    def pixel_pitch_um(cls, extent_um: float = 2000.0) -> float:
        """A nominal raster pitch, never actually rastered.

        The plate compositor takes ``max(central_pitch, longest_side / 1500)``,
        so anything comfortably fine leaves the plate canvas decided by the
        plate — which is what we want, since a blank face has no feature to
        resolve.
        """
        return extent_um / 1024.0

    @classmethod
    def min_feature_um(cls, extent_um: float = 2000.0) -> float:
        """The process floor, NOT ``inf``.

        There is no feature to measure, so ``float('inf')`` would be the honest
        answer — but this number is copied verbatim into the plate manifest,
        and ``json.dumps`` writes it as the non-standard token ``Infinity``,
        which ``JSON.parse`` in the browser refuses. The floor is the safe
        finite stand-in: no consumer can be misled by it (nothing is written),
        and ``check_litho_floor`` passes.
        """
        return LITHO_FLOOR_UM

    @classmethod
    def extra_metadata(
        cls, extent_um: float = 2000.0
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
        return ({"blank": True}, {"blank": True}, ())

    @classmethod
    def generate(cls, extent_um: float = 2000.0) -> GeneratedPattern:
        return GeneratedPattern(
            front=empty_layer(),
            back=empty_layer(),
            extent_um=(extent_um, extent_um),
            pixel_pitch_um=cls.pixel_pitch_um(extent_um=extent_um),
            min_feature_um=cls.min_feature_um(extent_um=extent_um),
            extra={"blank": True},
            recipe_data={"blank": True},
        )
