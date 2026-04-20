from __future__ import annotations

from shapely import affinity
from shapely.ops import unary_union

from .._helpers import crop, ring_grating
from ..base import GeneratedPattern, ParamSpec, Pattern, ensure_multipolygon, register
from ..motifs import compass


@register
class CompassRoseSpiral(Pattern):
    slug = "compass-rose-spiral"
    name = "Compass-rose spiral moiré"
    description = (
        "A mariner's compass rose seeded into concentric ring gratings of "
        "slightly different radial periods, front and back. The two radial "
        "gratings beat against each other into a rotating pinwheel; the "
        "compass rays hint at cardinal bearings as the wheel turns."
    )
    tags = ["moire", "rotational", "ambient", "compass"]
    tier = 2
    theme = "Global Travel"
    # Radial-period mismatch between ring gratings beats into a pinwheel as
    # the viewer orbits — same parallax-shift trick as the other moiré
    # patterns, just with concentric rings instead of straight weaves.
    render_recipe = "moire_interactive"
    params = [
        ParamSpec("period_front_um", "Front radial period", "float", 30.0, 4.0, 100.0, 0.5, "μm"),
        ParamSpec("period_back_um", "Back radial period", "float", 32.0, 4.0, 100.0, 0.5, "μm"),
        ParamSpec("duty", "Duty cycle", "float", 0.5, 0.1, 0.9, 0.05),
        ParamSpec("rose_radius_um", "Compass rose radius", "float", 400.0, 100.0, 1500.0, 20.0, "μm"),
        ParamSpec("back_offset_um", "Back origin offset", "float", 0.0, 0.0, 500.0, 10.0, "μm"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        period_front_um: float = 30.0,
        period_back_um: float = 32.0,
        duty: float = 0.5,
        rose_radius_um: float = 400.0,
        back_offset_um: float = 0.0,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Front: rings ∪ compass rose (the rose rides on top of the ring grating).
        rings_front = ring_grating(period_front_um, duty, extent)
        rose = compass.compass_rose(extent, rose_radius_um)
        front = crop(
            ensure_multipolygon(unary_union([rings_front, rose])),
            extent,
        )

        # Back: same-construction ring grating with different period, optionally
        # translated — the radial-period mismatch gives the rotational moiré.
        back = ring_grating(period_back_um, duty, extent)
        if back_offset_um:
            back = ensure_multipolygon(
                affinity.translate(back, xoff=back_offset_um, yoff=0)
            )
            back = crop(back, extent)

        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=max(0.25, min(period_front_um, period_back_um) * duty / 8),
            min_feature_um=min(period_front_um, period_back_um) * duty,
        )
