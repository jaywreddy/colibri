from __future__ import annotations

import math

from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import wayuu


@register
class WayuuKanasuMoire(Pattern):
    slug = "wayuu-kanasu-moire"
    name = "Wayuu kanasü moiré"
    description = (
        "Diamond-lattice kanasü weave borrowed from Wayuu mochila bags, with a "
        "small rotational mismatch between front and back layers. Held up to "
        "the light, the two weaves produce slow-drifting moiré fringes whose "
        "spacing is set by the lattice period and the rotation angle: "
        "d ≈ Λ / (2 sin(θ/2))."
    )
    tags = ["moire", "ambient", "tilt-reveal", "Wayuu"]
    tier = 1
    theme = "Colombia"
    # Shader samples front × back with a Snell-refracted parallax shift so
    # the rotation mismatch between layers becomes visible moiré fringes that
    # walk with the camera angle.
    render_recipe = "moire_interactive"
    # Both dials feed the reported min_feature_um (see generate), which
    # GeneratedPattern checks against the 2 µm litho floor, so the bounds are
    # chosen to keep every legal combination printable: the binding corner is
    # 14 µm × 0.6 → a 2.1 µm gap between threads, and the thin-thread corner is
    # 14 µm × 0.25 → a 3.5 µm thread. The old 4 µm × 0.1 corner advertised 0.4 µm
    # threads on a 2 µm process. The duty ceiling also keeps the lattice inside
    # the disjoint regime (diamonds overlap above 1/√2 ≈ 0.707, where the weave
    # closes into near-solid gold with sub-floor pinholes and the moiré beat is
    # gone anyway).
    params = [
        ParamSpec("period_um", "Weave period", "float", 20.0, 14.0, 200.0, 0.5, "μm"),
        ParamSpec("duty", "Thread thickness", "float", 0.5, 0.25, 0.6, 0.05),
        ParamSpec("rotation_deg", "Back rotation", "float", 2.0, 0.0, 20.0, 0.1, "°"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------

    @classmethod
    def pixel_pitch_um(
        cls,
        period_um: float = 20.0,
        duty: float = 0.5,
        rotation_deg: float = 2.0,
        extent_um: float = 2000.0,
    ) -> float:
        return max(1.0, period_um * duty / 4)

    @classmethod
    def min_feature_um(
        cls,
        period_um: float = 20.0,
        duty: float = 0.5,
        rotation_deg: float = 2.0,
        extent_um: float = 2000.0,
    ) -> float:
        # Narrower of the GOLD thread and the GAP between threads — both matter on
        # a 2 µm line / 2 µm gap process. A diamond is a 45°-rotated square of
        # side period·duty, so its flat-to-flat width is period·duty; its nearest
        # neighbour sits one period away in x, leaving period·(1 − duty·√2).
        # Reporting only the thread (as this did) advertised a printable line
        # while the trench between threads was sub-floor.
        return period_um * min(duty, 1.0 - duty * math.sqrt(2))

    @classmethod
    def extra_metadata(
        cls,
        period_um: float = 20.0,
        duty: float = 0.5,
        rotation_deg: float = 2.0,
        extent_um: float = 2000.0,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        return (
            {
                "expected_fringe_period_um": period_um / max(1e-6, abs(rotation_deg))
                * 57.2958,
            },
            {},
            (),
        )

    @classmethod
    def generate(
        cls,
        period_um: float = 20.0,
        duty: float = 0.5,
        rotation_deg: float = 2.0,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)
        front = wayuu.kanasu_diamonds(extent, period_um, duty, rotation_deg=0.0)
        back = wayuu.kanasu_diamonds(extent, period_um, duty, rotation_deg=rotation_deg)
        # Metadata comes from the accessors above so the numbers a plate reads
        # without generating are the ones a full generate publishes.
        kw = dict(
            period_um=period_um,
            duty=duty,
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
