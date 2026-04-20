from __future__ import annotations

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
    params = [
        ParamSpec("period_um", "Weave period", "float", 20.0, 4.0, 200.0, 0.5, "μm"),
        ParamSpec("duty", "Thread thickness", "float", 0.5, 0.1, 0.9, 0.05),
        ParamSpec("rotation_deg", "Back rotation", "float", 2.0, 0.0, 20.0, 0.1, "°"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

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
        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=max(1.0, period_um * duty / 4),
            min_feature_um=period_um * duty,
            extra={
                "expected_fringe_period_um": period_um / max(1e-6, abs(rotation_deg))
                * 57.2958,
            },
        )
