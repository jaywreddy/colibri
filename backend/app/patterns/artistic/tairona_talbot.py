from __future__ import annotations

from shapely import affinity

from .._helpers import crop
from ..base import GeneratedPattern, ParamSpec, Pattern, ensure_multipolygon, register
from ..motifs.tairona import concentric_goldwork


@register
class TaironaTalbot(Pattern):
    slug = "tairona-talbot"
    name = "Tairona Talbot revival"
    description = (
        "Front and back are concentric ring gratings recalling Tairona coiled "
        "goldwork. With the 500 μm fused-silica gap between them set near the "
        "Talbot distance z_T = 2·Λ²·n/λ (valid paraxially far from the axis), "
        "the back sits on a self-image of the front — and a half-period "
        "lateral shift decenters the ring system to break that revival."
    )
    tags = ["talbot", "self-imaging", "diffraction", "Tairona"]
    tier = 3
    theme = "Colombia"
    render_recipe = "near_field_carpet"
    params = [
        ParamSpec("period_um", "Period", "float", 20.0, 8.0, 80.0, 0.5, "μm"),
        ParamSpec("duty", "Duty cycle", "float", 0.5, 0.1, 0.9, 0.05),
        ParamSpec("wavelength_um", "Design wavelength", "float", 0.55, 0.4, 0.8, 0.005, "μm"),
        ParamSpec("back_phase_shift", "Back phase shift", "choice", "zero",
                  choices=["zero", "half"]),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        period_um: float = 20.0,
        duty: float = 0.5,
        wavelength_um: float = 0.55,
        back_phase_shift: str = "zero",
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)
        front = concentric_goldwork(extent, period_um, duty)
        back = concentric_goldwork(extent, period_um, duty)
        if back_phase_shift == "half":
            back = ensure_multipolygon(affinity.translate(back, xoff=period_um / 2))
            back = crop(back, extent)

        n = 1.46
        z_T_um = 2 * period_um**2 * n / wavelength_um
        plate_over_z_T = 500.0 / z_T_um
        # Carpet sweeps from the back face through 2·z_T so the revival at z=z_T
        # sits at the midpoint of the z-slider (easy target for the vision check
        # and the E2E test).
        z_min_um = 0.0
        z_max_um = 2.0 * z_T_um
        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=max(0.25, period_um * duty / 8),
            min_feature_um=period_um * duty,
            extra={
                "talbot_distance_um": z_T_um,
                "plate_over_z_T": plate_over_z_T,
            },
            recipe_data={
                "talbot_distance_um": z_T_um,
                "design_wavelength_um": wavelength_um,
                "z_min_um": z_min_um,
                "z_max_um": z_max_um,
                "n_slices": 64,
            },
        )
