from __future__ import annotations

from .._helpers import crop, empty_layer, raster_to_polygons, zone_plate
from ..base import GeneratedPattern, ParamSpec, Pattern, ensure_multipolygon, register
from ..motifs import muzo


@register
class MuzoEmeraldZone(Pattern):
    slug = "muzo-emerald-zone"
    name = "Muzo emerald zone plate"
    description = (
        "A binary Fresnel zone plate clipped inside the elongated hexagonal "
        "silhouette of a Muzo emerald crystal. Focuses collimated light to a "
        "bright spot at the design focal length, while the hex aperture shapes "
        "the beam beyond focus into a six-pointed gem-flare."
    )
    tags = ["diffraction", "focusing", "laser", "Muzo"]
    tier = 2
    theme = "Colombia"
    params = [
        ParamSpec("focal_length_um", "Focal length", "float", 10000.0, 1000.0, 50000.0, 100.0, "μm"),
        ParamSpec("wavelength_um", "Design wavelength", "float", 0.55, 0.3, 1.0, 0.005, "μm"),
        ParamSpec("crystal_grid", "Crystal mask grid", "int", 256, 128, 512, 32),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
        ParamSpec("layer", "Layer", "choice", "back", choices=["front", "back"]),
    ]

    @classmethod
    def generate(
        cls,
        focal_length_um: float = 10000.0,
        wavelength_um: float = 0.55,
        crystal_grid: int = 256,
        extent_um: float = 2000.0,
        layer: str = "back",
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)
        zp = zone_plate(focal_length_um, wavelength_um, extent, min_feature_um=2.0)
        mask_grid = muzo.hex_crystal(extent_um, n_grid=int(crystal_grid))
        cell = extent_um / mask_grid.shape[0]
        mask = raster_to_polygons(mask_grid.astype("uint8"), cell, extent)
        # Clip the zone plate to the crystal silhouette.
        clipped = zp.intersection(mask)
        clipped = ensure_multipolygon(clipped)
        clipped = crop(clipped, extent)

        front = clipped if layer == "front" else empty_layer()
        back = clipped if layer == "back" else empty_layer()
        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=0.5,
            min_feature_um=2.0,
            extra={
                "focal_length_um": focal_length_um,
                "design_wavelength_um": wavelength_um,
            },
        )
