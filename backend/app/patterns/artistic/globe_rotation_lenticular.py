from __future__ import annotations

import math

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon

from .._helpers import crop, linear_grating, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, ensure_multipolygon, register
from ..motifs import globe


@register
class GlobeRotationStereo(Pattern):
    slug = "globe-rotation-stereo"
    name = "Spinning globe"
    description = (
        "A parallax-barrier (lenticular) Moiré tile where both interlace "
        "channels hold the SAME globe at two spin angles: the -tilt channel "
        "sees the globe rotated -Δ/2 about its polar axis, the +tilt channel "
        "sees +Δ/2 — meridians foreshorten differently and the landmass "
        "slides across the disk. Snell-refracted parallax through the "
        "substrate gates which channel the eye sees, so rocking the plate "
        "left ↔ right spins the earth. Switch half-angle ≈ arctan(p/2·t)."
    )
    tags = ["moire", "lenticular", "tilt-reveal", "Global Travel"]
    tier = 1
    theme = "Global Travel"
    render_recipe = "stereo_lenticular"
    params = [
        ParamSpec(
            "rotation_delta_deg", "Spin between views", "float", 60.0, 10.0, 180.0, 5.0, "°"
        ),
        ParamSpec("slit_period_um", "Slit period", "float", 40.0, 8.0, 200.0, 0.5, "μm"),
        ParamSpec("slit_duty", "Slit duty", "float", 0.5, 0.1, 0.9, 0.05),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        rotation_delta_deg: float = 60.0,
        slit_period_um: float = 40.0,
        slit_duty: float = 0.5,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Front: vertical slit barrier. 1 - slit_duty so the "duty" parameter
        # reads naturally as "fraction of the surface that is open slit".
        barrier = linear_grating(slit_period_um, 1 - slit_duty, extent)
        barrier = affinity.translate(barrier, xoff=slit_period_um / 2)
        front = ensure_multipolygon(crop(barrier, extent))

        # Rasterize the two spin states onto a shared pixel grid so each can
        # be sliced into vertical strips that interlace one to one with the
        # slit lattice.
        n_grid = max(192, int(extent_um / max(2.0, slit_period_um / 8)))
        half_delta = rotation_delta_deg / 2.0
        globe_a = globe.globe_silhouette(extent, n_grid=n_grid, rotation_deg=-half_delta)
        globe_b = globe.globe_silhouette(extent, n_grid=n_grid, rotation_deg=+half_delta)

        cell_um = extent_um / n_grid
        # Number of pixel columns per slit period. Each slit period gets one
        # half per spin state: left half = -Δ/2 view, right half = +Δ/2 view.
        cols_per_period = max(2, int(round(slit_period_um / cell_um)))
        half = cols_per_period // 2

        cols = np.arange(n_grid)
        phase = (cols % cols_per_period) < half
        strips_a = globe_a & phase[None, :]
        strips_b = globe_b & (~phase[None, :])

        # Scene A = globe at -Δ/2 (the +tilt side sees this);
        # Scene B = globe at +Δ/2 (the -tilt side sees this).
        view_a = raster_to_polygons(strips_a.astype(np.uint8), cell_um, extent)
        view_b = raster_to_polygons(strips_b.astype(np.uint8), cell_um, extent)

        # Legacy back layer combines the two interlace channels — kept so the
        # stylized fallback (in case a recipe binding drops to flat) still has
        # something visible behind the slits. The channels occupy complementary
        # column sets (they can share edges but never overlap) and everything
        # downstream is fill-only, so plain concatenation is safe (the strips
        # already live inside the extent).
        back = MultiPolygon([*view_a.geoms, *view_b.geoms])

        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=max(0.5, slit_period_um / 20),
            min_feature_um=slit_period_um * min(slit_duty, 1 - slit_duty),
            extra={
                "rotation_delta_deg": rotation_delta_deg,
                # Switch completes at a back shift of p/4 (slit straddles the
                # channel boundary); Snell maps the in-substrate angle out.
                "switch_half_angle_deg": float(
                    np.degrees(
                        math.asin(
                            min(1.0, 1.46 * math.sin(math.atan(slit_period_um / 4 / 500.0)))
                        )
                    )
                ),
            },
            extra_layers={
                "view_a": ensure_multipolygon(view_a),
                "view_b": ensure_multipolygon(view_b),
            },
            recipe_data={
                "slit_axis_deg": 0.0,
                "slit_period_um": slit_period_um,
            },
        )
