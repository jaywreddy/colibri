from __future__ import annotations

import math

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon

from .._helpers import crop, linear_grating, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, ensure_multipolygon, register
from ..motifs import colibri, globe


@register
class ColibriGlobeLenticular(Pattern):
    slug = "colibri-globe-lenticular"
    name = "Colibrí ↔ globe lenticular"
    description = (
        "A parallax-barrier (lenticular) Moiré tile. The front layer is a "
        "vertical slit grating; the back is two interlaced silhouettes — "
        "hummingbird stripes occupying the odd columns of the slit lattice, "
        "globe stripes occupying the even columns. Snell-refracted parallax "
        "through the substrate gates which interlace channel the eye sees, "
        "so tilting the plate left ↔ right hard-switches the visible image. "
        "Switch half-angle ≈ arctan(p/2·t)."
    )
    tags = ["moire", "lenticular", "tilt-reveal", "Colombia", "Global Travel"]
    tier = 1
    theme = "Colombia"
    render_recipe = "stereo_lenticular"
    params = [
        ParamSpec("slit_period_um", "Slit period", "float", 40.0, 8.0, 200.0, 0.5, "μm"),
        ParamSpec("slit_duty", "Slit duty", "float", 0.5, 0.1, 0.9, 0.05),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
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

        # Rasterize the two silhouettes onto a shared pixel grid so each
        # silhouette can be sliced into vertical strips that interlace one to
        # one with the slit lattice.
        n_grid = max(192, int(extent_um / max(2.0, slit_period_um / 8)))
        bird = colibri.colibri_silhouette(extent, n_grid=n_grid)
        gl = globe.globe_silhouette(extent, n_grid=n_grid)

        cell_um = extent_um / n_grid
        # Number of pixel columns per slit period. Each slit period gets one
        # "bird" half and one "globe" half. We split the period in two: left
        # half = bird, right half = globe.
        cols_per_period = max(2, int(round(slit_period_um / cell_um)))
        half = cols_per_period // 2

        cols = np.arange(n_grid)
        phase = (cols % cols_per_period) < half
        bird_strips = bird & phase[None, :]
        globe_strips = gl & (~phase[None, :])

        # Scene A = hummingbird interlace (the +tilt side sees this);
        # Scene B = globe interlace (the -tilt side sees this).
        view_a = raster_to_polygons(bird_strips.astype(np.uint8), cell_um, extent)
        view_b = raster_to_polygons(globe_strips.astype(np.uint8), cell_um, extent)

        # Legacy back layer combines the two interlace channels — kept so the
        # stylized fallback (in case a recipe binding drops to flat) still has
        # something visible behind the slits. The channels occupy complementary
        # column sets (they can share edges but never overlap) and everything
        # downstream is fill-only, so plain concatenation replaces the old
        # unary_union + no-op crop (the strips already live inside the extent).
        back = MultiPolygon([*view_a.geoms, *view_b.geoms])

        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=max(0.5, slit_period_um / 20),
            min_feature_um=slit_period_um * min(slit_duty, 1 - slit_duty),
            extra={
                "switch_half_angle_deg": float(
                    np.degrees(math.atan(slit_period_um / 2 / 500.0))
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
