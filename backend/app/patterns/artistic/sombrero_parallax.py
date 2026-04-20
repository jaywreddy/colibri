from __future__ import annotations

import math

import numpy as np
from shapely import affinity
from shapely.geometry import box
from shapely.ops import unary_union

from .._helpers import crop, linear_grating
from ..base import GeneratedPattern, ParamSpec, Pattern, ensure_multipolygon, register
from ..motifs import cana_flecha


@register
class SombreroVueltiaoParallax(Pattern):
    slug = "sombrero-vueltiao-parallax"
    name = "Sombrero Vueltiao parallax"
    description = (
        "A caña flecha 'pinta' barrier on the front: vertical slits punched "
        "through the concentric bands of a Sombrero Vueltiao brim. The back "
        "carries two interlaced scenes. Tilting the plate left/right swaps "
        "which scene peeks through the slits — parallax at ±arctan(p/2·t)."
    )
    tags = ["parallax", "tilt-reveal", "Sombrero Vueltiao"]
    tier = 1
    theme = "Colombia"
    # Front slit barrier + two separate scene layers (view_a / view_b). The
    # shader picks which scene to show based on the sign of the tangent-space
    # view-vector component along the slit axis, so tilting the plate actually
    # swaps scenes — the parallax that used to be discarded now lives here.
    render_recipe = "stereo_lenticular"
    params = [
        ParamSpec("period_um", "Slit period", "float", 40.0, 8.0, 200.0, 0.5, "μm"),
        ParamSpec("slit_duty", "Slit duty", "float", 0.5, 0.1, 0.9, 0.05),
        ParamSpec("brim_band_um", "Brim band period", "float", 200.0, 60.0, 600.0, 10.0, "μm"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        period_um: float = 40.0,
        slit_duty: float = 0.5,
        brim_band_um: float = 200.0,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Front: caña flecha concentric brim bands ∪ vertical barrier (slits).
        # We mask the barrier so slits appear only inside the brim bands.
        barrier = linear_grating(period_um, 1 - slit_duty, extent)
        barrier = affinity.translate(barrier, xoff=period_um / 2)
        barrier = ensure_multipolygon(crop(barrier, extent))
        brim = cana_flecha.concentric_bands(extent, brim_band_um, duty=0.4)
        front = ensure_multipolygon(unary_union([barrier, brim]))
        front = crop(front, extent)

        # Scene A: caña flecha concentric bands — the hat silhouette.
        view_a = cana_flecha.concentric_bands(extent, brim_band_um, duty=0.55)

        # Scene B: triangular pinta tooth ring — the hat's woven brim motif.
        view_b = cana_flecha.pinta_triangles(
            extent,
            brim_band_um,
            triangle_size_um=brim_band_um * 0.4,
        )

        # Legacy back = classic interlace of the two scenes, kept so the
        # stylized_amplitude fallback still has something to show if the
        # shader drops back to it.
        w = period_um / 2
        strips_a = []
        strips_b = []
        n = int(extent_um / period_um) + 2
        for i in range(-n, n + 1):
            cx = i * period_um
            strips_a.append(box(cx - period_um / 2, -extent_um / 2, cx, extent_um / 2))
            for yi in range(-n, n + 1):
                cy = yi * 80.0
                strips_b.append(box(cx, cy - 20, cx + period_um / 2, cy + 20))
        back = crop(unary_union(strips_a + strips_b), extent)

        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=max(0.5, period_um / 20),
            min_feature_um=period_um * min(slit_duty, 1 - slit_duty),
            extra={
                "half_angle_deg_view_switch": float(
                    np.degrees(math.atan(period_um / 2 / 500.0))
                )
            },
            extra_layers={
                "view_a": ensure_multipolygon(view_a),
                "view_b": ensure_multipolygon(view_b),
            },
            recipe_data={
                # Slit axis is +X (vertical slits → horizontal parallax).
                "slit_axis_deg": 0.0,
                "slit_period_um": period_um,
            },
        )
