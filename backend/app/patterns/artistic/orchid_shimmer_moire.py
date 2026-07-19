from __future__ import annotations

import math

import numpy as np

from .._helpers import check_lattice_budget, linear_grating, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import orchid


@register
class OrchidShimmerMoire(Pattern):
    slug = "orchid-shimmer-moire"
    name = "Cattleya shimmer"
    description = (
        "A Cattleya trianae orchid — Colombia's national flower — carved "
        "into a fine vertical carrier grating on the front layer. The back "
        "layer is a full-field grating at a slightly detuned period, skewed "
        "a few degrees. Snell-refracted parallax slides the back carrier "
        "under the front, so inside the flower the two gratings beat into "
        "broad shimmer fringes that sweep across the petals as the plate "
        "tilts (Λ_beat ≈ p/detune), while the surround carries only the "
        "faint uniform back carrier."
    )
    tags = ["moire", "ambient", "tilt-reveal", "Colombia", "orchid"]
    tier = 1
    theme = "Colombia"
    render_recipe = "moire_interactive"
    params = [
        ParamSpec("period_um", "Carrier period", "float", 16.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("detune", "Back detune", "float", 0.06, 0.0, 0.25, 0.01),
        ParamSpec("skew_deg", "Back skew", "float", 4.0, 0.0, 15.0, 0.1, "°"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        period_um: float = 16.0,
        detune: float = 0.06,
        skew_deg: float = 4.0,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Resolve the carrier with ≥8 samples per period, capped at 1200 so
        # the raster grid itself stays cheap (1.4M bool cells max).
        n_grid = min(1200, max(384, int(extent_um / max(1.0, period_um / 8))))

        # Rect-count estimate for the front layer: raster_to_polygons emits
        # one rectangle per horizontal run, and the carrier chops every
        # silhouette row into ~extent/period runs. Worst case (silhouette
        # covering the full grid) that is n_grid rows × stripes-per-extent —
        # gate it before building anything.
        n_stripes = int(math.ceil(extent_um / period_um))
        check_lattice_budget(
            n_grid * n_stripes,
            "orchid-shimmer-moire front carrier",
            period_um=period_um,
            extent_um=extent_um,
        )

        cell_um = extent_um / n_grid
        flower = orchid.orchid_silhouette(extent, n_grid=n_grid)

        # Front: orchid ∩ carrier, intersected in RASTER space (silhouette
        # grid & per-column phase mask, exactly like the lenticular column
        # masks) — never a GEOS boolean.
        period_pix = period_um / cell_um
        cols = np.arange(n_grid, dtype=np.float32)
        phase = (cols % period_pix) < (period_pix / 2.0)
        front_mask = flower & phase[None, :]
        front = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)

        # Back: full-field carrier at the detuned period, skewed so the beat
        # fringes have both a spacing (detune) and an orientation (skew).
        back_period_um = period_um * (1.0 + detune)
        back = linear_grating(back_period_um, 0.5, extent, rotation_deg=skew_deg)

        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=period_um * 0.5,
            extra={
                "back_period_um": back_period_um,
                "expected_beat_period_um": period_um * (1.0 + detune)
                / max(1e-6, detune),
            },
        )
