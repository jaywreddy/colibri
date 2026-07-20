from __future__ import annotations

import math

import numpy as np

from .._helpers import raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import colibri, globe


def _stripes_at_angle(
    n_grid: int,
    period_pix: float,
    rotation_deg: float,
) -> np.ndarray:
    """Return a binary (n_grid × n_grid) array of opaque vertical-ish stripes
    of the given period, rotated by `rotation_deg` about the grid center.

    Pure-numpy: we project the (x, y) coordinate onto the rotated period axis
    and threshold by the duty fraction (50/50 stripe / gap).
    """
    ys, xs = np.indices((n_grid, n_grid)).astype(np.float32)
    cx = cy = n_grid / 2.0
    th = math.radians(rotation_deg)
    # Unit vector along the period axis (perpendicular to the stripe direction).
    ax = math.cos(th)
    ay = math.sin(th)
    proj = (xs - cx) * ax + (ys - cy) * ay
    phase = (proj / period_pix) % 1.0
    return phase < 0.5


@register
class ColibriGlobeMoire(Pattern):
    slug = "colibri-globe-moire"
    name = "Colibrí ↔ globe dual-grating moiré"
    description = (
        "A pure Moiré tile: hummingbird and globe silhouettes are each carved "
        "into a high-frequency line grating, but the two gratings differ in "
        "orientation and period. Stacked on the two faces of the substrate, "
        "they beat against each other. Both silhouettes are always visible — "
        "the front bird can never vanish under parallax (its mask does not "
        "move with tilt) — but parallax-shifted sampling through the quartz "
        "makes the BEAT PHASE walk with the camera, so a shimmer band sweeps "
        "across the pair as the plate rocks (per-region brightness "
        "modulation, not an image switch). Beat wavelength obeys "
        "1/d_beat ≈ |1/Λ_front − 1/Λ_back| for the period mismatch plus a "
        "rotational contribution from the angle mismatch."
    )
    tags = ["moire", "tilt-shimmer", "Colombia", "Global Travel"]
    tier = 1
    theme = "Colombia"
    render_recipe = "moire_interactive"
    params = [
        ParamSpec("period_um", "Grating period", "float", 25.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("period_mismatch_um", "Period mismatch", "float", 0.6, 0.0, 5.0, 0.05, "μm"),
        ParamSpec("rotation_deg", "Back rotation", "float", 1.5, 0.0, 10.0, 0.1, "°"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        period_um: float = 25.0,
        period_mismatch_um: float = 0.6,
        rotation_deg: float = 1.5,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Pixel grid sized so we resolve the carrier grating with several
        # samples per period — moire beating is a sub-period effect, so we
        # need fine resolution even though the carrier itself is ~25 μm.
        n_grid = max(384, int(extent_um / max(1.5, period_um / 8)))
        cell_um = extent_um / n_grid

        bird = colibri.colibri_silhouette(extent, n_grid=n_grid)
        gl = globe.globe_silhouette(extent, n_grid=n_grid)

        period_front_pix = period_um / cell_um
        period_back_pix = (period_um + period_mismatch_um) / cell_um

        front_grating = _stripes_at_angle(n_grid, period_front_pix, rotation_deg=0.0)
        back_grating = _stripes_at_angle(n_grid, period_back_pix, rotation_deg=rotation_deg)

        # AND the carrier grating with the silhouette amplitude mask: gold
        # lives where the silhouette covers the grating stripe, nowhere else.
        front_mask = bird & front_grating
        back_mask = gl & back_grating

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        # Expected beat period from the period mismatch (the rotational
        # contribution is small for the default ~1.5°). Stash it so the
        # manifest's `extra` block reports something the optics-aware user
        # can sanity-check the rendering against.
        if period_mismatch_um > 1e-6:
            beat_period_um = period_um * (period_um + period_mismatch_um) / period_mismatch_um
        else:
            beat_period_um = float("inf")

        return GeneratedPattern(
            front=front_poly,
            back=back_poly,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=period_um * 0.5,
            extra={
                "beat_period_um_from_pitch_mismatch": beat_period_um,
                "rotation_beat_period_um": (
                    period_um / math.radians(rotation_deg) if rotation_deg else float("inf")
                ),
            },
        )
