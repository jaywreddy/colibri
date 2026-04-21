from __future__ import annotations

import numpy as np

from .._helpers import empty_layer, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import colibri


@register
class ColibriHologram(Pattern):
    slug = "colibri-hologram"
    name = "Colibrí binary hologram"
    description = (
        "A Lohmann-style binary amplitude computer-generated hologram whose "
        "far-field reconstruction is a colibrí (hummingbird) silhouette. "
        "Random-phase seed + inverse FFT + sign threshold — a direct CGH on "
        "lithographed gold that sings under coherent illumination."
    )
    tags = ["hologram", "laser", "fft", "colibri"]
    tier = 2
    theme = "Colombia"
    render_recipe = "far_field_hologram"
    params = [
        ParamSpec("grid", "Hologram grid (cells)", "int", 256, 64, 512, 32),
        ParamSpec("cell_um", "Cell size", "float", 4.0, 2.0, 16.0, 0.5, "μm"),
        ParamSpec("seed", "Phase seed", "int", 42, 0, 999, 1),
    ]

    # Off-axis carrier shift in units of cells (pad/4 = N/4 in frequency
    # space); the Lohmann binary-amplitude reconstruction is
    # Hermitian-symmetric (target + conjugate co-exist), so a center-fed
    # target collides with DC. Shifting the target by (N/4, N/4) places
    # the reconstruction replica cleanly in the upper-right quadrant,
    # with DC at center and the conjugate in the lower-left.
    _CARRIER_CELLS = 4

    @classmethod
    def generate(
        cls,
        grid: int = 256,
        cell_um: float = 4.0,
        seed: int = 42,
    ) -> GeneratedPattern:
        N = int(grid)
        target = colibri.colibri_silhouette(extent_um=0.0, n_grid=N).astype(np.float32)
        # Off-axis carrier: shift target by (N/C, N/C) in Fourier space so
        # the reconstruction's first-order replica lands off-axis, cleanly
        # separated from the DC spike and from the Hermitian-symmetric
        # conjugate replica. C = _CARRIER_CELLS; default 4 -> quarter-image shift.
        iy, ix = np.indices((N, N))
        carrier = np.exp(
            1j * 2 * np.pi * (ix + iy) / cls._CARRIER_CELLS
        ).astype(np.complex64)
        rng = np.random.default_rng(int(seed))
        phase = np.exp(1j * 2 * np.pi * rng.random((N, N)))
        field = np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(target * carrier * phase))
        )
        binary = (field.real > 0).astype(np.uint8)

        extent_um = N * cell_um
        extent = (extent_um, extent_um)
        front = raster_to_polygons(binary, cell_um, extent)
        return GeneratedPattern(
            front=front,
            back=empty_layer(),
            extent_um=extent,
            pixel_pitch_um=max(0.5, cell_um / 4),
            min_feature_um=cell_um,
            extra={"grid": N, "cell_um": cell_um, "target": "colibri"},
            recipe_data={
                # R/G/B design wavelengths fed to /sim/farfield. The
                # reconstruction is spatially identical at all three in
                # this approximation — the chromatic smear comes from
                # the merge, not from geometric scaling — which still
                # gives a plausible "rainbow-edged" hummingbird under
                # white coherent illumination.
                "wavelengths_um": [0.65, 0.55, 0.45],
                "target": "colibri",
                # Carrier shift in units of image-width fractions: 4 means
                # the reconstruction is shifted by 1/4 of the field in both
                # axes, landing in the upper-right quadrant. /sim/farfield
                # uses this to crop to the correct quadrant.
                "carrier_cells": cls._CARRIER_CELLS,
            },
        )
