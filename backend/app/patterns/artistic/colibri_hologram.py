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
    params = [
        ParamSpec("grid", "Hologram grid (cells)", "int", 256, 64, 512, 32),
        ParamSpec("cell_um", "Cell size", "float", 4.0, 2.0, 16.0, 0.5, "μm"),
        ParamSpec("seed", "Phase seed", "int", 42, 0, 999, 1),
    ]

    @classmethod
    def generate(
        cls,
        grid: int = 256,
        cell_um: float = 4.0,
        seed: int = 42,
    ) -> GeneratedPattern:
        N = int(grid)
        target = colibri.colibri_silhouette(extent_um=0.0, n_grid=N).astype(np.float32)
        rng = np.random.default_rng(int(seed))
        phase = np.exp(1j * 2 * np.pi * rng.random((N, N)))
        field = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(target * phase)))
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
        )
