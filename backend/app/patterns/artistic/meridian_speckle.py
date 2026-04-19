from __future__ import annotations

import numpy as np
from shapely.ops import unary_union

from .._helpers import crop, empty_layer, ensure_multipolygon, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import meridian


@register
class MeridianSpeckle(Pattern):
    slug = "meridian-speckle"
    name = "Meridian speckle diffuser"
    description = (
        "A random binary speckle aperture riding inside a meridian/parallel "
        "globe grid. Coherent light gets flat-topped into a beam with soft "
        "speckle; broadband light sees a mild angle-preserving diffuser with "
        "the grid's great-circle scaffolding faintly outlined."
    )
    tags = ["diffuser", "laser", "random", "meridian"]
    tier = 2
    theme = "Global Travel"
    params = [
        ParamSpec("grid", "Speckle grid (cells)", "int", 200, 64, 400, 16),
        ParamSpec("cell_um", "Cell size", "float", 4.0, 2.0, 20.0, 0.5, "μm"),
        ParamSpec("fill", "Fill fraction", "float", 0.5, 0.1, 0.9, 0.05),
        ParamSpec("seed", "Random seed", "int", 1, 0, 999, 1),
        ParamSpec("n_meridians", "Meridians", "int", 12, 4, 36, 1),
        ParamSpec("n_parallels", "Parallels", "int", 6, 2, 18, 1),
        ParamSpec("grid_line_um", "Grid line width", "float", 4.0, 1.0, 20.0, 0.5, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        grid: int = 200,
        cell_um: float = 4.0,
        fill: float = 0.5,
        seed: int = 1,
        n_meridians: int = 12,
        n_parallels: int = 6,
        grid_line_um: float = 4.0,
    ) -> GeneratedPattern:
        rng = np.random.default_rng(int(seed))
        N = int(grid)
        binary = (rng.random((N, N)) < fill).astype(np.uint8)
        extent_um = N * cell_um
        extent = (extent_um, extent_um)
        speckle = raster_to_polygons(binary, cell_um, extent)
        grid_geom = meridian.meridian_grid(
            extent, n_meridians=n_meridians, n_parallels=n_parallels,
            line_width_um=grid_line_um,
        )
        front = crop(
            ensure_multipolygon(unary_union([speckle, grid_geom])),
            extent,
        )
        return GeneratedPattern(
            front=front,
            back=empty_layer(),
            extent_um=extent,
            pixel_pitch_um=max(0.5, cell_um / 4),
            min_feature_um=min(cell_um, grid_line_um),
            extra={"grid": N, "fill": fill},
        )
