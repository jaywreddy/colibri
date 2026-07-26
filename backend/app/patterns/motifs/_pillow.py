from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image, ImageDraw


# Hard ceiling on the cells ONE silhouette raster may allocate. Deliberately
# separate from ``_helpers.MAX_LATTICE_CELLS`` (400k): that cap prices GEOS
# POLYGON cells at ~4-5 kB each, while a silhouette cell costs ~3 B (PIL "L"
# buffer + uint8 view + bool result), so the raster cap can sit ~40x looser
# without approaching the polygon cap's commit. 16M cells (n_grid <= 4000)
# clears every n_grid reachable from the ParamSpec slider ranges (the widest is
# ~2500 — a 5000 um extent at a 16 um slit period) yet still refuses the
# unbounded-POST corner: extent_um=40000 at period_um=6 asks for n_grid=26666,
# a 711 MB image whose downstream np.indices() is ~11 GB on a 13.7 GB host.
MAX_SILHOUETTE_CELLS = 16_000_000


def check_silhouette_budget(n_grid: int, what: str, **dials: float) -> None:
    """Refuse silhouette rasters that would not fit in memory.

    Same contract as ``_helpers.check_lattice_budget`` — a ValueError the
    /patterns and /boxes routes surface verbatim as an HTTP 400 — measured
    against the raster cap instead of the polygon-lattice cap. Generators size
    ``n_grid`` from caller-supplied ``extent_um`` / period params that nothing
    validates against their ParamSpec bounds, so this is the last line of
    defense before the allocation.
    """
    n = int(n_grid)
    n_cells = n * n
    if n_cells <= MAX_SILHOUETTE_CELLS:
        return
    knobs = ", ".join(f"{k}={v:g}" for k, v in dials.items())
    raise ValueError(
        f"{what} would rasterize {n}x{n} = {n_cells:,} silhouette cells"
        f"{f' ({knobs})' if knobs else ''}; the cap is "
        f"{MAX_SILHOUETTE_CELLS:,}. Increase period_um or shrink extent_um — "
        "the plate compositor upscales the pattern raster to fill the aperture, "
        "so patterns never need multi-mm extents at micron periods."
    )


def render_silhouette(
    draw_fn: Callable[[ImageDraw.ImageDraw, int], None],
    n_grid: int,
    threshold: int = 127,
) -> np.ndarray:
    """Rasterize a silhouette via Pillow, return a bool ndarray of shape (n_grid, n_grid).

    `draw_fn(draw, size)` paints on an 'L' image at full brightness (255 = silhouette,
    0 = background). The returned array has True where the silhouette covers.

    Shared chokepoint for every hand-drawn motif (colibri, orchid, monogram,
    and all of ``lab/``), so the budget pre-gate here covers all of them.
    """
    check_silhouette_budget(n_grid, "Motif silhouette")
    img = Image.new("L", (n_grid, n_grid), 0)
    draw = ImageDraw.Draw(img)
    draw_fn(draw, n_grid)
    arr = np.asarray(img, dtype=np.uint8)
    return arr > threshold
