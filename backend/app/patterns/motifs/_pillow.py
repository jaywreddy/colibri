from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image, ImageDraw


def render_silhouette(
    draw_fn: Callable[[ImageDraw.ImageDraw, int], None],
    n_grid: int,
    threshold: int = 127,
) -> np.ndarray:
    """Rasterize a silhouette via Pillow, return a bool ndarray of shape (n_grid, n_grid).

    `draw_fn(draw, size)` paints on an 'L' image at full brightness (255 = silhouette,
    0 = background). The returned array has True where the silhouette covers.
    """
    img = Image.new("L", (n_grid, n_grid), 0)
    draw = ImageDraw.Draw(img)
    draw_fn(draw, n_grid)
    arr = np.asarray(img, dtype=np.uint8)
    return arr > threshold
