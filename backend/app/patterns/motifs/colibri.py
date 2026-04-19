from __future__ import annotations

import numpy as np
from PIL import ImageDraw

from ._pillow import render_silhouette


def _draw_colibri(draw: ImageDraw.ImageDraw, n: int) -> None:
    """Stylized colibrí (hummingbird) silhouette: body ellipse + long beak + wing triangle + tail."""
    s = n
    cx = 0.5 * s
    cy = 0.55 * s
    # Body (ellipse)
    draw.ellipse(
        (cx - 0.18 * s, cy - 0.09 * s, cx + 0.18 * s, cy + 0.09 * s),
        fill=255,
    )
    # Head (small circle)
    head_cx = cx - 0.20 * s
    head_cy = cy - 0.04 * s
    draw.ellipse(
        (head_cx - 0.07 * s, head_cy - 0.07 * s,
         head_cx + 0.07 * s, head_cy + 0.07 * s),
        fill=255,
    )
    # Long curved beak (polygon approximating a thin triangle)
    beak = [
        (head_cx - 0.05 * s, head_cy - 0.01 * s),
        (head_cx - 0.28 * s, head_cy - 0.06 * s),
        (head_cx - 0.05 * s, head_cy + 0.01 * s),
    ]
    draw.polygon(beak, fill=255)
    # Upper wing (triangular, angled up-back)
    wing_upper = [
        (cx, cy - 0.02 * s),
        (cx + 0.25 * s, cy - 0.28 * s),
        (cx + 0.10 * s, cy - 0.03 * s),
    ]
    draw.polygon(wing_upper, fill=255)
    # Lower wing
    wing_lower = [
        (cx + 0.02 * s, cy + 0.03 * s),
        (cx + 0.22 * s, cy + 0.22 * s),
        (cx + 0.08 * s, cy + 0.04 * s),
    ]
    draw.polygon(wing_lower, fill=255)
    # Forked tail
    tail = [
        (cx + 0.15 * s, cy),
        (cx + 0.40 * s, cy - 0.04 * s),
        (cx + 0.36 * s, cy),
        (cx + 0.40 * s, cy + 0.04 * s),
    ]
    draw.polygon(tail, fill=255)
    # Eye dot (small, stays as gold)
    draw.ellipse(
        (head_cx - 0.02 * s, head_cy - 0.02 * s,
         head_cx + 0.02 * s, head_cy + 0.02 * s),
        fill=0,  # eye is a hole in silhouette
    )


def colibri_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — colibrí facing left with outspread wings."""
    del extent_um  # silhouette is scale-free; caller controls cell_um to hit extent
    return render_silhouette(_draw_colibri, n_grid)
