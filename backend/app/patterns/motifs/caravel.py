from __future__ import annotations

import numpy as np
from PIL import ImageDraw

from ._pillow import render_silhouette


def _draw_caravel(draw: ImageDraw.ImageDraw, n: int) -> None:
    """Stylized caravel silhouette — Columbus-era three-masted sailing ship."""
    s = n
    # Hull: a trapezoidal curve (shallow V)
    hull = [
        (0.12 * s, 0.62 * s),
        (0.88 * s, 0.62 * s),
        (0.78 * s, 0.78 * s),
        (0.22 * s, 0.78 * s),
    ]
    draw.polygon(hull, fill=255)
    # Upper deck / castle at stern
    castle = [
        (0.68 * s, 0.55 * s),
        (0.85 * s, 0.55 * s),
        (0.85 * s, 0.62 * s),
        (0.68 * s, 0.62 * s),
    ]
    draw.polygon(castle, fill=255)
    # Three masts (vertical bars)
    for mx in (0.32, 0.50, 0.68):
        draw.rectangle(
            (mx * s - 0.008 * s, 0.20 * s, mx * s + 0.008 * s, 0.62 * s),
            fill=255,
        )
    # Main square sail (largest, central mast)
    sail_main = [
        (0.38 * s, 0.25 * s),
        (0.62 * s, 0.25 * s),
        (0.60 * s, 0.55 * s),
        (0.40 * s, 0.55 * s),
    ]
    draw.polygon(sail_main, fill=255)
    # Fore sail (smaller, left mast)
    sail_fore = [
        (0.24 * s, 0.32 * s),
        (0.40 * s, 0.32 * s),
        (0.38 * s, 0.55 * s),
        (0.26 * s, 0.55 * s),
    ]
    draw.polygon(sail_fore, fill=255)
    # Mizzen sail (triangular, stern mast)
    sail_mizzen = [
        (0.62 * s, 0.35 * s),
        (0.76 * s, 0.35 * s),
        (0.68 * s, 0.55 * s),
    ]
    draw.polygon(sail_mizzen, fill=255)
    # Flag at top of main mast
    flag = [
        (0.50 * s, 0.18 * s),
        (0.58 * s, 0.20 * s),
        (0.50 * s, 0.22 * s),
    ]
    draw.polygon(flag, fill=255)


def caravel_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid — three-masted caravel sailing ship."""
    del extent_um
    return render_silhouette(_draw_caravel, n_grid)
