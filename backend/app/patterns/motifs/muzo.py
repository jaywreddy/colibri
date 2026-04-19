from __future__ import annotations

import math

import numpy as np
from PIL import ImageDraw

from ._pillow import render_silhouette


def _draw_hex_crystal(draw: ImageDraw.ImageDraw, n: int) -> None:
    """Elongated hexagonal prism silhouette viewed from the side — Muzo emerald crystal."""
    s = n
    cx = 0.5 * s
    cy = 0.5 * s
    half_w = 0.22 * s
    half_h = 0.38 * s
    # Hexagonal prism seen end-on: a regular hexagon, slightly wider than tall
    pts = []
    for k in range(6):
        theta = math.radians(60 * k + 30)
        pts.append((cx + half_w * math.cos(theta), cy + half_h * math.sin(theta)))
    draw.polygon(pts, fill=255)
    # Central facet lines (darker strokes would be carve-outs — we use the outline only,
    # so just render the outer silhouette as solid)


def hex_crystal(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid — single hexagonal emerald crystal silhouette."""
    del extent_um
    return render_silhouette(_draw_hex_crystal, n_grid)
