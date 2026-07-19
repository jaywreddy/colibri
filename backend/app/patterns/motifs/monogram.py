"""J+P monogram and heart-with-globe keepsake silhouettes.

Extracted from the retired phase-overlay generator so the barrier switch
(`artistic/monogram_phase.py`) and the carrier reveal
(`artistic/carrier_reveal.py`) can share the same deterministic rasters.
"""
from __future__ import annotations

import numpy as np
from PIL import ImageDraw

from ._pillow import render_silhouette


def _draw_jp_monogram(draw: ImageDraw.ImageDraw, n: int) -> None:
    """Bold blocky "J + P" monogram silhouette.

    Glyphs are built from plain rectangles rather than ImageDraw.text — the
    default PIL bitmap font is tiny and its metrics vary across Pillow
    builds, so hand-built strokes keep the raster deterministic. Layout:
    three glyph cells across the middle band, stroke ≈ 0.07·s.
    """
    s = n
    t = 0.07 * s  # stroke thickness
    y0 = 0.28 * s  # glyph band top
    y1 = 0.72 * s  # glyph band bottom

    # --- J (cell x ∈ [0.14, 0.36]) ----------------------------------------
    jx0, jx1 = 0.14 * s, 0.36 * s
    draw.rectangle((jx0, y0, jx1, y0 + t), fill=255)            # top bar
    draw.rectangle((jx1 - t, y0, jx1, y1 - t), fill=255)        # right stem
    draw.rectangle((jx0, y1 - t, jx1, y1), fill=255)            # bottom bar
    draw.rectangle((jx0, y1 - 2.2 * t, jx0 + t, y1), fill=255)  # left up-hook

    # --- + (cell x ∈ [0.41, 0.59]) ----------------------------------------
    pcx = 0.50 * s
    pcy = 0.50 * s
    arm = 0.09 * s
    draw.rectangle((pcx - t / 2, pcy - arm, pcx + t / 2, pcy + arm), fill=255)
    draw.rectangle((pcx - arm, pcy - t / 2, pcx + arm, pcy + t / 2), fill=255)

    # --- P (cell x ∈ [0.64, 0.86]) ----------------------------------------
    px0, px1 = 0.64 * s, 0.86 * s
    bowl_bottom = 0.52 * s
    draw.rectangle((px0, y0, px0 + t, y1), fill=255)                    # stem
    draw.rectangle((px0, y0, px1, y0 + t), fill=255)                    # bowl top
    draw.rectangle((px0, bowl_bottom - t, px1, bowl_bottom), fill=255)  # bowl bottom
    draw.rectangle((px1 - t, y0, px1, bowl_bottom), fill=255)           # bowl right


def _draw_heart_globe(draw: ImageDraw.ImageDraw, n: int) -> None:
    """Heart outline with a small wireframe globe inside.

    The heart is two lobe disks + a point triangle, filled, then a scaled-down
    inner heart is punched out (fill=0) to leave an outline band. The
    mini-globe (disk + one latitude + one meridian ellipse, wireframe punched
    like motifs/globe.py) sits in the punched interior.
    """
    s = n
    cx = 0.50 * s
    cy = 0.50 * s

    def heart(scale: float, fill: int) -> None:
        lobe_r = 0.17 * s * scale
        lobe_dx = 0.15 * s * scale
        lobe_cy = cy - 0.12 * s * scale
        for sign in (-1.0, 1.0):
            lx = cx + sign * lobe_dx
            draw.ellipse(
                (lx - lobe_r, lobe_cy - lobe_r, lx + lobe_r, lobe_cy + lobe_r),
                fill=fill,
            )
        draw.polygon(
            [
                (cx - 0.30 * s * scale, cy - 0.045 * s * scale),
                (cx + 0.30 * s * scale, cy - 0.045 * s * scale),
                (cx, cy + 0.34 * s * scale),
            ],
            fill=fill,
        )

    heart(1.0, 255)   # outer body
    heart(0.72, 0)    # punch interior -> outline band

    # Mini-globe centered in the punched interior.
    gr = 0.10 * s
    gcy = cy - 0.01 * s
    draw.ellipse((cx - gr, gcy - gr, cx + gr, gcy + gr), fill=255)
    line_thickness = max(1, int(0.010 * s))
    for i in range(line_thickness):
        d = i - line_thickness // 2
        # Equator — thin horizontal ellipse.
        draw.ellipse(
            (cx - gr, gcy - 0.25 * gr + d, cx + gr, gcy + 0.25 * gr + d),
            outline=0,
        )
        # Prime meridian — thin vertical ellipse.
        draw.ellipse(
            (cx - 0.45 * gr + d, gcy - gr, cx + 0.45 * gr + d, gcy + gr),
            outline=0,
        )


def jp_monogram_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — blocky "J + P" monogram."""
    del extent_um  # silhouette is scale-free; caller controls cell_um to hit extent
    return render_silhouette(_draw_jp_monogram, n_grid)


def heart_globe_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — heart outline with a mini-globe inside."""
    del extent_um  # silhouette is scale-free; caller controls cell_um to hit extent
    return render_silhouette(_draw_heart_globe, n_grid)
