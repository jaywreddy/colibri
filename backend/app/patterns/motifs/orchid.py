from __future__ import annotations

import math

import numpy as np
from PIL import ImageDraw

from ._pillow import render_silhouette


def _rotated_ellipse(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float,
    n_pts: int = 48,
) -> list[tuple[float, float]]:
    """Polygon approximation of an ellipse rotated about its own center.

    PIL's ``ellipse`` only draws axis-aligned; petals need arbitrary tilt, so
    each petal is a 48-gon sampled from the parametric ellipse and rotated.
    Angles are in PIL image coordinates (x right, y DOWN), so a petal at
    -90° points toward the top of the canvas.
    """
    rot = math.radians(angle_deg)
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    pts = []
    for k in range(n_pts):
        t = 2.0 * math.pi * k / n_pts
        x = rx * math.cos(t)
        y = ry * math.sin(t)
        pts.append((cx + x * cos_r - y * sin_r, cy + x * sin_r + y * cos_r))
    return pts


def _petal(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    angle_deg: float,
    length: float,
    width: float,
) -> None:
    """Filled elliptical petal radiating from (cx, cy) along angle_deg."""
    ang = math.radians(angle_deg)
    px = cx + (length / 2.0) * math.cos(ang)
    py = cy + (length / 2.0) * math.sin(ang)
    draw.polygon(
        _rotated_ellipse(px, py, length / 2.0, width / 2.0, angle_deg),
        fill=255,
    )


def _draw_orchid(draw: ImageDraw.ImageDraw, n: int) -> None:
    """Stylized Cattleya trianae (Colombia's national flower) silhouette.

    Built like globe.py: filled body first, then structure lines punched in
    black so the silhouette carries internal detail when used as a Moiré
    mask. Layout — five members radiating from the flower center (one narrow
    dorsal sepal straight up, two broad upper petals, two wide lateral
    sepals just below horizontal) plus the prominent frilled lip (labellum)
    hanging below: a central trumpet ellipse with a scalloped fringe of
    small disks along its lower edge.
    """
    s = n
    cx = 0.50 * s
    cy = 0.44 * s

    # --- five petals / sepals (filled) -----------------------------------
    # Narrow dorsal sepal, straight up.
    _petal(draw, cx, cy, -90.0, 0.42 * s, 0.11 * s)
    # Two broad upper petals, flanking the dorsal sepal.
    _petal(draw, cx, cy, -90.0 - 52.0, 0.44 * s, 0.20 * s)
    _petal(draw, cx, cy, -90.0 + 52.0, 0.44 * s, 0.20 * s)
    # Two wide lateral sepals, drooping just below horizontal.
    _petal(draw, cx, cy, 180.0 - 18.0, 0.42 * s, 0.14 * s)
    _petal(draw, cx, cy, 18.0, 0.42 * s, 0.14 * s)

    # --- labellum (frilled lip) below the center --------------------------
    lip_cx = cx
    lip_cy = cy + 0.15 * s
    # Central trumpet.
    draw.polygon(
        _rotated_ellipse(lip_cx, lip_cy, 0.11 * s, 0.15 * s, 90.0),
        fill=255,
    )
    # Scalloped fringe: a fan of small disks along the lip's lower arc gives
    # the ruffled Cattleya edge without any curve primitives.
    frill_r = 0.045 * s
    for ang_deg in range(20, 161, 14):
        ang = math.radians(ang_deg)
        fx = lip_cx + 0.12 * s * math.cos(ang)
        fy = lip_cy + 0.06 * s + 0.11 * s * math.sin(ang)
        draw.ellipse(
            (fx - frill_r, fy - frill_r, fx + frill_r, fy + frill_r),
            fill=255,
        )

    # --- structure lines punched in black (like globe.py's wireframe) -----
    vein_w = max(1, int(0.012 * s))
    # Midrib vein down each petal/sepal, from just outside the throat to
    # ~85% of the petal length.
    for ang_deg, length in (
        (-90.0, 0.42 * s),
        (-142.0, 0.44 * s),
        (-38.0, 0.44 * s),
        (162.0, 0.42 * s),
        (18.0, 0.42 * s),
    ):
        ang = math.radians(ang_deg)
        x0 = cx + 0.06 * s * math.cos(ang)
        y0 = cy + 0.06 * s * math.sin(ang)
        x1 = cx + 0.85 * length * math.cos(ang)
        y1 = cy + 0.85 * length * math.sin(ang)
        draw.line((x0, y0, x1, y1), fill=0, width=vein_w)
    # Lip midline vein.
    draw.line(
        (lip_cx, lip_cy - 0.06 * s, lip_cx, lip_cy + 0.16 * s),
        fill=0,
        width=vein_w,
    )
    # Throat mouth — a small dark ellipse outline where the column sits, so
    # the flower center reads as a trumpet opening rather than a solid blot.
    mouth_thickness = max(1, int(0.012 * s))
    for i in range(mouth_thickness):
        d = i - mouth_thickness // 2
        draw.ellipse(
            (cx - 0.055 * s - d, cy - 0.04 * s - d,
             cx + 0.055 * s + d, cy + 0.05 * s + d),
            outline=0,
        )
    # Column eye — dark dot at the very center.
    draw.ellipse(
        (cx - 0.018 * s, cy - 0.012 * s, cx + 0.018 * s, cy + 0.024 * s),
        fill=0,
    )


def orchid_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — Cattleya trianae orchid.

    True where the flower covers; the punched veins / throat are False, so
    the silhouette has internal structure and reads as an orchid rather than
    a blob when intersected with a carrier grating.
    """
    del extent_um  # silhouette is scale-free; caller controls cell_um to hit extent
    return render_silhouette(_draw_orchid, n_grid)
