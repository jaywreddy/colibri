from __future__ import annotations

import math
from functools import partial

import numpy as np
from PIL import ImageDraw

from ._pillow import render_silhouette


def _draw_globe(draw: ImageDraw.ImageDraw, n: int, rotation_deg: float = 0.0) -> None:
    """Stylized globe silhouette: filled disk + latitude/longitude wireframe.

    The pattern reads as "globe" rather than "circle" because of the three
    foreshortened longitude ellipses (vertical meridians at 0°, ±60° in their
    apparent x-radius) and three latitude arcs (equator + two tropics) drawn
    as thin horizontal ellipses. A small landmass blob over the lower-left
    sector — abstract enough not to dominate — nods to the Colombia theme.

    ``rotation_deg`` spins the globe about its polar axis: each meridian's
    apparent longitude becomes ``lon_deg + rotation_deg`` before the cos
    foreshortening, and the landmass blob translates horizontally across the
    disk (wrapping around the limb), so two renders at different rotations
    genuinely read as "the globe turned". ``rotation_deg=0`` draws the exact
    same primitives in the same order as before the parameter existed, so the
    default output stays byte-identical.
    """
    s = n
    cx = 0.5 * s
    cy = 0.5 * s
    r = 0.42 * s

    # Earth disk — the filled body of the globe.
    draw.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        fill=255,
    )

    # Latitude arcs: equator + ±tropics. Drawn as thin black ellipses on the
    # filled disk so they read as wireframe rings on a solid sphere. Each
    # latitude ellipse has the same horizontal radius (= r) and a small
    # vertical radius set by the latitude angle.
    lat_thickness = max(1, int(0.014 * s))
    for lat_deg in (-23.0, 0.0, 23.0):
        lat = math.radians(lat_deg)
        y_offset = r * math.sin(lat)
        ry = max(1.5, r * 0.10 * math.cos(lat))
        # Outline ellipse via two filled ellipses (the thicker one in black,
        # then a smaller white inset would re-cover the body; instead just
        # punch a thin black band by drawing two arcs as polylines).
        for i in range(lat_thickness):
            offset = i - lat_thickness // 2
            draw.ellipse(
                (cx - r, cy + y_offset - ry + offset,
                 cx + r, cy + y_offset + ry + offset),
                outline=0,
            )

    # Longitude arcs: prime meridian + ±60°. The apparent x-radius shrinks as
    # the meridian rotates away from the viewer (cos), giving the foreshortened
    # 3D look.
    lon_thickness = max(1, int(0.014 * s))
    for lon_deg in (-60.0, 0.0, 60.0):
        # Apparent longitude = true longitude + spin (rotation about the pole).
        lon = math.radians(lon_deg + rotation_deg)
        rx = max(1.5, r * abs(math.cos(lon)))
        for i in range(lon_thickness):
            offset = i - lon_thickness // 2
            draw.ellipse(
                (cx - rx + offset, cy - r,
                 cx + rx + offset, cy + r),
                outline=0,
            )

    # Limb circle (outline) — keep the disk crisp against the lat/lon wireframe.
    limb_thickness = max(2, int(0.018 * s))
    for i in range(limb_thickness):
        d = i - limb_thickness // 2
        draw.ellipse(
            (cx - r - d, cy - r - d, cx + r + d, cy + r + d),
            outline=0,
        )
    # Re-fill the inside so the lat/lon strokes are the only dark structure;
    # otherwise the limb_thickness outline closes over the body.
    draw.ellipse(
        (cx - r + limb_thickness, cy - r + limb_thickness,
         cx + r - limb_thickness, cy + r - limb_thickness),
        fill=255,
    )
    # Redraw lat/lon over the cleaned interior.
    for lat_deg in (-23.0, 0.0, 23.0):
        lat = math.radians(lat_deg)
        y_offset = r * math.sin(lat)
        ry = max(1.5, r * 0.10 * math.cos(lat))
        for i in range(lat_thickness):
            offset = i - lat_thickness // 2
            draw.ellipse(
                (cx - r, cy + y_offset - ry + offset,
                 cx + r, cy + y_offset + ry + offset),
                outline=0,
            )
    for lon_deg in (-60.0, 0.0, 60.0):
        # Apparent longitude = true longitude + spin (rotation about the pole).
        lon = math.radians(lon_deg + rotation_deg)
        rx = max(1.5, r * abs(math.cos(lon)))
        for i in range(lon_thickness):
            offset = i - lon_thickness // 2
            draw.ellipse(
                (cx - rx + offset, cy - r,
                 cx + rx + offset, cy + r),
                outline=0,
            )

    # Small landmass blob — abstract South-America-ish shape, lower-left.
    # Spin translates the blob horizontally across the disk face. The wrap
    # period 4r maps 360° of rotation onto one traverse of the visible face
    # (2r) plus the hidden back hemisphere (2r), so the blob slides off one
    # limb and re-enters at the other. Wrap copies that land entirely on the
    # hidden hemisphere (outside the disk's x-range) are skipped — with
    # rotation_deg=0 only the untranslated copy draws, keeping the default
    # output byte-identical.
    blob = [
        (cx - 0.10 * s, cy + 0.02 * s),
        (cx - 0.02 * s, cy - 0.04 * s),
        (cx + 0.04 * s, cy + 0.06 * s),
        (cx - 0.01 * s, cy + 0.20 * s),
        (cx - 0.12 * s, cy + 0.12 * s),
    ]
    wrap = 4.0 * r
    shift = (rotation_deg / 360.0) * wrap
    shift = (shift + wrap / 2.0) % wrap - wrap / 2.0
    for dx in (shift, shift - wrap, shift + wrap):
        xs = [x + dx for x, _ in blob]
        if max(xs) < cx - r or min(xs) > cx + r:
            continue
        draw.polygon([(x, y) for x, (_, y) in zip(xs, blob)], fill=0)


def globe_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — globe disk with wireframe markings.

    True where the rendered globe pixel is gold (foreground); False on the
    wireframe lines + outside the disk. The wireframe being False means the
    silhouette has structure inside the disk, so it reads as a globe rather
    than a featureless circle when used as a Moiré mask.

    ``rotation_deg`` spins the globe about its polar axis (see
    :func:`_draw_globe`); the default 0.0 reproduces the legacy output
    byte-for-byte.
    """
    del extent_um  # silhouette is scale-free; caller controls cell_um to hit extent
    return render_silhouette(partial(_draw_globe, rotation_deg=rotation_deg), n_grid)
