from __future__ import annotations

"""Food pair — a steaming coffee cup beside an arepa on a plate.

The couple's shared food love (Colombian coffee + arepa). The read: a rounded
COFFEE CUP with a handle on a saucer at the left, elegant STEAM curls rising
from it; to the right a round PLATE seen in slight perspective with a thick
round AREPA on it. Steam is the showpiece — long tapering S-curves. Authored in
a normalized 0..1 art box (Pillow y-down).
"""

import numpy as np
from PIL import ImageDraw

from .._pillow import render_silhouette
from ._draw import Pen, bezier, chain


def _steam(p: Pen, x0, y0, height, width, phase=1.0):
    """One elegant curling steam ribbon rising from (x0, y0).

    A double-S: the centerline weaves left-right-left as it rises while the
    ribbon WIDTH tapers from `width` at the base to a fine point at the top, so
    it reads as a wisp of curling steam rather than a stiff reed. `phase` (+/-1)
    mirrors the weave so adjacent curls alternate and interleave.
    """
    ph = phase
    sway = 0.06  # how far the centerline weaves side to side

    # centerline: base -> (sway one way) -> (sway back) -> tip
    cx0, cy0 = x0, y0
    cx1, cy1 = x0 + ph * sway, y0 - height * 0.30
    cx2, cy2 = x0 - ph * sway, y0 - height * 0.62
    cx3, cy3 = x0 + ph * sway * 0.4, y0 - height           # tip
    center = bezier((cx0, cy0), (cx1, cy1), (cx2, cy2), (cx3, cy3), n=32)

    # build a ribbon by offsetting the centerline left/right, tapering width->0.
    import math
    left_pts, right_pts = [], []
    npts = len(center)
    for i in range(npts):
        t = i / (npts - 1)
        w = width * (1.0 - t) ** 1.3          # taper to a point at the tip
        # local tangent for perpendicular offset
        j = min(i + 1, npts - 1)
        k = max(i - 1, 0)
        dx = center[j][0] - center[k][0]
        dy = center[j][1] - center[k][1]
        L = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / L, dx / L
        cxp, cyp = center[i]
        left_pts.append((cxp + nx * w, cyp + ny * w))
        right_pts.append((cxp - nx * w, cyp - ny * w))
    p.poly(left_pts + list(reversed(right_pts)))


# Base y-coordinate the three steam curls rise from (cup mouth). Shared by the
# body/steam split so the chirp pattern can locate the steam band precisely.
_STEAM_BASE_Y = 0.56
_STEAM_CURLS = (
    # (x0, height, width, phase)
    (0.25, 0.28, 0.020, +1),
    (0.31, 0.36, 0.024, -1),
    (0.37, 0.25, 0.018, +1),
)

# ---------------------------------------------------------------------------
# AREPA geometry (shared by the bodies-only mask and the full silhouette so the
# two can never drift). An arepa is a FAT round corn patty — the cues we lean
# on to make it unmistakable (vs. a generic bun/tortilla):
#   * real THICKNESS: a tall front wall + a crease ring, seen in slight 3/4 view;
#   * CHAR MARKS: two crossed sets of parallel grill sear lines on the top face
#     (the budare/parrilla read) — punched as negative-space grooves;
#   * SPLIT open with a cheese/filling bulge + its own steam wisp (arepa rellena);
#   * a rough/DIMPLED corn-masa rim.
_AREPA_CX = 0.72          # patty centre x
_AREPA_TOP_CY = 0.705     # top-face centre y
_AREPA_RX = 0.150         # top-face radius x
_AREPA_RY = 0.056         # top-face radius y
_AREPA_WALL = 0.066       # thickness (top face -> bottom of wall); fat disc
_AREPA_WISP_BASE_Y = 0.676  # where the split-steam wisp starts (on the seam)


def _draw_steam(p: Pen) -> None:
    """The three cup steam ribbons + the small wisp rising from the split arepa.

    All steam lives in the chirped-carrier mask, so the arepa's own wisp shimmers
    on tilt exactly like the cup steam — reinforcing the "hot off the budare"
    read of a freshly opened arepa.
    """
    for x0, h, w, ph in _STEAM_CURLS:
        _steam(p, x0, _STEAM_BASE_Y, h, w, phase=ph)
    # Arepa wisp: short, rising from the split in the patty face.
    _steam(p, _AREPA_CX, _AREPA_WISP_BASE_Y, 0.14, 0.012, phase=-1)


def _arepa_top_ring(scale: float = 1.0):
    """Sample the elliptical outline of the patty top face (for dimpled rim)."""
    import math

    cx, cy, rx, ry = _AREPA_CX, _AREPA_TOP_CY, _AREPA_RX * scale, _AREPA_RY * scale
    pts = []
    for i in range(96):
        a = 2 * math.pi * i / 96
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return pts


def _draw_arepa(p: Pen) -> None:
    """Solid gold of the arepa-on-a-plate: plate, thick patty (top face + tall
    front wall), and the corn-masa DIMPLES bumped out around the rim.

    Drawn identically in the bodies-only mask and the full silhouette."""
    import math

    cx = _AREPA_CX
    rx, ry = _AREPA_RX, _AREPA_RY
    top_cy = _AREPA_TOP_CY
    wall = _AREPA_WALL
    bot_cy = top_cy + wall  # bottom rim centre

    # Plate: a wide shallow dish. Kept a touch lower/wider so a gold rim of
    # plate shows AROUND the patty (patty no longer merges into the plate).
    p.ellipse(cx, 0.815, 0.235, 0.070, fill=255)

    # Front wall of the patty (thickness): the visible band between the top-face
    # front lip and the bottom-rim front lip. A tall wall = a FAT arepa.
    p.poly(chain(
        bezier((cx - rx, top_cy), (cx - rx, top_cy + wall * 0.55),
               (cx - rx * 0.6, bot_cy), (cx, bot_cy + ry * 0.15)),
        bezier((cx, bot_cy + ry * 0.15), (cx + rx * 0.6, bot_cy),
               (cx + rx, top_cy + wall * 0.55), (cx + rx, top_cy)),
        # back up along the top-face front lip
        bezier((cx + rx, top_cy), (cx + rx * 0.55, top_cy + ry),
               (cx - rx * 0.55, top_cy + ry), (cx - rx, top_cy)),
    ))

    # Top face of the patty.
    p.ellipse(cx, top_cy, rx, ry, fill=255)

    # Corn-masa DIMPLES: small gold bumps studded around the rim so the edge
    # reads rough/handmade rather than machined-smooth.
    for i in range(18):
        a = 2 * math.pi * i / 18
        # only the lower ~2/3 of the ring (front + sides read as texture)
        if math.sin(a) < -0.35:
            continue
        bx = cx + rx * 1.005 * math.cos(a)
        by = top_cy + ry * 1.005 * math.sin(a)
        p.ellipse(bx, by, 0.010, 0.008, fill=255)


def _hatch_family(p: Pen, angle_deg: float, offsets, foreshorten: float,
                  width_unit: float = 0.006) -> None:
    """One family of parallel grill-sear lines across the arepa top face.

    Lines run at ``angle_deg`` (measured on the visually-flattened face), spaced
    at ``offsets`` (fractions of the face radius along the perpendicular). Each
    line is clipped to the top ellipse and squashed vertically by
    ``foreshorten`` (<1) so the marks sit believably on a disc seen in 3/4 view.
    Punched fill=0 (dark grooves in the gold)."""
    import math

    cx = _AREPA_CX
    rx, ry = _AREPA_RX, _AREPA_RY
    top_cy = _AREPA_TOP_CY
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)      # line direction (unit)
    nx, ny = -dy, dx                       # perpendicular (unit)

    for off in offsets:
        # A point on the line: face centre shifted along the perpendicular.
        px = off * rx * nx
        py = off * rx * ny
        # Intersect the infinite line (through (px,py), dir (dx,dy)) with the
        # unit circle (we work in circle space, then scale x by rx, y by ry).
        # Circle-space direction/point:
        Dx, Dy = dx, dy
        Px, Py = px / rx, py / rx  # px,py were built from rx units → normalize
        A = Dx * Dx + Dy * Dy
        B = 2 * (Px * Dx + Py * Dy)
        C = Px * Px + Py * Py - 1.0
        disc = B * B - 4 * A * C
        if disc <= 0:
            continue
        sq = math.sqrt(disc)
        t0, t1 = (-B - sq) / (2 * A), (-B + sq) / (2 * A)
        # inset the ends slightly so lines stop short of the rim
        span = t1 - t0
        t0 += span * 0.10
        t1 -= span * 0.10
        x0 = cx + (Px + Dx * t0) * rx
        x1 = cx + (Px + Dx * t1) * rx
        y0 = top_cy + (Py + Dy * t0) * ry * foreshorten
        y1 = top_cy + (Py + Dy * t1) * ry * foreshorten
        p.line([(x0, y0), (x1, y1)], width_unit, fill=0)


def _draw_arepa_marks(p: Pen) -> None:
    """Negative-space grooves on the patty: the top/wall crease, two CROSSED
    families of parallel grill char lines (the budare/parrilla read), and the
    SPLIT cut with a filling pocket the wisp rises from. All grooves punched
    fill=0, kept ≥ ~0.006 art units (~3 px @512, > 2 px at fab) for litho."""
    cx = _AREPA_CX
    rx, ry = _AREPA_RX, _AREPA_RY
    top_cy = _AREPA_TOP_CY

    # Crease separating the top face from the thickness wall (front arc).
    p.line(
        [(cx - rx * 0.96, top_cy + ry * 0.34),
         (cx, top_cy + ry * 0.66),
         (cx + rx * 0.96, top_cy + ry * 0.34)],
        0.006, fill=0,
    )

    # ---- CROSSED GRILL CHAR MARKS: two families at STEEP circle-space angles
    # (±58°). The 3/4-view squash of the disc (rx≫ry) flattens these to a
    # readable ~±22° visual X — the budare/parrilla sear. foreshorten=1.0: the
    # ellipse mapping already supplies the perspective squash, so no extra flat-
    # tening (that was collapsing the two families into parallel horizontals).
    # Symmetric offsets → a bold, even X-crosshatch reading unmistakably as
    # budare/parrilla sear marks (the single strongest arepa cue).
    _hatch_family(p, +58.0, (-0.55, 0.0, 0.55), foreshorten=1.0)
    _hatch_family(p, -58.0, (-0.55, 0.0, 0.55), foreshorten=1.0)

    # ---- SPLIT / arepa rellena: ONE clean seam arcing across the back edge of
    # the face (above the sear zone) with a thin dark filling pocket hugging its
    # lower lip — the "opened, filled" read without competing with the sears.
    # The steam wisp rises from this seam (wisp lives in the steam mask).
    split_y = top_cy - ry * 0.52
    p.line([(cx - rx * 0.52, split_y + 0.006),
            (cx, split_y),
            (cx + rx * 0.52, split_y + 0.006)],
           0.008, fill=0)
    p.poly(chain(
        bezier((cx - rx * 0.40, split_y + 0.011), (cx - rx * 0.14, split_y + 0.020),
               (cx + rx * 0.14, split_y + 0.020), (cx + rx * 0.40, split_y + 0.011)),
        bezier((cx + rx * 0.40, split_y + 0.011), (cx + rx * 0.14, split_y + 0.014),
               (cx - rx * 0.14, split_y + 0.014), (cx - rx * 0.40, split_y + 0.011)),
    ), fill=0)


def _draw_bodies(draw: ImageDraw.ImageDraw, n: int) -> None:
    """The cup + saucer + arepa + plate WITHOUT the steam (body-only mask)."""
    p = Pen(draw, n)
    _draw_body_shapes(p)


def _draw_body_shapes(p: Pen) -> None:
    # ===================== COFFEE CUP (left) ===============================
    # Saucer: a shallow ellipse the cup sits on.
    p.ellipse(0.30, 0.80, 0.20, 0.045, fill=255)
    # Cup body: a rounded tapering vessel.
    cup = chain(
        bezier((0.17, 0.58), (0.17, 0.70), (0.21, 0.77), (0.30, 0.775)),   # left wall down
        bezier((0.30, 0.775), (0.39, 0.77), (0.43, 0.70), (0.43, 0.58)),   # right wall up
        bezier((0.43, 0.58), (0.35, 0.615), (0.25, 0.615), (0.17, 0.58)),  # rim (front lip)
    )
    p.poly(cup)
    # Rim back edge so the cup mouth reads as an opening.
    p.poly(chain(
        bezier((0.17, 0.58), (0.25, 0.55), (0.35, 0.55), (0.43, 0.58)),
        bezier((0.43, 0.58), (0.35, 0.605), (0.25, 0.605), (0.17, 0.58)),
    ))
    # Handle: a C-loop on the right side.
    p.poly(chain(
        bezier((0.43, 0.63), (0.52, 0.61), (0.54, 0.68), (0.50, 0.72)),
        bezier((0.50, 0.72), (0.485, 0.695), (0.475, 0.685), (0.44, 0.685)),
        bezier((0.44, 0.685), (0.49, 0.67), (0.485, 0.635), (0.44, 0.645)),
        bezier((0.44, 0.645), (0.43, 0.64), (0.43, 0.635), (0.43, 0.63)),
    ))

    # ===================== AREPA ON A PLATE (right) ========================
    _draw_arepa(p)

    # ----- NEGATIVE SPACE on the bodies -----
    # Cup interior shadow (the dark coffee surface just inside the rim).
    p.poly(chain(
        bezier((0.19, 0.582), (0.26, 0.562), (0.34, 0.562), (0.41, 0.582)),
        bezier((0.41, 0.582), (0.34, 0.60), (0.26, 0.60), (0.19, 0.582)),
    ), fill=0)
    # Handle hole (so the C-handle reads open).
    p.ellipse(0.485, 0.672, 0.018, 0.026, fill=0)
    # Saucer/cup separation crease.
    p.line([(0.13, 0.795), (0.47, 0.795)], 0.004, fill=0)
    # Arepa char marks + crease + split (crossed sear lines = the budare read).
    _draw_arepa_marks(p)


def _draw_food(draw: ImageDraw.ImageDraw, n: int) -> None:
    p = Pen(draw, n)

    # ===================== COFFEE CUP (left) ===============================
    # Saucer: a shallow ellipse the cup sits on.
    p.ellipse(0.30, 0.80, 0.20, 0.045, fill=255)
    # Cup body: a rounded tapering vessel.
    cup = chain(
        bezier((0.17, 0.58), (0.17, 0.70), (0.21, 0.77), (0.30, 0.775)),   # left wall down
        bezier((0.30, 0.775), (0.39, 0.77), (0.43, 0.70), (0.43, 0.58)),   # right wall up
        bezier((0.43, 0.58), (0.35, 0.615), (0.25, 0.615), (0.17, 0.58)),  # rim (front lip)
    )
    p.poly(cup)
    # Rim back edge so the cup mouth reads as an opening.
    p.poly(chain(
        bezier((0.17, 0.58), (0.25, 0.55), (0.35, 0.55), (0.43, 0.58)),
        bezier((0.43, 0.58), (0.35, 0.605), (0.25, 0.605), (0.17, 0.58)),
    ))
    # Handle: a C-loop on the right side.
    p.poly(chain(
        bezier((0.43, 0.63), (0.52, 0.61), (0.54, 0.68), (0.50, 0.72)),
        bezier((0.50, 0.72), (0.485, 0.695), (0.475, 0.685), (0.44, 0.685)),
        bezier((0.44, 0.685), (0.49, 0.67), (0.485, 0.635), (0.44, 0.645)),
        bezier((0.44, 0.645), (0.43, 0.64), (0.43, 0.635), (0.43, 0.63)),
    ))

    # ----- STEAM: three tapering curls rising from the cup mouth. -----------
    _steam(p, 0.25, 0.56, 0.28, 0.020, phase=+1)
    _steam(p, 0.31, 0.56, 0.36, 0.024, phase=-1)
    _steam(p, 0.37, 0.56, 0.25, 0.018, phase=+1)

    # ===================== AREPA ON A PLATE (right) ========================
    _draw_arepa(p)

    # ----- STEAM: the arepa's own split wisp (cup steam drawn above). ---------
    _steam(p, _AREPA_CX, _AREPA_WISP_BASE_Y, 0.14, 0.012, phase=-1)

    # =======================================================================
    # NEGATIVE SPACE
    # =======================================================================
    # Cup interior shadow (the dark coffee surface just inside the rim).
    p.poly(chain(
        bezier((0.19, 0.582), (0.26, 0.562), (0.34, 0.562), (0.41, 0.582)),
        bezier((0.41, 0.582), (0.34, 0.60), (0.26, 0.60), (0.19, 0.582)),
    ), fill=0)
    # Handle hole (so the C-handle reads open).
    p.ellipse(0.485, 0.672, 0.018, 0.026, fill=0)
    # Saucer/cup separation crease.
    p.line([(0.13, 0.795), (0.47, 0.795)], 0.004, fill=0)
    # Arepa char marks + crease + split (crossed sear lines = the budare read).
    _draw_arepa_marks(p)


def food_pair_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — steaming coffee cup + arepa on plate."""
    del extent_um
    return render_silhouette(_draw_food, n_grid)


def food_bodies_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid — cup + saucer + arepa + plate ONLY (no steam).

    The region that gets the standard uniform grating fill in ``food-pair-chirp``.
    """
    del extent_um
    return render_silhouette(_draw_bodies, n_grid)


def food_steam_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid — the three rising STEAM ribbons ONLY.

    The region that carries the chirped (fine→coarse) grating in
    ``food-pair-chirp`` so a tilt sends a shimmer wave up the steam.
    """
    del extent_um
    return render_silhouette(lambda d, n: _draw_steam(Pen(d, n)), n_grid)
