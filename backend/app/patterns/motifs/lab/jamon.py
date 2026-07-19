from __future__ import annotations

"""Jamón ibérico on a jamonero — "the Spanish table she grew up slicing at".

The read: a whole cured HAM LEG (jamón) clamped hoof-UP on a wooden carving
stand (jamonero), with a couple of thin carved SLICES resting on the board.

Anatomy the silhouette leans on so it reads unmistakably as a jamón (not a
generic leg of meat):
  * HOOF (pezuña) — the little pointed black trotter, up in the air, set off
    from the shank by the ankle-joint groove.
  * SHANK (caña) — the slim ankle the clamp grips.
  * MAZA — the full, heavy, rounded body of the leg swelling below the shank,
    tapering to a rounded PUNTA (hip tip).
  * RIND edge (corteza) — a thin negative-space line hugging the back of the
    leg, and a flat CUT FACE with a couple of score lines near the punta where
    slices have been taken.

The jamonero:
  * a wooden BASE BOARD in slight perspective (with a thickness lip);
  * a vertical POST on the right rising from the board;
  * a CLAMP arm + cup gripping the shank at the top.

Authored in a normalized 0..1 art box (Pillow convention, y grows DOWN) and
scaled to the raster grid at draw time — same contract as the other lab motifs.
Everything is kept ≥ ~0.006 art units wide (> the 2 µm litho floor at fab
pitch) so no feature prints below the minimum line.
"""

import math

import numpy as np
from PIL import ImageDraw

from .._pillow import render_silhouette
from ._draw import Pen, bezier, chain

# --- leg geometry (shared by the outline, the rind line, and the cut face) ---
# Centerline of the leg, hoof (top-right) -> punta (lower-left), tilted so the
# heavy body hangs out over the board.
_HOOF = (0.72, 0.13)
_LEG_CTRL = ((0.685, 0.27), (0.52, 0.46), (0.27, 0.66))   # c1, c2, punta
_SHANK_END = 0.22        # centerline param where the slim shank ends & body swells
_PUNTA_R = 0.030         # rounded-cap radius at the punta (hip tip)


def _leg_center(nseg: int = 64):
    return bezier(_HOOF, _LEG_CTRL[0], _LEG_CTRL[1], _LEG_CTRL[2], n=nseg)


def _leg_width(t: float) -> tuple[float, float]:
    """(left, right) perpendicular half-widths of the leg at centerline param t.

    ``left`` is the back/maza side (bulges heavily); ``right`` is the front side
    (slimmer, where the cut face sits). The defining jamón cue is the CONTRAST
    between the slim shank near the hoof and the heavy, rounded maza below it, so
    the shank is held thin and the body swells fast to a plump belly biased
    toward the lower leg, then rounds off (never tapering to a point) at the
    punta — which a circular cap finishes off.
    """
    if t < _SHANK_END:                   # hoof + shank: slim, near-constant
        base = 0.024 + 0.007 * (t / _SHANK_END)
        return base, base * 0.9
    u = (t - _SHANK_END) / (1.0 - _SHANK_END)   # 0..1 across the body
    # swell biased to the lower-middle leg (heavy maza), floored so the punta
    # stays blunt (rounded by the _PUNTA_R cap) rather than tapering to a point.
    swell = math.sin((u ** 1.15) * math.pi)
    left = 0.030 + 0.098 * swell         # maza (back) — the full round bulge
    right = 0.028 + 0.060 * swell        # front side, slimmer
    return left, right


def _leg_outline():
    center = _leg_center()
    n = len(center)
    left_pts, right_pts = [], []
    for i in range(n):
        t = i / (n - 1)
        wl, wr = _leg_width(t)
        j = min(i + 1, n - 1)
        k = max(i - 1, 0)
        dx = center[j][0] - center[k][0]
        dy = center[j][1] - center[k][1]
        L = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / L, dx / L          # left-hand normal
        cx, cy = center[i]
        left_pts.append((cx + nx * wl, cy + ny * wl))
        right_pts.append((cx - nx * wr, cy - ny * wr))
    return left_pts, right_pts, center


def _draw_jamon(draw: ImageDraw.ImageDraw, n: int) -> None:
    p = Pen(draw, n)

    left_pts, right_pts, center = _leg_outline()

    # ===================== JAMONERO STAND ==================================
    # Base board: a plank in slight perspective (far edge shorter than near),
    # with a thin front lip for thickness.
    p.poly([
        (0.16, 0.845), (0.86, 0.845),            # far (top) edge
        (0.90, 0.905), (0.12, 0.905),            # near (bottom) edge
    ])
    # board thickness lip (front face), a touch darker read via a crease line
    p.line([(0.12, 0.905), (0.90, 0.905)], 0.006, fill=255)

    # Post: vertical bar rising on the right from the board to the clamp.
    p.poly([
        (0.795, 0.30), (0.85, 0.30),
        (0.85, 0.85), (0.795, 0.85),
    ])
    # Clamp arm: from the post reaching left to the shank.
    p.poly([
        (0.80, 0.34), (0.80, 0.40),
        (0.665, 0.34), (0.665, 0.28),
    ])
    # Clamp cup: a small curved bracket cradling the shank just below the hoof.
    p.poly(chain(
        bezier((0.615, 0.265), (0.595, 0.325), (0.66, 0.35), (0.71, 0.335)),
        bezier((0.71, 0.335), (0.685, 0.30), (0.665, 0.28), (0.645, 0.255)),
    ))

    # ===================== HAM LEG =========================================
    p.poly(left_pts + list(reversed(right_pts)))
    # Rounded punta cap (hip tip) so the leg ends blunt-round, not pointed.
    punta = center[-1]
    p.ellipse(punta[0], punta[1], _PUNTA_R, _PUNTA_R, fill=255)

    # Hoof (pezuña): a small compact rounded trotter knob at the hoof end, with
    # a stubby blunt tip and a short cleft — reads as a foot, not a flame.
    p.ellipse(0.725, 0.115, 0.026, 0.030, fill=255)   # trotter knob
    p.poly([(0.715, 0.095), (0.732, 0.075), (0.742, 0.10)])  # stubby blunt tip
    p.line([(0.726, 0.088), (0.726, 0.118)], 0.005, fill=0)  # cleft in the hoof

    # ===================== CARVED SLICES on the board ======================
    # Three thin oval slices fanned on the board, slightly overlapping, each
    # with a thin fat-rim arc so they read as loncheado jamón, not coins.
    for (sx, sy, rot) in ((0.30, 0.872, -10.0), (0.40, 0.882, 2.0), (0.50, 0.876, 12.0)):
        _slice(p, sx, sy, 0.072, 0.024, rot)

    # =======================================================================
    # NEGATIVE SPACE — the anatomy cues
    # =======================================================================
    # Ankle-joint groove separating the hoof/shank from the body.
    p.line([(0.655, 0.205), (0.705, 0.185)], 0.006, fill=0)

    # Rind (corteza) line: a long thin groove hugging the back (maza) edge,
    # a hair inside the outline, marking the cured-skin boundary.
    back = []
    for i in range(6, len(center) - 4):
        t = i / (len(center) - 1)
        wl, _ = _leg_width(t)
        j = min(i + 1, len(center) - 1)
        k = max(i - 1, 0)
        dx = center[j][0] - center[k][0]
        dy = center[j][1] - center[k][1]
        L = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / L, dx / L
        cx, cy = center[i]
        back.append((cx + nx * (wl - 0.018), cy + ny * (wl - 0.018)))
    p.line(back, 0.005, fill=0)

    # Cut face on the front of the lower leg: a couple of score lines marking
    # where slices have been carved off the maza.
    p.line([(0.38, 0.585), (0.45, 0.53)], 0.005, fill=0)
    p.line([(0.41, 0.615), (0.485, 0.56)], 0.005, fill=0)


def _slice_oval(cx, cy, rx, ry, ca, sa, ns=48):
    pts = []
    for i in range(ns):
        th = 2 * math.pi * i / ns
        ex, ey = rx * math.cos(th), ry * math.sin(th)
        pts.append((cx + ex * ca - ey * sa, cy + ex * sa + ey * ca))
    return pts


def _slice(p: Pen, cx: float, cy: float, rx: float, ry: float, rot_deg: float) -> None:
    """One thin ham slice resting on the (gold) board.

    Because the board is also gold, a plain gold oval would vanish into it — so
    the slice is separated by a thin DARK halo punched around it (board gold →
    dark ring → gold slice), then a thin dark rim arc marks the slice's fat edge.
    """
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    g = 0.010   # dark separation-halo width
    p.poly(_slice_oval(cx, cy, rx + g, ry + g, ca, sa), fill=0)   # dark halo
    p.poly(_slice_oval(cx, cy, rx, ry, ca, sa), fill=255)         # gold slice
    # thin dark rim arc along the upper edge (the slice's fat/rind border)
    rim = []
    for i in range(24):
        th = math.pi + math.pi * i / 23          # upper half
        ex, ey = rx * 0.80 * math.cos(th), ry * 0.80 * math.sin(th)
        rim.append((cx + ex * ca - ey * sa, cy + ex * sa + ey * ca))
    p.line(rim, 0.004, fill=0)


def jamon_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — jamón ibérico on a jamonero."""
    del extent_um  # scale-free; caller controls cell pitch to hit the extent
    return render_silhouette(_draw_jamon, n_grid)
