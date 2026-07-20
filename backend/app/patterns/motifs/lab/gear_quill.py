from __future__ import annotations

"""Gear + quill/book — "the engineer and the historian, building a life".

The read: a toothed GEAR (his engineering) interlocking with an open BOOK and a
QUILL pen (her history). The gear sits upper-left; the open book lies lower-right
with a quill laid diagonally across it. Where the gear meets the book, a gear
tooth slots into the book's top page corner so the two motifs mesh rather than
merely sit side by side. Authored in a normalized 0..1 art box (Pillow y-down).
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .._pillow import render_silhouette
from ..trace import TraceParams, trace_silhouette
from ._draw import Pen, bezier, chain

# CC0/PD reference feather (public-domain dedication by BWCNY). See
# lab/refs/LICENSE_NOTE.md. A real writing quill: asymmetric vane, curved
# rachis, tapering bare shaft to a nib, frayed barbs at the vane base — it
# traces into an unmistakable QUILL where the old hand-drawn vane read as a leaf.
_QUILL_REF = Path(__file__).parent / "refs" / "quill_pen_bwcny_source.png"


def _gear(p: Pen, cx, cy, r_out, r_in, teeth, hub_r, fill=255):
    """A spur gear: `teeth` trapezoidal teeth around a rim, with a hub hole.

    Built as one polygon walking the outer/inner radii, then the center is
    punched to a hub ring afterward by the caller.
    """
    pts = []
    for i in range(teeth):
        a0 = 2 * math.pi * i / teeth
        a1 = 2 * math.pi * (i + 0.32) / teeth
        a2 = 2 * math.pi * (i + 0.5) / teeth
        a3 = 2 * math.pi * (i + 0.82) / teeth
        # tooth: rise to outer radius across the tooth top, drop to inner in gap
        pts.append((cx + r_in * math.cos(a0), cy + r_in * math.sin(a0)))
        pts.append((cx + r_out * math.cos(a1), cy + r_out * math.sin(a1)))
        pts.append((cx + r_out * math.cos(a2), cy + r_out * math.sin(a2)))
        pts.append((cx + r_in * math.cos(a3), cy + r_in * math.sin(a3)))
    p.poly(pts, fill=fill)


def _draw_gear_quill(draw: ImageDraw.ImageDraw, n: int) -> None:
    p = Pen(draw, n)

    gcx, gcy = 0.34, 0.36
    r_out, r_in = 0.24, 0.185
    teeth = 12

    # ===================== GEAR (upper-left) ===============================
    _gear(p, gcx, gcy, r_out, r_in, teeth, hub_r=0.07)

    # ===================== OPEN BOOK (lower-right) =========================
    # Two facing pages tenting up from a central spine, seen in gentle
    # perspective. The book's upper-left page corner tucks UNDER the gear so the
    # gear tooth there meshes into the page corner.
    spine_top = (0.60, 0.52)
    spine_bot = (0.60, 0.78)
    # left page (toward the gear)
    left_page = chain(
        bezier(spine_top, (0.50, 0.50), (0.42, 0.52), (0.36, 0.57)),   # top edge up to gear
        bezier((0.36, 0.57), (0.40, 0.66), (0.46, 0.74), (0.52, 0.80)),  # outer edge down
        bezier((0.52, 0.80), spine_bot, spine_bot, spine_bot),            # bottom to spine
        bezier(spine_bot, spine_top, spine_top, spine_top),               # spine up
    )
    p.poly(left_page)
    # right page (away from the gear)
    right_page = chain(
        bezier(spine_top, (0.70, 0.50), (0.80, 0.52), (0.87, 0.57)),
        bezier((0.87, 0.57), (0.84, 0.66), (0.78, 0.74), (0.71, 0.80)),
        bezier((0.71, 0.80), spine_bot, spine_bot, spine_bot),
        bezier(spine_bot, spine_top, spine_top, spine_top),
    )
    p.poly(right_page)

    # ===================== QUILL (diagonal across the book) ================
    # A feather pen laid from lower-left (nib, dipped toward the spine) up to the
    # upper-right (plume). Shaft is a thin tapering sliver; plume is a barbed
    # teardrop; a nib point at the low end.
    nib = (0.50, 0.74)
    shaft_top = (0.855, 0.32)
    # shaft: a slim tapering sliver from nib (thin) to where the plume begins.
    # Build via a centerline with perpendicular offset tapering base->tip.
    shaft_center = bezier(nib, (0.63, 0.60), (0.75, 0.46), shaft_top, n=28)
    lft, rgt = [], []
    for i in range(len(shaft_center)):
        t = i / (len(shaft_center) - 1)
        w = 0.006 + 0.010 * t          # thin at nib, a touch wider up top
        j = min(i + 1, len(shaft_center) - 1)
        k = max(i - 1, 0)
        dx = shaft_center[j][0] - shaft_center[k][0]
        dy = shaft_center[j][1] - shaft_center[k][1]
        L = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / L, dx / L
        cxp, cyp = shaft_center[i]
        lft.append((cxp + nx * w, cyp + ny * w))
        rgt.append((cxp - nx * w, cyp - ny * w))
    p.poly(lft + list(reversed(rgt)))
    # nib point (a small dart at the low end).
    p.poly([(0.50, 0.735), (0.475, 0.775), (0.517, 0.752)])

    # plume: a pointed feather along the upper shaft. A tapered leaf-of-barbs
    # shape whose tip points up-right past shaft_top, with a clear rachis.
    plume_tip = (0.90, 0.20)
    plume_lo = (0.79, 0.44)   # where the plume meets the shaft
    plume = chain(
        bezier(plume_lo, (0.80, 0.34), (0.83, 0.26), plume_tip),          # inner/upper edge
        bezier(plume_tip, (0.90, 0.28), (0.88, 0.38), (0.82, 0.46)),      # outer edge
        bezier((0.82, 0.46), (0.805, 0.46), (0.795, 0.45), plume_lo),     # back to shaft
    )
    p.poly(plume)

    # =======================================================================
    # NEGATIVE SPACE
    # =======================================================================
    # Gear hub hole (ring).
    p.ellipse(gcx, gcy, 0.075, fill=0)
    p.ellipse(gcx, gcy, 0.032, fill=255)  # small center boss back in
    # Book spine crease + a few page lines on each page (reads as text/pages).
    p.line([spine_top, spine_bot], 0.006, fill=0)
    for yy in (0.60, 0.65, 0.70):
        p.line([(0.44, yy), (0.565, yy - 0.01)], 0.004, fill=0)     # left page lines
        p.line([(0.635, yy - 0.01), (0.80, yy)], 0.004, fill=0)     # right page lines
    # Quill plume rachis crease (the feather's central shaft) + barb slots so
    # the plume reads as a feather rather than a solid leaf.
    p.line([(0.80, 0.44), (0.895, 0.215)], 0.005, fill=0)   # rachis up the plume
    for s in (0.25, 0.45, 0.65, 0.85):
        # a point along the rachis, and a short barb slot angling off it.
        rx = 0.80 + s * (0.895 - 0.80)
        ry = 0.44 + s * (0.215 - 0.44)
        p.line([(rx, ry), (rx + 0.03, ry + 0.006)], 0.0035, fill=0)


def gear_quill_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — meshing gear + open book with quill."""
    del extent_um
    return render_silhouette(_draw_gear_quill, n_grid)


# ===========================================================================
# SPLIT SILHOUETTES for the gear<->quill TILT SWITCH (gear-quill-switch pattern)
# ---------------------------------------------------------------------------
# The switch interlaces the two images into alternating half-period back-layer
# columns behind a slit barrier (parallax-barrier construction, same as
# jp-monogram-phase) so tilt reveals one or the other. For that to read cleanly each
# image must stand ALONE and fill a comparable footprint (matched visual
# weight): a bold CENTERED gear for "the engineer", a bold CENTERED open book
# with a quill for "the historian". These are deliberately re-centered and
# scaled up versus the meshed composite above (which crowds each motif into a
# corner) so that head-on the two interlace as equals.
# ===========================================================================


def _draw_gear_only(draw: ImageDraw.ImageDraw, n: int) -> None:
    """A single bold centered spur gear — the engineer's emblem."""
    p = Pen(draw, n)
    gcx, gcy = 0.5, 0.5
    r_out, r_in = 0.40, 0.31
    teeth = 12
    _gear(p, gcx, gcy, r_out, r_in, teeth, hub_r=0.12)
    # Spoke arms: three bars across the web so the gear reads as machined, not a
    # solid disc — punch a wide hub ring, then lay spokes and a center boss.
    p.ellipse(gcx, gcy, 0.22, fill=0)                    # web window (open)
    for k in range(6):
        a = math.pi * k / 3.0
        dx, dy = math.cos(a), math.sin(a)
        # a spoke bar from just outside the boss to the inner rim
        r0, r1 = 0.085, 0.235
        wperp = 0.035
        nx, ny = -dy, dx
        p.poly([
            (gcx + dx * r0 + nx * wperp, gcy + dy * r0 + ny * wperp),
            (gcx + dx * r1 + nx * wperp, gcy + dy * r1 + ny * wperp),
            (gcx + dx * r1 - nx * wperp, gcy + dy * r1 - ny * wperp),
            (gcx + dx * r0 - nx * wperp, gcy + dy * r0 - ny * wperp),
        ], fill=255)
    p.ellipse(gcx, gcy, 0.085, fill=255)                 # center boss
    p.ellipse(gcx, gcy, 0.032, fill=0)                   # bore hole


def _draw_open_book(draw: ImageDraw.ImageDraw, n: int) -> None:
    """A clean, bold open book lying open on a central spine — the base the quill
    is laid across. Built strictly symmetric about x=0.50 from a single per-page
    template mirrored left/right, so the two leaves are identical mirror images:

      * CENTER GUTTER V — the spine is a vertical valley; both pages meet along
        it at a shared top point (``st``) and bottom point (``sb``).
      * PAGE-BELLY CURVES — each page's top and bottom long edges bow gently
        outward (the top edge bellies UP, the bottom edge bellies DOWN) so the
        leaf reads as a real sheet of paper lifting off the spine, not a flat wedge.
      * FORE-EDGE THICKNESS — a thin second leaf drawn just under each page's
        bottom belly (a hair inset), separated by a dark hairline, reads as the
        stacked cut edges of the pages at the outer fore-edge.

    A spine crease and a few ruled text lines are punched as negative space.
    Deliberately simple so the traced feather laid over it stays the hero.
    """
    p = Pen(draw, n)
    st, sb = (0.50, 0.46), (0.50, 0.78)          # gutter top / bottom (the V)
    # Bottom fore-edge curve of one leaf (shared by the face, the thickness
    # sliver, and the separating hairline so they can never drift).
    def _bottom(sign, y_off=0.0):
        return bezier((0.50 + sign * 0.46, 0.70 + y_off), (0.50 + sign * 0.31, 0.775 + y_off),
                      (0.50 + sign * 0.15, 0.795 + y_off), (0.50, 0.78 + y_off))

    def page(sign: float, y_off: float = 0.0):
        """One leaf. ``sign`` = -1 left, +1 right; ``y_off`` shifts the whole
        leaf down (used to lay the thickness leaf just beneath the top face)."""
        def P(dx, y):
            return (0.50 + sign * dx, y + y_off)
        return chain(
            # top edge: gutter-top -> outer-top corner, a gentle near-flat splay
            # (soft rise, no pointed "horn" at the corner)
            bezier(P(0.0, 0.46), P(0.16, 0.445), P(0.32, 0.430), P(0.46, 0.435)),
            # fore-edge: outer-top -> outer-bottom, softly rounded outer corner
            bezier(P(0.46, 0.435), P(0.485, 0.52), P(0.485, 0.615), P(0.46, 0.70)),
            # bottom edge: outer-bottom -> gutter-bottom, a shallow belly DOWN
            _bottom(sign, y_off) if y_off == 0.0 else
            bezier(P(0.46, 0.70), P(0.31, 0.775), P(0.15, 0.795), P(0.0, 0.78)),
            # spine: straight back up the gutter to the top
            bezier(P(0.0, 0.78), P(0.0, 0.46), P(0.0, 0.46), P(0.0, 0.46)),
        )

    # Fore-edge thickness: a thin leaf shifted down first (drawn UNDER the top
    # faces), so a sliver of it peeks below each page's bottom belly as the
    # stacked page edges.
    for s in (-1.0, +1.0):
        p.poly(page(s, y_off=0.032))
    for s in (-1.0, +1.0):
        p.poly(page(s))

    # Dark hairline along each page's bottom belly, so the thickness sliver
    # reads as separate stacked page edges rather than merging into the face.
    for s in (-1.0, +1.0):
        p.line(_bottom(s), 0.005, fill=0)

    # Spine crease (the gutter valley) + a few ruled text lines per page. Each
    # line runs from the outer edge (a hair higher, following the splay) in to
    # the gutter, so the ruling sits believably on the tilted leaf.
    p.line([st, sb], 0.008, fill=0)
    for yy in (0.500, 0.555, 0.610, 0.665):
        p.line([(0.14, yy - 0.022), (0.45, yy)], 0.006, fill=0)   # left page
        p.line([(0.55, yy), (0.86, yy - 0.022)], 0.006, fill=0)   # right page


# Trace recipe for the feather. threshold 0.92 segments the light-grey feather
# off its near-white ground (the PD source has no usable alpha); flood_bg peels
# the ground, fill_holes closes the vane, and the litho floor thickens the nib.
_QUILL_TRACE = TraceParams(
    use_alpha=False,        # source is opaque grey-on-white, not a clean alpha
    threshold=0.92,
    flood_bg=True,
    fill_holes=True,
    min_speck_frac=0.0008,
    min_feature_px=3,       # nib/shaft must clear the litho floor
    max_thicken_px=5,       # blob guard
    fill=0.90,              # feather's long dim ~= 90% of the grid before rotate
    anchor_x=0.5,
    anchor_y=0.5,
)
_QUILL_ROT_DEG = -22.0      # lay the (near-vertical) source feather onto a diagonal
_QUILL_AX, _QUILL_AY = 0.50, 0.46   # bbox-center placement over the book
_HALO_FRAC = 0.013          # dark separation channel width (fraction of grid)


def _placed_feather(n: int) -> np.ndarray:
    """Trace the PD feather, rotate it onto the writing diagonal, and re-center
    its bbox at the target anchor on an ``n`` x ``n`` grid."""
    m = trace_silhouette(_QUILL_REF, n_grid=n, params=_QUILL_TRACE)
    if _QUILL_ROT_DEG:
        rot = Image.fromarray((m * 255).astype(np.uint8)).rotate(
            _QUILL_ROT_DEG, resample=Image.BILINEAR, expand=True
        )
        big = np.asarray(rot) > 127
        H, W = big.shape
        canvas = np.zeros((n, n), dtype=bool)
        oy, ox = (n - H) // 2, (n - W) // 2
        sy0, sx0 = max(0, -oy), max(0, -ox)
        dy0, dx0 = max(0, oy), max(0, ox)
        hh, ww = min(H - sy0, n - dy0), min(W - sx0, n - dx0)
        canvas[dy0:dy0 + hh, dx0:dx0 + ww] = big[sy0:sy0 + hh, sx0:sx0 + ww]
        m = canvas
    ys, xs = np.where(m)
    if xs.size == 0:
        return m
    cx = (int(xs.min()) + int(xs.max())) / 2
    cy = (int(ys.min()) + int(ys.max())) / 2
    m = np.roll(
        m, (int(round(_QUILL_AY * n - cy)), int(round(_QUILL_AX * n - cx))),
        axis=(0, 1),
    )
    return m


def _feather_texture(feat: np.ndarray, n: int) -> np.ndarray:
    """Carve a rachis + swept barb slits into the vane (negative space) so the
    traced feather reads as barbed, not a solid gold sliver.

    The feather's long axis is found by PCA of the mask (robust to the exact
    rotation); slits run perpendicular to it, on both sides, swept toward the
    tip, only along the vane (upper ~2/3, skipping the bare shaft). Slits are
    intersected with an *eroded* vane so they never touch the outer contour
    (which would nick a boundary speck below the litho floor)."""
    from scipy import ndimage as ndi

    ys, xs = np.where(feat)
    if xs.size == 0:
        return np.zeros_like(feat)
    cx, cy = xs.mean(), ys.mean()
    pts = np.stack([xs - cx, ys - cy]).astype(float)
    cov = pts @ pts.T / pts.shape[1]
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]           # unit long-axis (dx, dy)
    perp = np.array([-axis[1], axis[0]])
    proj = pts.T @ axis
    pmin, pmax = float(proj.min()), float(proj.max())

    tex = Image.new("L", (n, n), 0)
    dt = ImageDraw.Draw(tex)
    a0 = (cx + axis[0] * pmin, cy + axis[1] * pmin)
    a1 = (cx + axis[0] * pmax, cy + axis[1] * pmax)
    dt.line([a0, a1], fill=255, width=max(1, n // 200))   # rachis
    nslit = 42
    for i in range(nslit):
        t = i / (nslit - 1)
        if t < 0.34:                          # skip the bare shaft third
            continue
        s = pmin + (pmax - pmin) * t
        bx, by = cx + axis[0] * s, cy + axis[1] * s
        blen = n * 0.11 * math.sin(math.pi * ((t - 0.34) / 0.66)) ** 0.7
        sweep = 0.35                          # barbs lean toward the tip
        for side in (+1, -1):
            ex = bx + side * perp[0] * blen + axis[0] * blen * sweep
            ey = by + side * perp[1] * blen + axis[1] * blen * sweep
            dt.line([(bx, by), (ex, ey)], fill=255, width=max(1, n // 320))
    slit = np.asarray(tex) > 127
    slit &= ndi.binary_erosion(feat, iterations=max(1, n // 220))
    return slit


def _despeckle(mask: np.ndarray, min_px: int = 12) -> np.ndarray:
    """Drop tiny disconnected gold islands (rasterization aliasing).

    Negative-space cuts that graze a silhouette boundary can slice off 1–2 px
    specks. On a gold-on-quartz plate any sub-2um island is a fabrication-risk
    defect (and reads as dirt), so we strip connected components below
    ``min_px``. Cheap, grid-local, and it protects against future aliasing as
    the artwork evolves — not a substitute for authoring clean geometry.
    """
    from scipy import ndimage

    lab, ncc = ndimage.label(mask)
    if ncc <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False  # background label
    return keep[lab]


def gear_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid — a bold centered spur gear (engineer emblem)."""
    del extent_um
    return _despeckle(render_silhouette(_draw_gear_only, n_grid))


def quill_book_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid — a bold centered open book with a traced quill laid
    diagonally across it (historian emblem).

    Compositing order: (1) draw the open book, (2) trace + place the PD feather,
    (3) carve a dark separation channel (dilated feather punched to background)
    so the gold feather never melts into the gold pages, (4) lay the feather
    gold on top, (5) carve rachis + barb slits into the vane. The feather is a
    genuine writing quill traced from a public-domain source (see
    ``lab/refs/LICENSE_NOTE.md``), which reads unmistakably as a quill where the
    earlier hand-drawn vane read as a leaf.
    """
    from scipy import ndimage as ndi

    del extent_um  # scale-free; caller controls cell_um to hit the extent
    n = max(64, int(n_grid))

    book = render_silhouette(_draw_open_book, n)
    feat = _placed_feather(n)
    halo = ndi.binary_dilation(feat, iterations=max(2, int(n * _HALO_FRAC)))

    out = book.copy()
    out[halo] = False          # dark channel around the quill
    out[feat] = True           # quill gold on top
    out[_feather_texture(feat, n)] = False   # rachis + barb slits
    return _despeckle(out)
