from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Hidden cursive inscription — a single centred line of formal script.
#
# This is the *bottom* of the ring box: a private line the couple reads when
# they pick the box up, engraved the way a jeweller inscribes the inside of a
# band. Default text is "J & P · 2026" (the engagement year), but the text is
# authored from editable parts so the UI can change the date later without a
# code edit (see ``artistic/inscription_line.py`` for the ParamSpec plumbing).
#
# Rendered in the same vendored Great Vibes (OFL) copperplate face as the J+P
# monogram so the whole box speaks one hand. Unlike the monogram, this is a
# *line of running text*, so we draw the whole string in one pass (the letters
# should read as connected script, not interlock) and centre it in a wide, low
# band with comfortable margins top and bottom.
#
# Litho survival: script hairlines and the thin joins between letters fall
# below the 2 px (== 2 um at the plate cell pitch) minimum feature after
# scaling. We thicken with the same morphological dilation the monogram uses,
# sized so the *thinnest* surviving stroke clears the floor but no more. The
# bottom plate is 50x50 mm, so there is plenty of room to set the line large
# enough that thickening barely has to touch it.
#
# The silhouette is scale-free: authored in a normalized art box, rasterized to
# the requested ``n_grid`` (a SQUARE grid so the plate compositor can dispatch
# it interchangeably with the colibrí / globe / monogram motifs), with the text
# occupying a horizontal band across the middle and blank margins above/below.
# ---------------------------------------------------------------------------

_FONT_PATH = Path(__file__).parent / "fonts" / "GreatVibes-Regular.ttf"

# Default inscription. The middle dot (U+00B7) reads as an elegant separator in
# Great Vibes; kept as the default so the year floats a touch apart from the
# initials. ``artistic/inscription_line.py`` composes this from editable parts.
DEFAULT_TEXT = "J & P · 2026"


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a square structuring element of ``radius`` px.

    Same vectorized numpy-shift dilation the monogram uses to lift hairline
    script joins to the litho minimum feature; ``radius`` is in working-grid
    pixels, so callers size it from the target cell pitch.
    """
    if radius <= 0:
        return mask
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(mask)
            ys0, ys1 = max(0, dy), min(h, h + dy)
            xs0, xs1 = max(0, dx), min(w, w + dx)
            yd0, yd1 = max(0, -dy), min(h, h - dy)
            xd0, xd1 = max(0, -dx), min(w, w - dx)
            shifted[yd0:yd1, xd0:xd1] = mask[ys0:ys1, xs0:xs1]
            out |= shifted
    return out


def _min_run(mask: np.ndarray) -> int:
    """Thinnest stroke proxy: smallest nonzero run of True cells over all rows
    and columns. Cheap min-feature estimate for sizing the safety dilation."""
    best = 10**9
    for axis_mask in (mask, mask.T):
        for row in axis_mask:
            run = 0
            for v in row:
                if v:
                    run += 1
                elif run:
                    if run < best:
                        best = run
                    run = 0
            if run and run < best:
                best = run
    return 0 if best == 10**9 else best


def _draw_heart(draw: "ImageDraw.ImageDraw", cx: float, cy: float, r: float) -> None:
    """Fill a small symmetric heart centred at (cx, cy) with lobe radius ~r.

    A tiny optional flourish set between the initials and the date. Built from
    two circles (the lobes) plus a downward triangle (the point) so it reads as
    a heart at the coarse plate resolution without needing a bezier path.
    """
    # Two top lobes.
    lobe = r * 0.6
    lx = cx - r * 0.5
    rx = cx + r * 0.5
    ly = cy - r * 0.25
    draw.ellipse([lx - lobe, ly - lobe, lx + lobe, ly + lobe], fill=255)
    draw.ellipse([rx - lobe, ly - lobe, rx + lobe, ly + lobe], fill=255)
    # Bottom point — triangle from the outer edges of the lobes down to a tip.
    draw.polygon(
        [
            (cx - r * 1.08, cy - r * 0.18),
            (cx + r * 1.08, cy - r * 0.18),
            (cx, cy + r * 1.15),
        ],
        fill=255,
    )


def inscription_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
    text: str = DEFAULT_TEXT,
    heart: bool = True,
    band_frac: float = 0.42,
    margin_frac: float = 0.09,
    min_stroke_px: int = 2,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — a centred cursive inscription line.

    The ``text`` is drawn in one pass in Great Vibes, scaled to fill
    ``1 - 2*margin_frac`` of the grid width (whichever of width/height binds
    first, so a long string never overflows the plate), and vertically centred
    in a horizontal band ``band_frac`` tall with blank margins above and below —
    the elegant "single line on an open field" look for the hidden bottom.

    ``heart=True`` slips a small heart flourish into the widest interior gap of
    the string (typically between the initials and the date) — off it if the
    text has no natural gap. ``extent_um`` is accepted for signature parity with
    the other centrepiece motifs and ignored (the silhouette is scale-free).
    """
    del extent_um  # scale-free; caller controls cell pitch to hit the extent

    n = max(64, int(n_grid))
    # Supersample so LANCZOS downscaling keeps the copperplate crisp. Cap the
    # working grid to stay light on the box (square, matches the plate paste).
    work = min(1024, max(384, n * 2))

    # Layout budget in working pixels.
    margin_px = int(round(work * margin_frac))
    avail_w = work - 2 * margin_px
    band_px = int(round(work * band_frac))

    # Find a font size whose rendered string fits avail_w AND band_px. Great
    # Vibes' long swashes overshoot the nominal cap height, so measure the true
    # inked bbox and binary-search the point size down until it fits both ways.
    canvas = work
    lo, hi = 8, int(work * 0.9)
    best_px = lo

    def _measure(font_px: int) -> tuple[int, int, tuple[int, int, int, int]] | None:
        font = ImageFont.truetype(str(_FONT_PATH), font_px)
        img = Image.new("L", (canvas, canvas), 0)
        d = ImageDraw.Draw(img)
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        return r - l, b - t, (l, t, r, b)

    while lo <= hi:
        mid = (lo + hi) // 2
        gw, gh, _ = _measure(mid)  # type: ignore[misc]
        if gw <= avail_w and gh <= band_px:
            best_px = mid
            lo = mid + 1
        else:
            hi = mid - 1

    font = ImageFont.truetype(str(_FONT_PATH), best_px)
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    gw, gh = r - l, b - t
    # Centre the inked bbox horizontally and vertically in the grid.
    ox = (canvas - gw) // 2 - l
    oy = (canvas - gh) // 2 - t
    draw.text((ox, oy), text, fill=255, font=font)

    if heart:
        # Drop a small heart into the widest interior column gap of the inked
        # string — for "J & P · 2026" that lands between the "P" and the date,
        # exactly where an engraver would set a device. We scan the current
        # inked image column occupancy to find the widest blank interior run.
        arr = np.asarray(img, dtype=np.uint8) > 127
        col_any = arr.any(axis=0)
        xs = np.where(col_any)[0]
        if xs.size:
            x0, x1 = int(xs.min()), int(xs.max())
            # widest run of blank columns strictly inside [x0, x1]
            best_gap = (0, 0, -1)  # (width, start, end)
            run_start = None
            for x in range(x0, x1 + 1):
                if not col_any[x]:
                    if run_start is None:
                        run_start = x
                elif run_start is not None:
                    w_run = x - run_start
                    if w_run > best_gap[0]:
                        best_gap = (w_run, run_start, x - 1)
                    run_start = None
            gap_w, gs, ge = best_gap
            # Only place if there is a comfortably wide gap (avoid crowding).
            if gap_w > work * 0.045:
                hcx = (gs + ge) / 2.0
                hcy = canvas / 2.0
                hr = min(gap_w * 0.32, band_px * 0.16)
                if hr >= 3:
                    _draw_heart(draw, hcx, hcy, hr)

    dst = np.asarray(img, dtype=np.uint8) > 127

    # Downsample to the requested grid FIRST so the min-feature test and the
    # dilation are measured in output pixels (== output cell pitch).
    if work != n:
        dst = np.asarray(
            Image.fromarray(dst.astype(np.uint8) * 255, "L").resize(
                (n, n), Image.LANCZOS
            )
        ) > 100

    # Litho thickening: dilate until the thinnest stroke clears min_stroke_px.
    thin = _min_run(dst)
    if 0 < thin < min_stroke_px:
        rr = max(1, (min_stroke_px - thin + 1) // 2)
        dst = _dilate(dst, rr)

    return dst
