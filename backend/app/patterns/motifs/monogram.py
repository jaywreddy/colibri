from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Cursive J+P interlocked wedding monogram.
#
# A classic engagement monogram: a large flowing capital ``J`` and capital
# ``P`` rendered in a formal script face (Great Vibes, OFL) and set so their
# central zones OVERLAP and their swashes interlace — the way an engraver would
# entwine a couple's initials on a signet or a wedding invitation.
#
# The two letters are rasterized SEPARATELY (each in its own layer at high
# resolution), positioned with a controllable horizontal overlap, then the two
# binary masks are unioned. Rendering separately (rather than drawing the string
# "JP") is what lets the strokes cross *through* each other and read as
# interlocked rather than merely adjacent.
#
# Litho survival: script capitals have very thin hairline joins. After scaling
# to the composed plate grid those joins can fall below the 2 px (== 2 um at the
# plate's cell pitch) minimum feature. We thicken with a morphological dilation
# (MaxFilter equivalent) sized so the *thinnest* surviving stroke clears the
# 2 px floor, but no more — over-thickening turns elegant script into a blob.
#
# Everything is authored in a normalized 0..1 art box and rasterized to the
# requested ``n_grid``; the silhouette is scale-free (the caller controls the
# cell pitch that maps grid cells to micrometers), exactly like the colibri and
# globe motifs so the plate compositor can dispatch it interchangeably.
# ---------------------------------------------------------------------------

_FONT_PATH = Path(__file__).parent / "fonts" / "GreatVibes-Regular.ttf"


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a square structuring element of the given pixel radius.

    Equivalent to PIL's ``MaxFilter`` on the mask but vectorized with numpy
    shifts so it stays cheap and dependency-free. Used to thicken hairline
    script joins up to the litho minimum feature; ``radius`` is in pixels of the
    working grid, so callers size it from the target cell pitch.
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
    """Approximate the thinnest stroke: the smallest nonzero run of True cells
    across all rows and columns. Cheap proxy for min feature width used to size
    the litho-safety dilation."""
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


class _Glyph:
    """A rasterized glyph plus the metrics needed to size it by its LETTER BODY.

    ``mask``/``bbox`` are the swash-inclusive inked pixels (used for the final
    union crop). ``cap_top``/``baseline`` bracket the glyph's core body — the
    baseline is the font's typographic baseline (where the row sits) and
    ``cap_top`` is the top of the capital body EXCLUDING the extreme entry/exit
    swashes. ``cap_height = baseline - cap_top`` is the stable metric we scale
    on so J and P read the same visual size regardless of how far Great Vibes'
    swashes overshoot the body.
    """

    __slots__ = ("mask", "bbox", "cap_top", "baseline")

    def __init__(
        self,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        cap_top: int,
        baseline: int,
    ) -> None:
        self.mask = mask
        self.bbox = bbox
        self.cap_top = cap_top
        self.baseline = baseline

    @property
    def cap_height(self) -> int:
        return max(1, self.baseline - self.cap_top)


def _body_cap_top(arr: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> int:
    """Estimate the top of the letter's CORE BODY, excluding the thin entry
    swash that copperplate capitals fling up above the cap line.

    Script capitals begin with a hairline flourish that spikes well above the
    body; measuring cap height from the topmost inked pixel therefore lets that
    flourish, not the letter, drive the scale. We walk down from the top and
    take the first row that is "solidly body" — its inked width reaches a
    fraction of the glyph's widest row — as the effective cap top. That ignores
    the 1-2 px hairline overshoot while still landing on the true top of the
    bowl/stem.
    """
    h = y1 - y0
    band = arr[y0:y1]
    row_ink = band.sum(axis=1).astype(np.int64)
    if not row_ink.any():
        return y0
    widest = int(row_ink.max())
    # A row counts as "body" once it carries a real chunk of horizontal ink,
    # not just a hairline swash pixel. 22% of the widest row is comfortably
    # above hairline noise yet below the bowl width.
    thr = max(2, int(widest * 0.22))
    for i in range(h):
        if row_ink[i] >= thr:
            return y0 + i
    return y0


def _render_glyph(letter: str, canvas: int, font_px: int) -> _Glyph:
    """Render a single glyph anchored on its baseline near the canvas center.

    Returns a ``_Glyph`` carrying the swash-inclusive mask/bbox plus the
    baseline and body cap-top so the caller can normalize each letter by its
    core body height (not its swash-inflated bbox).
    """
    font = ImageFont.truetype(str(_FONT_PATH), font_px)
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    # Baseline anchor ('ls' = left / baseline) puts the typographic baseline at
    # a KNOWN y, so cap height and descender depth are directly measurable.
    baseline = canvas // 2
    draw.text((canvas // 6, baseline), letter, fill=255, font=font, anchor="ls")
    arr = np.asarray(img, dtype=np.uint8) > 127
    ys, xs = np.where(arr)
    if len(xs) == 0:
        return _Glyph(arr, (0, 0, canvas, canvas), 0, baseline)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    cap_top = _body_cap_top(arr, x0, x1, y0, y1)
    return _Glyph(arr, (x0, y0, x1, y1), cap_top, baseline)


def _place(
    glyph: _Glyph,
    dst: np.ndarray,
    body_center_x: float,
    baseline_y: float,
    cap_px: float,
) -> tuple[float, float]:
    """Scale ``glyph`` so its CORE BODY (cap_top→baseline) is ``cap_px`` pixels
    tall, then OR it into ``dst`` aligned on the baseline.

    Because the scale is derived from ``cap_height`` — not the swash-inflated
    bbox — two letters placed with the same ``cap_px`` read as the SAME visual
    size even though one (the J) drags a long tail below the baseline and a tall
    entry swash above the body. ``body_center_x`` is the fractional-x where the
    glyph's inked bbox center should land; ``baseline_y`` is the fractional-y of
    the shared baseline. Returns the placed ``(left_x, right_x)`` bbox edges as
    fractions of the destination width so the caller can verify the interlock.
    """
    x0, y0, x1, y1 = glyph.bbox
    gw, gh = x1 - x0, y1 - y0
    if gw <= 0 or gh <= 0:
        return (body_center_x, body_center_x)
    scale = cap_px / glyph.cap_height
    crop = glyph.mask[y0:y1, x0:x1]
    new_w = max(1, int(round(gw * scale)))
    new_h = max(1, int(round(gh * scale)))
    scaled = np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255, "L").resize(
            (new_w, new_h), Image.LANCZOS
        )
    ) > 127
    H, W = dst.shape
    # Horizontal: center the inked bbox on body_center_x.
    tx = int(round(body_center_x * W - new_w / 2))
    # Vertical: pin the glyph's baseline (a row inside the crop) onto baseline_y.
    baseline_in_crop = (glyph.baseline - y0) * scale
    ty = int(round(baseline_y * H - baseline_in_crop))
    sx0, sy0 = max(0, -tx), max(0, -ty)
    dx0, dy0 = max(0, tx), max(0, ty)
    cw = min(new_w - sx0, W - dx0)
    ch = min(new_h - sy0, H - dy0)
    if cw > 0 and ch > 0:
        dst[dy0:dy0 + ch, dx0:dx0 + cw] |= scaled[sy0:sy0 + ch, sx0:sx0 + cw]
    return (tx / W, (tx + new_w) / W)


def _recenter(mask: np.ndarray, fill: float = 0.82) -> np.ndarray:
    """Crop the union to its tight bbox and re-paste it centered, scaled so the
    larger dimension fills ``fill`` of the grid. Guarantees the composite art is
    centered and consistently sized regardless of per-glyph swash asymmetry."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    crop = mask[y0:y1, x0:x1]
    ch, cw = crop.shape
    H, W = mask.shape
    target = fill * min(H, W)
    s = target / max(ch, cw)
    nw, nh = max(1, int(round(cw * s))), max(1, int(round(ch * s)))
    scaled = np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255, "L").resize((nw, nh), Image.LANCZOS)
    ) > 127
    out = np.zeros_like(mask)
    tx, ty = (W - nw) // 2, (H - nh) // 2
    out[ty:ty + nh, tx:tx + nw] = scaled
    return out


def monogram_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
    letters: str = "JP",
    overlap: float = 0.68,
    surround: bool = False,
    min_stroke_px: int = 2,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — interlocked cursive monogram.

    Renders each of the two ``letters`` separately in Great Vibes, scales and
    positions them so their central zones overlap by ``overlap`` (fraction of a
    letter's width), unions the two masks, then dilates so the thinnest surviving
    stroke clears ``min_stroke_px`` (litho minimum feature). ``surround=True``
    adds a thin oval frame around the pair (off by default). ``extent_um`` is
    accepted for signature parity with the other centerpiece motifs and ignored
    (the silhouette is scale-free).
    """
    del extent_um  # scale-free; caller controls cell pitch to hit the extent

    n = max(64, int(n_grid))
    # Work at a generous supersample so LANCZOS scaling keeps the script crisp;
    # downsample to n at the end. Cap the working grid to stay light on the box.
    work = min(1024, max(384, n * 2))

    a, b = (letters + "JP")[0], (letters + "JP")[1]

    # Render each glyph on a canvas MUCH larger than the glyph so Great Vibes'
    # long entry/exit swashes are never clipped. Each returns its swash-inclusive
    # bbox PLUS its body cap-top and baseline so we can scale by the LETTER BODY.
    canvas = work
    font_px = int(canvas * 0.42)
    glyph_a = _render_glyph(a, canvas, font_px)
    glyph_b = _render_glyph(b, canvas, font_px)

    dst = np.zeros((work, work), dtype=bool)

    # Interlock layout — the fix for "the J prints bigger than the P".
    #
    # ROOT CAUSE of the old imbalance: both glyphs were scaled by the same
    # factor applied to their swash-INCLUSIVE bbox, then the union was recentred
    # to fill the grid. Great Vibes' J carries a huge descender tail + tall entry
    # swash, so its bbox towered over the P's compact bbox; equal-factor scaling
    # therefore blew the J's BODY up relative to the P's.
    #
    # FIX: scale each glyph so its CORE BODY (cap_top→baseline, measured
    # excluding the hairline swash overshoot) is the same pixel height, and align
    # both on a SHARED baseline. The J's tail then hangs below that baseline and
    # its entry swash rises above — decoration, not size — while the two capital
    # bodies read identically. A tiny optical nudge (P set 2% taller) compensates
    # for the J's visually heavier bowl+tail mass so they balance to the eye.
    cap_px = work * 0.34  # core-body height target, in working pixels
    sep = (1.0 - overlap) * 0.5  # each body center offset from the midline
    baseline_y = 0.60  # shared baseline low enough to leave room for the J tail
    # OPTICAL-WEIGHT BALANCE (supersedes the earlier cap-height overcorrection).
    #
    # Great Vibes' P and J cannot match on BOTH body height and body area: at
    # equal cap height the P's fat bowl carries ~40% more ink than the J's slim
    # stem, while the J's tall entry swash + long descender make it sprawl taller
    # overall. The previous fix leaned hard on cap height (J×1.04, P×0.93) and
    # overcorrected — the P ended up reading distinctly SMALLER than the J.
    #
    # The measured sweet spot is a near-equal cap height with only a whisper of
    # correction: J a hair taller, P a hair shorter. That lands the two BODY
    # heights within ~4% and the full inked footprints within ~5% (the J's tail
    # mass offsetting the P's heavier bowl), so the pair reads as the SAME visual
    # size. The P also gets a little extra rightward clearance so its bowl does
    # not swallow the J's stem at the interlock.
    _place(glyph_a, dst, body_center_x=0.5 - sep, baseline_y=baseline_y, cap_px=cap_px * 1.02)
    _place(glyph_b, dst, body_center_x=0.5 + sep * 1.10, baseline_y=baseline_y, cap_px=cap_px * 0.98)

    dst = _recenter(dst)

    # Downsample to the requested grid FIRST (so the min-feature test and the
    # dilation are measured in *output* pixels == output cell pitch).
    if work != n:
        dst = np.asarray(
            Image.fromarray(dst.astype(np.uint8) * 255, "L").resize(
                (n, n), Image.LANCZOS
            )
        ) > 100

    # Litho thickening: dilate until the thinnest stroke clears min_stroke_px.
    thin = _min_run(dst)
    if 0 < thin < min_stroke_px:
        r = max(1, (min_stroke_px - thin + 1) // 2)
        dst = _dilate(dst, r)

    if surround:
        dst = _add_surround(dst, n, min_stroke_px)

    return dst


def _add_surround(mask: np.ndarray, n: int, min_stroke_px: int) -> np.ndarray:
    """Add a thin oval frame around the monogram (optional).

    The ring width is clamped to at least ``min_stroke_px`` so the frame itself
    is litho-safe (a 1 px outline would print below the 2 um minimum line).
    """
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    draw = ImageDraw.Draw(img)
    pad = int(n * 0.05)
    ring_w = max(min_stroke_px, int(n * 0.014))
    for i in range(ring_w):
        draw.ellipse(
            [pad + i, pad + i, n - 1 - pad - i, n - 1 - pad - i], outline=255
        )
    return np.asarray(img, dtype=np.uint8) > 127
