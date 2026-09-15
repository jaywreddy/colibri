from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageDraw
from ._pillow import check_silhouette_budget, render_silhouette

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
    # Litho weight: Great Vibes' hairlines are ~2% of the font size, which on
    # a 17 mm lid is a 0.3 mm line the eye reads as a scratch next to the
    # 1 mm stems. A stroke outline of ~0.45% of the font size (2 px at the
    # working 430 px) thickens hairlines by half and the stems by an eighth,
    # the weight of an engraver's copperplate rather than a pen's.
    sw = max(1, int(round(font_px * 0.0045)))
    draw.text((canvas // 6, baseline), letter, fill=255, font=font, anchor="ls",
              stroke_width=sw, stroke_fill=255)
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


def _recenter_transform(
    mask: np.ndarray, fill: float = 0.92
) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int]] | None:
    """The crop/scale/offset :func:`_recenter` would apply to ``mask``.

    Split out so the SAME transform can be replayed on each letter layer
    (``_recenter_layers``): the region map needs per-letter masks that land
    exactly where the union silhouette lands, and a transform re-derived from a
    single letter's own bbox would not.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    ch, cw = y1 - y0, x1 - x0
    H, W = mask.shape
    target = fill * min(H, W)
    s = target / max(ch, cw)
    nw, nh = max(1, int(round(cw * s))), max(1, int(round(ch * s)))
    return (x0, x1, y0, y1), (nw, nh), ((W - nw) // 2, (H - nh) // 2)


def _apply_recenter(
    layer: np.ndarray,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    offset: tuple[int, int],
) -> np.ndarray:
    x0, x1, y0, y1 = box
    nw, nh = size
    tx, ty = offset
    crop = layer[y0:y1, x0:x1]
    scaled = np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255, "L").resize((nw, nh), Image.LANCZOS)
    ) > 127
    out = np.zeros_like(layer)
    out[ty:ty + nh, tx:tx + nw] = scaled
    return out


def _recenter(mask: np.ndarray, fill: float = 0.92) -> np.ndarray:
    """Crop the union to its tight bbox and re-paste it centered, scaled so the
    larger dimension fills ``fill`` of the grid. Guarantees the composite art is
    centered and consistently sized regardless of per-glyph swash asymmetry."""
    t = _recenter_transform(mask, fill)
    if t is None:
        return mask
    return _apply_recenter(mask, *t)


def _recenter_layers(
    layers: list[np.ndarray], fill: float = 0.92
) -> list[np.ndarray]:
    """Recenter every layer by the transform their UNION would get, so the
    layers stay in register with each other and with the union silhouette."""
    union = layers[0].copy()
    for m in layers[1:]:
        union |= m
    t = _recenter_transform(union, fill)
    if t is None:
        return layers
    return [_apply_recenter(m, *t) for m in layers]


def _compose_letters(
    work: int, letters: str, overlap: float
) -> tuple[np.ndarray, np.ndarray]:
    """The two glyph layers, placed and interlocked, at the working resolution.

    Split out of :func:`monogram_silhouette` so the region map can get the
    letters SEPARATELY (which letter owns which pixel is the whole point of a
    two-colour monogram) while the silhouette callers keep the identical union:
    the silhouette is exactly ``a | b`` of what this returns.

    Interlock layout — the fix for "the J prints bigger than the P".

    ROOT CAUSE of the old imbalance: both glyphs were scaled by the same factor
    applied to their swash-INCLUSIVE bbox, then the union was recentred to fill
    the grid. Great Vibes' J carries a huge descender tail + tall entry swash,
    so its bbox towered over the P's compact bbox; equal-factor scaling
    therefore blew the J's BODY up relative to the P's.

    FIX: scale each glyph so its CORE BODY (cap_top→baseline, measured
    excluding the hairline swash overshoot) is the same pixel height, and align
    both on a SHARED baseline. The J's tail then hangs below that baseline and
    its entry swash rises above — decoration, not size — while the two capital
    bodies read identically.

    OPTICAL-WEIGHT BALANCE (supersedes the earlier cap-height overcorrection).
    Great Vibes' P and J cannot match on BOTH body height and body area: at
    equal cap height the P's fat bowl carries ~40% more ink than the J's slim
    stem, while the J's tall entry swash + long descender make it sprawl taller
    overall. The previous fix leaned hard on cap height (J×1.04, P×0.93) and
    overcorrected — the P ended up reading distinctly SMALLER than the J. The
    measured sweet spot is a near-equal cap height with only a whisper of
    correction: J a hair taller, P a hair shorter. That lands the two BODY
    heights within ~4% and the full inked footprints within ~5% (the J's tail
    mass offsetting the P's heavier bowl), so the pair reads as the SAME visual
    size. The P also gets a little extra rightward clearance so its bowl does
    not swallow the J's stem at the interlock.
    """
    a, b = (letters + "JP")[0], (letters + "JP")[1]

    # Render each glyph on a canvas MUCH larger than the glyph so Great Vibes'
    # long entry/exit swashes are never clipped. Each returns its swash-inclusive
    # bbox PLUS its body cap-top and baseline so we can scale by the LETTER BODY.
    canvas = work
    font_px = int(canvas * 0.42)
    glyph_a = _render_glyph(a, canvas, font_px)
    glyph_b = _render_glyph(b, canvas, font_px)

    cap_px = work * 0.34  # core-body height target, in working pixels
    sep = (1.0 - overlap) * 0.5  # each body center offset from the midline
    baseline_y = 0.60  # shared baseline low enough to leave room for the J tail

    layer_a = np.zeros((work, work), dtype=bool)
    layer_b = np.zeros((work, work), dtype=bool)
    _place(glyph_a, layer_a, body_center_x=0.5 - sep, baseline_y=baseline_y,
           cap_px=cap_px * 1.02)
    _place(glyph_b, layer_b, body_center_x=0.5 + sep * 1.10, baseline_y=baseline_y,
           cap_px=cap_px * 0.98)
    return layer_a, layer_b


def monogram_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
    letters: str = "JP",
    overlap: float = 0.76,
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
    # Own raster path (not render_silhouette), so gate n here too — callers size
    # n_grid straight off unvalidated extent_um/period params.
    check_silhouette_budget(n, "Cursive monogram silhouette")
    # Work at a generous supersample so LANCZOS scaling keeps the script crisp;
    # downsample to n at the end. Cap the working grid to stay light on the box.
    work = min(1024, max(384, n * 2))

    layer_a, layer_b = _compose_letters(work, letters, overlap)
    dst = layer_a | layer_b

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


# ---------------------------------------------------------------------------
# SINGLE-LAYER DIFFRACTION region map (app/region_art.py).
#
# On a one-ply face the monogram is not a silhouette filled with a carrier: it
# is a MAP OF REGIONS, each written as its own fine vertical 50 % grating whose
# PERIOD is the colour it flashes. Here that means one colour per LETTER — the
# J flashes at one rung of ``plates.SINGLE_PLY_LEAF_HUE_PERIODS_UM`` and the P
# at another, each uniform over the whole letter (Jay's rule: an effect applies
# uniformly over its motif, no partial patches or arbitrary strips).
# ---------------------------------------------------------------------------

JP_REGION_J = 1
JP_REGION_P = 2


def _square(radius: int) -> np.ndarray:
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)


def _grow(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a square element. ``scipy.ndimage`` rather than the
    shift loop in :func:`_dilate`: the region map runs up to 1400 cells and the
    weave radii reach 4, which is 80 whole-array shifts per call."""
    if radius <= 0:
        return mask
    from scipy import ndimage

    return ndimage.binary_dilation(mask, structure=_square(radius))


def _open(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological opening — drops specks and hairline tapers thinner than
    ``2·radius + 1`` cells while leaving every real stroke its full width."""
    if radius <= 0:
        return mask
    from scipy import ndimage

    return ndimage.binary_opening(mask, structure=_square(radius), border_value=0)


def _drop_crumbs(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Delete connected pieces smaller than ``min_area`` cells.

    The weave cut below splits the under-letter into a few large pieces (half a
    bowl, a stem segment) — and, where the two letters graze, the odd isolated
    crumb of a terminal. A crumb is not a letter and it is not a region: it is a
    speck of a third colour sitting beside the stroke that swallowed it."""
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_area
    keep[0] = False
    return keep[lab]


def _downsample(mask: np.ndarray, n: int) -> np.ndarray:
    if mask.shape[0] == n:
        return mask
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255, "L").resize((n, n), Image.LANCZOS)
    ) > 100


def _runs(mask: np.ndarray) -> np.ndarray:
    """Every nonzero run length in the mask, both axes, vectorised.

    ``_min_run`` above is a per-element Python loop — fine on a 256 px
    silhouette, ~3 M iterations on a 1400 px region map. This is the same
    measurement in numpy, and it returns the whole distribution because the
    MINIMUM run of an antialiased mask is always 1 (a corner pixel) and says
    nothing about the stroke.
    """
    out: list[np.ndarray] = []
    for m in (mask, mask.T):
        pad = np.zeros((m.shape[0], 1), dtype=bool)
        flat = np.concatenate([pad, m, pad], axis=1).ravel()
        d = np.diff(flat.astype(np.int8))
        starts = np.flatnonzero(d == 1)
        if starts.size:
            out.append(np.flatnonzero(d == -1) - starts)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _thin_stroke_px(mask: np.ndarray, pct: float = 10.0) -> float:
    """Width of the mask's THIN strokes, in cells: the ``pct``-th percentile of
    its run lengths. Every morphological radius below is a fraction of this, so
    the region map looks the SAME at every resolution it is asked for — and it
    is asked for several: the fine bake wants one cell per
    ``region_art.REGION_ZONE_PITCH_UM``, the composed preview and the period
    map want one cell per plate texel. A radius fixed in cells would weave the
    letters differently in the preview than in the part."""
    r = _runs(mask)
    return float(np.percentile(r, pct)) if r.size else 0.0


def monogram_regions(
    n_px: int,
    overlap: float = 0.76,
    letters: str = "JP",
    first_period_um: float = 4.47,
    second_period_um: float = 6.02,
    over: str = "first",
    weave_gap_frac: float = 0.20,
    weave_open_frac: float = 0.25,
    speck_open_frac: float = 0.12,
    crumb_frac: float = 1.2,
):
    """The single-layer diffraction map of the interlocked monogram.

    Returns a :class:`app.region_art.RegionArt` over the centrepiece square:
    label 1 is the whole FIRST letter at ``first_period_um``, label 2 the whole
    SECOND letter at ``second_period_um``. Under a lamp the two initials flash
    two different colours (at 4.47 vs 6.02 µm the first order of one is ~1.35×
    the wavelength of the other — a blue-green J against an orange-red P at any
    one tilt), and each letter is ONE colour over its whole length.

    WHY THERE IS NO THIRD "KNOT" REGION. The obvious extra region is the
    intersection of the two letters, so the interlock reads as its own colour.
    Measured on the real map (870 cells across the 17.4 mm lid art box, one
    cell = ``region_art.REGION_ZONE_PITCH_UM``), that intersection is 6.6 % of
    the ink but it is NOT a set of compact crossings: Great Vibes sets the P's
    bowl-top nearly TANGENT to the J's arch, so most of the intersection is a
    long tapering sliver whose 2nd-percentile run is 1.8 cells — under the
    one-cell glass gutter the emitter insets every region by, and far under the
    ~2 periods (12 µm) a region needs before it has a spectrum at all. A knot
    region would therefore be exactly the "arbitrary strip" the uniformity rule
    bans, and half of it would not survive to be written.

    Instead the overlap is resolved as an engraver's WEAVE: the ``over`` letter
    keeps its full stroke and the other is cut back clear of it by
    ``weave_gap_frac`` of a stroke width, so it visibly passes UNDER — a clean
    break plus the emitter's glass gutter, rather than a stroke shaved to a
    crescent. Where the two run TANGENT the cut alone would still leave the
    under-letter a hairline crescent alongside the over-letter, so the
    under-letter is then OPENED at ``weave_open_frac``: any remnant thinner
    than that is not part of the letter, it is the leftover of a near-miss, and
    it goes. ``speck_open_frac`` does the same job for the antialiasing specks
    and hairline tapers the LANCZOS downsample leaves.

    Every one of those three radii is a FRACTION of the map's own measured thin
    stroke (:func:`_thin_stroke_px`), never a fixed number of cells, because
    this function is called at several resolutions for the same part — see that
    docstring.
    """
    from ...region_art import Region, RegionArt

    n = max(32, int(n_px))
    check_silhouette_budget(n, "Cursive monogram region map")
    # Work well above the region grid so LANCZOS keeps the script crisp; the
    # region map runs up to REGION_ZONE_MAX_PX (1400), above the silhouette's
    # 1024 cap, so never resolve a letter by UPSAMPLING it.
    work = min(1408, max(384, n * 2))

    layer_a, layer_b = _compose_letters(work, letters, overlap)
    layer_a, layer_b = _recenter_layers([layer_a, layer_b])
    mask_a = _downsample(layer_a, n)
    mask_b = _downsample(layer_b, n)

    stroke = _thin_stroke_px(mask_a | mask_b)

    def _radius(frac: float) -> int:
        return max(1, int(round(frac * stroke)))

    mask_a = _open(mask_a, _radius(speck_open_frac))
    mask_b = _open(mask_b, _radius(speck_open_frac))

    gap, weave_open = _radius(weave_gap_frac), _radius(weave_open_frac)
    crumb = max(4, int(round((crumb_frac * max(1.0, stroke)) ** 2)))

    def _under(under: np.ndarray, over_mask: np.ndarray) -> np.ndarray:
        return _drop_crumbs(_open(under & ~_grow(over_mask, gap), weave_open), crumb)

    if over == "first":
        mask_b = _under(mask_b, mask_a)
    else:
        mask_a = _under(mask_a, mask_b)

    labels = np.zeros((n, n), dtype=np.int32)
    labels[mask_a] = JP_REGION_J
    labels[mask_b] = JP_REGION_P
    name_a, name_b = (letters + "JP")[0], (letters + "JP")[1]
    return RegionArt(
        labels,
        {
            JP_REGION_J: Region(name_a, float(first_period_um)),
            JP_REGION_P: Region(name_b, float(second_period_um)),
        },
    )


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


# ---------------------------------------------------------------------------
# Blocky keepsake silhouettes (J+P barrier switch + heart-globe) — used by
# monogram_phase.py and carrier_reveal.py. Kept alongside the Great Vibes
# script monogram above; the two APIs are disjoint.
# ---------------------------------------------------------------------------

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