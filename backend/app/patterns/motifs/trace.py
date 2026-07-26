from __future__ import annotations

"""Image -> litho-safe silhouette tracing submodule.

Turns a *raster reference image* (a photo, an engraving, or an already-clean
PhyloPic/Commons silhouette) into a binary bool grid on the same contract as the
hand-authored motifs (``colibri.py`` / ``globe.py`` / ``monogram.py``):

    bool ndarray, shape (n_grid, n_grid), True == gold, Pillow y-down.

The pipeline is deterministic and fully parameterized so a traced motif bakes to
the same mask every run (spec-hashable). Stages:

  1. **Load + normalize** — open, optional crop box, grayscale, optional contrast
     stretch. Alpha-carrying PNGs (PhyloPic silhouettes) use the alpha channel
     directly as the foreground signal, which is cleaner than luma thresholding.
  2. **Segment foreground** — Otsu threshold (auto) or a fixed threshold, with an
     ``invert`` flag for dark-on-light vs light-on-dark sources.
  3. **Background removal** — optional flood-fill from the image corners so a
     textured/paper background that survives thresholding is peeled away, leaving
     only the connected subject.
  4. **Morphological cleanup** — fill interior holes, drop connected specks below
     a min-area fraction (removes JPEG crumbs / stray marks).
  5. **Litho-thicken thin necks** — grow the mask just enough that its thinnest
     surviving run clears the min-feature floor (legs, tails), reusing the same
     ``_dilate`` / ``_min_run`` idea as ``monogram.py`` but via ``scipy.ndimage``
     for speed. Never over-thickens (blob guard).
  6. **Place into a centered art box** — crop to the tight subject bbox, scale so
     the subject fills a target fraction, and paste at a controllable anchor
     (e.g. belly low for a waterline composition). Optional horizontal flip so
     the subject faces a chosen direction.

The heavy binary morphology uses ``scipy.ndimage`` (already a project dep). No
GEOS / shapely booleans are involved, so this is off the geometry hot-path
entirely.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from ._pillow import check_silhouette_budget


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TraceParams:
    """Deterministic knobs for :func:`trace_silhouette`.

    All fractions are of the *working grid* unless noted. Defaults are tuned for
    a clean, already-silhouetted PhyloPic PNG (alpha or solid black on white).
    """

    # -- load / normalize --
    crop: tuple[float, float, float, float] | None = None  # (x0,y0,x1,y1) frac
    use_alpha: bool = True          # prefer the alpha channel if the image has one
    invert: bool = False            # True if subject is LIGHT on a DARK ground
    threshold: float | None = None  # 0..1 fixed cut; None -> Otsu (auto)
    contrast: float = 1.0           # >1 stretches luma contrast before threshold

    # -- background removal --
    flood_bg: bool = True           # peel a connected background from the corners

    # -- cleanup --
    fill_holes: bool = True         # fill interior background pockets
    min_speck_frac: float = 0.002   # drop components smaller than this*area

    # -- litho safety --
    min_feature_px: int = 3         # thinnest surviving run must clear this
    max_thicken_px: int = 6         # dilation cap (blob guard)

    # -- placement into the art box --
    fill: float = 0.86              # subject's larger dim as a fraction of grid
    anchor_x: float = 0.5           # subject-bbox center x in the art box
    anchor_y: float = 0.5           # subject-bbox center y in the art box
    flip_h: bool = False            # mirror left<->right (face a chosen way)

    # -- optional engraved negative-space marks (art-box fractions) --
    # Each is (cx, cy, rx, ry): an ellipse punched to background AFTER placement,
    # used to add an eye / nostril so the read is "animal" not "blob".
    holes: tuple[tuple[float, float, float, float], ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Stage helpers
# --------------------------------------------------------------------------- #
def _load_gray_alpha(path: Path, crop, work: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Load ``path`` at ``work`` x ``work``. Returns (gray 0..1, alpha 0..1|None).

    Preserves aspect by pasting onto a square transparent/ white canvas, so the
    subject is never distorted before we measure its bbox.
    """
    img = Image.open(path)
    has_alpha = "A" in img.getbands()
    img = img.convert("RGBA")
    if crop is not None:
        w, h = img.size
        x0, y0, x1, y1 = crop
        img = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))

    # Fit onto a square canvas, longest side -> work, preserving aspect.
    w, h = img.size
    s = work / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (work, work), (0, 0, 0, 0))
    canvas.paste(img, ((work - nw) // 2, (work - nh) // 2))

    rgba = np.asarray(canvas, dtype=np.float32) / 255.0
    alpha = rgba[..., 3] if has_alpha else None
    # Composite over white for the luma path so transparent regions read bright.
    rgb = rgba[..., :3]
    a = rgba[..., 3:4]
    comp = rgb * a + (1.0 - a)
    gray = comp @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return gray, alpha


def _otsu(gray: np.ndarray) -> float:
    """Otsu's threshold on a 0..1 image (returns a 0..1 cut)."""
    hist, edges = np.histogram(gray, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    omega = np.cumsum(p)
    centers = (edges[:-1] + edges[1:]) * 0.5
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return float(centers[int(np.nanargmax(sigma_b2))])


def _foreground(gray: np.ndarray, alpha: np.ndarray | None, pr: TraceParams) -> np.ndarray:
    """Binary foreground mask from the alpha channel or a thresholded luma."""
    if pr.use_alpha and alpha is not None and float(alpha.max()) > 0.0:
        return alpha > 0.5

    g = gray
    if pr.contrast != 1.0:
        g = np.clip((g - 0.5) * pr.contrast + 0.5, 0.0, 1.0)
    cut = pr.threshold if pr.threshold is not None else _otsu(g)
    # Subject is dark-on-light by default -> foreground is BELOW the cut.
    fg = g < cut
    if pr.invert:
        fg = ~fg
    return fg


def _remove_bg_flood(fg: np.ndarray) -> np.ndarray:
    """Drop any *background* (False) component touching the border, then treat
    everything not thus reached as removable if it is background. Practically:
    keep the foreground; but also kill foreground blobs that are actually the
    frame/paper by flooding False from the corners and clearing FG islands that
    are fully enclosed by border-connected background only if they are specks
    (handled elsewhere). Here we simply guarantee the border ring is background
    by clearing any FG that is connected to the image border."""
    # A clean silhouette has NO foreground touching the border. If the source
    # has a filled background that thresholded as foreground, it will touch the
    # border; flip it off by removing the border-connected foreground component.
    lbl, n = ndi.label(fg)
    if n == 0:
        return fg
    border_ids = set(np.unique(np.concatenate([
        lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1],
    ])))
    border_ids.discard(0)
    if not border_ids:
        return fg
    # Only strip a border component if it is huge (a real background fill), not
    # a subject that merely grazes the edge.
    out = fg.copy()
    area = fg.shape[0] * fg.shape[1]
    for cid in border_ids:
        comp = lbl == cid
        if comp.sum() > 0.45 * area:
            out &= ~comp
    return out


def _keep_largest_and_declutter(fg: np.ndarray, min_frac: float) -> np.ndarray:
    """Drop connected components smaller than ``min_frac`` of the image area."""
    lbl, n = ndi.label(fg)
    if n <= 1:
        return fg
    area = fg.shape[0] * fg.shape[1]
    sizes = ndi.sum(np.ones_like(fg), lbl, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    for i, sz in enumerate(sizes, start=1):
        if sz >= min_frac * area:
            keep[i] = True
    return keep[lbl]


def _min_run(mask: np.ndarray) -> int:
    """Thinnest nonzero run of True cells across rows and columns (min feature).

    Same proxy as ``monogram._min_run`` but vectorized per-line for speed.
    """
    best = 10**9
    for m in (mask, mask.T):
        for row in m:
            idx = np.flatnonzero(row)
            if idx.size == 0:
                continue
            # Runs = maximal stretches of consecutive True.
            splits = np.where(np.diff(idx) > 1)[0]
            starts = np.concatenate(([0], splits + 1))
            ends = np.concatenate((splits, [idx.size - 1]))
            runs = idx[ends] - idx[starts] + 1
            r = int(runs.min())
            if r < best:
                best = r
    return 0 if best == 10**9 else best


def _litho_thicken(mask: np.ndarray, pr: TraceParams) -> np.ndarray:
    """Dilate just enough that the thinnest run clears ``min_feature_px``.

    Capped at ``max_thicken_px`` so we never turn a thin leg into a blob.
    """
    thin = _min_run(mask)
    if thin == 0 or thin >= pr.min_feature_px:
        return mask
    r = min(pr.max_thicken_px, max(1, (pr.min_feature_px - thin + 1) // 2))
    # Square structuring element == monogram's box dilation, but O(n) via scipy.
    struct = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
    return ndi.binary_dilation(mask, structure=struct)


def _place(mask: np.ndarray, n: int, pr: TraceParams) -> np.ndarray:
    """Crop to the subject bbox, scale to ``fill``, paste at the anchor on an
    ``n`` x ``n`` grid (optionally flipped horizontally)."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((n, n), dtype=bool)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = mask[y0:y1, x0:x1]
    if pr.flip_h:
        crop = crop[:, ::-1]
    ch, cw = crop.shape
    target = pr.fill * n
    s = target / max(ch, cw)
    nw, nh = max(1, int(round(cw * s))), max(1, int(round(ch * s)))
    scaled = np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255, "L").resize((nw, nh), Image.LANCZOS)
    ) > 127
    out = np.zeros((n, n), dtype=bool)
    tx = int(round(pr.anchor_x * n - nw / 2))
    ty = int(round(pr.anchor_y * n - nh / 2))
    sx0, sy0 = max(0, -tx), max(0, -ty)
    dx0, dy0 = max(0, tx), max(0, ty)
    cw2 = min(nw - sx0, n - dx0)
    ch2 = min(nh - sy0, n - dy0)
    if cw2 <= 0 or ch2 <= 0:
        return out
    out[dy0:dy0 + ch2, dx0:dx0 + cw2] = scaled[sy0:sy0 + ch2, sx0:sx0 + cw2]
    return out


def _punch_holes(mask: np.ndarray, n: int, pr: TraceParams) -> np.ndarray:
    """Punch engraved negative-space ellipses (eye / nostril) to background."""
    if not pr.holes:
        return mask
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    out = mask.copy()
    for cx, cy, rx, ry in pr.holes:
        cxp, cyp = cx * n, cy * n
        rxp, ryp = max(1.0, rx * n), max(1.0, ry * n)
        ell = ((xx - cxp) / rxp) ** 2 + ((yy - cyp) / ryp) ** 2 <= 1.0
        out &= ~ell
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def trace_silhouette(
    image_path: str | Path,
    n_grid: int = 256,
    params: TraceParams | None = None,
    work: int | None = None,
) -> np.ndarray:
    """Trace ``image_path`` into a litho-safe binary silhouette.

    Parameters
    ----------
    image_path : path to a raster reference (PNG/JPG). Alpha PNGs use their
        alpha channel as the foreground; otherwise luma is thresholded.
    n_grid : output grid size (returns ``(n_grid, n_grid)`` bool, True == gold).
    params : :class:`TraceParams` knobs (crop, threshold, placement, holes).
    work : working-grid resolution for the morphology (defaults to ``max(512,
        2*n_grid)`` capped at 1024) — high enough to keep edges crisp, capped to
        stay light on the 13.7 GB host.

    The result honors the same contract as the hand-authored motifs and is
    scale-free (the plate compositor picks the cell pitch that hits the extent).
    """
    pr = params or TraceParams()
    n = max(64, int(n_grid))
    # Own raster path (not render_silhouette), so gate n here too — callers size
    # n_grid straight off unvalidated extent_um/period params.
    check_silhouette_budget(n, "Traced silhouette")
    w = work if work is not None else min(1024, max(512, n * 2))
    # The working grid is uncapped when a caller overrides it; it carries the
    # float morphology stack, so it needs the same ceiling as the output grid.
    check_silhouette_budget(w, "Traced silhouette working grid")

    gray, alpha = _load_gray_alpha(Path(image_path), pr.crop, w)
    fg = _foreground(gray, alpha, pr)
    if pr.flood_bg:
        fg = _remove_bg_flood(fg)
    if pr.fill_holes:
        fg = ndi.binary_fill_holes(fg)
    if pr.min_speck_frac > 0:
        fg = _keep_largest_and_declutter(fg, pr.min_speck_frac)

    # Place into the centered art box at the requested output resolution FIRST,
    # so the litho min-feature test is measured in *output* cells.
    placed = _place(fg, n, pr)
    placed = _litho_thicken(placed, pr)
    placed = _punch_holes(placed, n, pr)
    return placed
