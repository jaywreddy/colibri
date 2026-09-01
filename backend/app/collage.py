"""Tilt-sweep COLLAGE: one pattern rendered across a range of view angles.

The purpose is visual inspection — can a person look at a pattern and confirm
it actually does what it claims? That question is surprisingly hard to answer
from the 3D preview, because the two-plane WebGL renderer filters each layer
independently before compositing (it computes <front>*<back>) while the real
object integrates the product, <front*back>. The correlation between the layers
IS the effect, so once the lattice drops below a screen pixel the preview loses
exactly the term the part keeps: measured on the shipping box, the frame moiré
survives at ~1% contrast and the A/B interlace collapses to a static blend.

This module has no such problem. ``sim2d.composite_parallax`` multiplies the
FRONT mask by the parallax-SHIFTED back mask at full raster resolution, which
is the correct integration, and only then is the result area-averaged down to a
tile. So a collage tile shows what the eye would actually integrate — and a
pattern that is broken here is broken in glass.

Each column is one tilt angle; the shift comes from the same Snell relation the
simulator and renderer use (``sim2d.parallax_shift_um``), so the angles printed
under the tiles are the real ones a wrist would produce.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import sim2d

# A sweep that covers the barrier family's whole behaviour: the A/B swap peaks
# near +/-2.5 deg and the pattern aliases around 10 deg, so +/-6 deg in 13 steps
# shows the swap, its reversal, and the approach to alias without wasting tiles.
DEFAULT_ANGLES_DEG: tuple[float, ...] = (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)

# The back layer is shifted by a WHOLE number of raster pixels (sim2d._shift_px
# rounds), but a stored variant raster only runs 1.0-1.5 px per degree of tilt.
# Rounding there costs up to half a pixel — a third of a degree on a switch that
# peaks at 2.5 — and it makes the steps between columns uneven, which is exactly
# the axis this sheet exists to show. So the pair is resampled onto a finer grid
# first, by an integer factor, until one angular step is worth at least this
# many pixels. Nearest-neighbour by an integer factor adds no detail that was
# not already in the raster: it re-samples the same rectangles on a finer grid,
# and the tile is area-averaged back down afterwards.
MIN_PX_PER_STEP = 4.0
MAX_FINE_SIDE = 2048

LABEL_H = 14
PAD = 4
BG = (14, 16, 22)
FG = (232, 234, 237)
ACCENT = (227, 181, 59)


def _as_array(m: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(m, Image.Image):
        return np.asarray(m.convert("L"), dtype=np.float32) / 255.0
    return np.asarray(m, dtype=np.float32)


def _step_shift_um(
    thickness_um: float, n: float, angles_deg, axis: str
) -> float:
    """The smallest non-zero parallax step the sweep asks for, in micrometres."""
    idx = 0 if axis == "x" else 1
    shifts = [
        sim2d.parallax_shift_um(
            a if axis == "x" else 0.0, 0.0 if axis == "x" else a, thickness_um, n
        )[idx]
        for a in angles_deg
    ]
    gaps = [abs(b - a) for a, b in zip(shifts, shifts[1:]) if abs(b - a) > 1e-9]
    return min(gaps) if gaps else 0.0


def _refine_factor(
    shape: tuple[int, ...], pixel_pitch_um: float,
    thickness_um: float, n: float, angles_deg, axis: str,
) -> int:
    """Integer upsample needed to give one angular step MIN_PX_PER_STEP pixels."""
    step = _step_shift_um(thickness_um, n, angles_deg, axis)
    if step <= 0 or pixel_pitch_um <= 0:
        return 1
    k = max(1, math.ceil(pixel_pitch_um / (step / MIN_PX_PER_STEP)))
    side = max(shape[:2])
    while k > 1 and side * k > MAX_FINE_SIDE:
        k -= 1
    return k


def sweep_frames(
    front: np.ndarray | Image.Image,
    back: np.ndarray | Image.Image,
    *,
    pixel_pitch_um: float,
    thickness_um: float,
    n: float,
    angles_deg: tuple[float, ...] | list[float] = DEFAULT_ANGLES_DEG,
    illum: str = "ambient",
    axis: str = "x",
    tile_px: int = 200,
) -> list[tuple[float, Image.Image]]:
    """Composite the pair at each tilt; return [(angle, tile image)].

    ``axis`` picks which way the wrist rotates — 'x' walks the back layer along
    the plate's x axis (the switch axis for every barrier face in the catalogue),
    'y' the other way.
    """
    if illum not in sim2d.ILLUMINATIONS:
        raise ValueError(f"unknown illum {illum!r}")
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y' (got {axis!r})")

    front = _as_array(front)
    back = _as_array(back)
    k = _refine_factor(front.shape, pixel_pitch_um, thickness_um, n, angles_deg, axis)
    if k > 1:
        front = np.repeat(np.repeat(front, k, axis=0), k, axis=1)
        back = np.repeat(np.repeat(back, k, axis=0), k, axis=1)
        pixel_pitch_um = pixel_pitch_um / k

    out: list[tuple[float, Image.Image]] = []
    for a in angles_deg:
        tx, ty = (a, 0.0) if axis == "x" else (0.0, a)
        dx, dy = sim2d.parallax_shift_um(tx, ty, thickness_um, n)
        img = sim2d.composite_parallax(front, back, dx, dy, pixel_pitch_um, illum=illum)
        # Downsample LAST: the product is formed at full raster resolution, then
        # averaged — the same order the eye works in, and the reason this tile is
        # a fair prediction where the 3D preview is not.
        # BOX = plain area average, which is exactly the integration a retinal
        # receptive field (or a camera pixel) performs. Lanczos would ring, and
        # ringing is an artifact of the resampler rather than of the optics.
        img = img.resize((tile_px, tile_px), Image.Resampling.BOX)
        out.append((float(a), img))
    return out


def tile_metrics(
    frames: list[tuple[float, Image.Image]], crop_frac: float = 0.1
) -> dict[str, Any]:
    """Measure the sweep: how much does the image actually change with tilt?

    ``effect_strength`` is mean |dL| over mean luminance for the MOST DIFFERENT
    PAIR of tiles in the sweep — the honest headline for an image SWAP, where A
    and B can have near-identical mean brightness while half the pixels invert.

    Comparing the sweep's two ENDPOINTS instead is wrong, and silently so: every
    one of these effects is periodic in the shift, so the extremes can land an
    integer number of periods apart and come back byte-identical. The test
    barrier does exactly that — +/-6 deg is -/+1.5 periods, the two ends match
    to 0.0, and endpoint differencing scores a working switch as dead while the
    real peak (-6 vs 0) swings 126/255. ``peak_pair_deg`` reports where it is.

    ``mean_swing`` is the Michelson swing of the per-tile means, which is what
    a shimmer/reveal moves instead.
    """
    if not frames:
        return {
            "effect_strength": 0.0, "mean_swing": 0.0,
            "changed_frac": 0.0, "peak_pair_deg": [], "tile_means": [],
        }
    # Measure a CENTRE CROP. Shifting a finite raster pulls zeros in at the
    # trailing border, so that band changes with tilt on ANY input — a
    # featureless pair reads as a ~25% effect if you include it. The band is as
    # wide as the shift itself, so the inset must SCALE with the sweep rather
    # than being a fixed fraction; build_collage passes the real shift.
    arrs = [np.asarray(im.convert("L"), dtype=float) for _, im in frames]
    h, w = arrs[0].shape
    frac = min(0.45, max(0.0, crop_frac))
    my, mx = int(h * frac), int(w * frac)
    if h - 2 * my >= 4 and w - 2 * mx >= 4:
        arrs = [a[my : h - my, mx : w - mx] for a in arrs]
    means = [float(a.mean()) for a in arrs]
    hi, lo = max(means), min(means)
    angles = [a for a, _ in frames]

    best = (0.0, 0, 0)
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            d = float(np.abs(arrs[i] - arrs[j]).mean())
            if d > best[0]:
                best = (d, i, j)
    mad, i, j = best
    ai, aj = arrs[i], arrs[j]
    lum = float((0.5 * (ai + aj)).mean())
    changed = float((np.abs(ai - aj) > 0.02 * 255).mean())
    return {
        "effect_strength": (mad / lum) if lum > 0 else 0.0,
        "mean_swing": ((hi - lo) / (hi + lo)) if (hi + lo) > 0 else 0.0,
        "changed_frac": changed,
        "peak_pair_deg": [angles[i], angles[j]],
        "tile_means": [round(m, 2) for m in means],
    }


def compose_grid(
    frames: list[tuple[float, Image.Image]],
    *,
    cols: int | None = None,
    title: str | None = None,
) -> Image.Image:
    """Lay the tiles out as a labelled grid, angles printed under each tile."""
    if not frames:
        raise ValueError("no frames to compose")
    tile_px = frames[0][1].width
    n_tiles = len(frames)
    cols = cols or min(n_tiles, 7)
    rows = math.ceil(n_tiles / cols)
    title_h = LABEL_H + PAD if title else 0

    w = cols * tile_px + (cols + 1) * PAD
    h = rows * (tile_px + LABEL_H) + (rows + 1) * PAD + title_h
    sheet = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((PAD, PAD // 2), title, fill=ACCENT)

    for i, (angle, img) in enumerate(frames):
        r, c = divmod(i, cols)
        x = PAD + c * (tile_px + PAD)
        y = title_h + PAD + r * (tile_px + LABEL_H + PAD)
        sheet.paste(img, (x, y))
        label = f"{angle:+.0f}deg" if angle else "0deg"
        draw.text((x + 2, y + tile_px + 1), label, fill=FG)
    return sheet


def border_crop_frac(
    shape: tuple[int, ...], pixel_pitch_um: float,
    thickness_um: float, n: float, angles_deg, axis: str,
) -> float:
    """How far in the metrics must inset to clear the shifted-in border.

    The border is as wide as the largest shift in the sweep, so this is that
    shift as a fraction of the raster, plus a little slack.
    """
    idx = 0 if axis == "x" else 1
    raster_um = float(shape[1] if axis == "x" else shape[0]) * pixel_pitch_um
    if raster_um <= 0 or not len(angles_deg):
        return 0.1
    max_shift = max(
        abs(sim2d.parallax_shift_um(
            a if axis == "x" else 0.0, 0.0 if axis == "x" else a, thickness_um, n
        )[idx])
        for a in angles_deg
    )
    return (max_shift / raster_um) + 0.02


def build_collage(
    front: np.ndarray | Image.Image,
    back: np.ndarray | Image.Image,
    *,
    pixel_pitch_um: float,
    thickness_um: float,
    n: float,
    angles_deg: tuple[float, ...] | list[float] = DEFAULT_ANGLES_DEG,
    illum: str = "ambient",
    axis: str = "x",
    tile_px: int = 200,
    cols: int | None = None,
    title: str | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Full sweep -> (collage sheet, metrics)."""
    frames = sweep_frames(
        front, back,
        pixel_pitch_um=pixel_pitch_um, thickness_um=thickness_um, n=n,
        angles_deg=angles_deg, illum=illum, axis=axis, tile_px=tile_px,
    )
    crop = border_crop_frac(
        np.asarray(front).shape, pixel_pitch_um, thickness_um, n, angles_deg, axis
    )
    return (
        compose_grid(frames, cols=cols, title=title),
        tile_metrics(frames, crop_frac=crop),
    )
