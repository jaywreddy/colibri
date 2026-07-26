"""2D parallax simulator — headless preview + metrics for dual-layer masks.

The 3D renderer (frontend/src/shaders/plate.frag, moire_interactive recipe) is
the source of truth for how a viewer perceives the stacked gold layers. This
module reimplements its core in numpy — Snell-refracted back-layer shift plus
the three illumination composites — so pattern development can render tilts
and QUANTIFY contrast-vs-tilt curves without a GPU, browser, or FFT machinery.

Shared shift contract (must match frontend/src/shaders/lib/parallax.glsl):
    sinV = sin(tilt); sinSub = sinV / n; cosSub = sqrt(max(0, 1 - sinSub^2))
    shift_um = thickness_um * sinSub / max(0.05, cosSub)
applied along the tilt axis (x tilt -> x shift, y tilt -> y shift).

Shared compositing contract (simplified from plate.frag; gold masks are
grayscale 0..1 where 1 = gold):
    back_shifted = back sampled at (uv - shift)   [zero outside the frame]
    transmission = (1 - front) * (1 - back_shifted)
    reflected    = max(front, 0.55 * back_shifted)
    overlap      = front * back_shifted           [ambient darkening only]

This module is a PAIRED contract with frontend/src/lab/composite2d.ts (the
Pattern Lab's 2D compositor): any change to the shift or composite formulas
must land in both files in the same change, and both pin the same
hand-computed ambient values against plate.frag.

Pure functions only: no file I/O — callers hand in PIL "L" images or 2D arrays
(see _as_unit_mask; a caller that area-averages a big raster down first passes
the reduced float grid, with pixel_pitch_um scaled by the same factor).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

# Shader constants (plate.frag) — keep byte-identical with the GLSL.
GOLD = (0.902, 0.737, 0.314)
GOLD_BACK = (0.4, 0.32, 0.12)
DEFAULT_LASER = (0.267, 1.0, 0.533)

ILLUMINATIONS = ("ambient", "laser", "backlight")


def _axis_shift_um(tilt_deg: float, thickness_um: float, n: float) -> float:
    """Snell-refracted lateral shift for a single tilt axis (see parallax.glsl)."""
    sin_v = math.sin(math.radians(tilt_deg))
    sin_sub = sin_v / n
    cos_sub = math.sqrt(max(0.0, 1.0 - sin_sub * sin_sub))
    return thickness_um * sin_sub / max(0.05, cos_sub)


def parallax_shift_um(
    tilt_x_deg: float,
    tilt_y_deg: float,
    thickness_um: float,
    n: float,
) -> tuple[float, float]:
    """Back-layer shift (dx_um, dy_um) for a view tilted about each axis.

    Each axis is treated independently with its own tilt angle — for the
    small-to-moderate tilts the studio uses this matches the shader's single
    lateral vector to well under a pixel.
    """
    return (
        _axis_shift_um(tilt_x_deg, thickness_um, n),
        _axis_shift_um(tilt_y_deg, thickness_um, n),
    )


def _to_unit(img: Image.Image) -> np.ndarray:
    """PIL 'L' mask -> float32 array in 0..1 (1 = gold)."""
    return np.asarray(img, dtype=np.float32) / 255.0


def _shift_back(back: np.ndarray, dx_px: int, dy_px: int) -> np.ndarray:
    """Sample `back` at (x - dx_px, y - dy_px) with zero fill.

    This is the shader's `texture2D(uBack, vUv - shift)` in pixel space:
    content moves +dx_px/+dy_px in the output, and anything sampled outside
    the frame reads as 0 (no gold).
    """
    h, w = back.shape
    out = np.zeros_like(back)
    ys_dst = slice(max(0, dy_px), min(h, h + dy_px))
    xs_dst = slice(max(0, dx_px), min(w, w + dx_px))
    ys_src = slice(max(0, -dy_px), min(h, h - dy_px))
    xs_src = slice(max(0, -dx_px), min(w, w - dx_px))
    if ys_dst.start < ys_dst.stop and xs_dst.start < xs_dst.stop:
        out[ys_dst, xs_dst] = back[ys_src, xs_src]
    return out


def _shift_px(dx_um: float, dy_um: float, pixel_pitch_um: float) -> tuple[int, int]:
    return (
        int(round(dx_um / pixel_pitch_um)),
        int(round(dy_um / pixel_pitch_um)),
    )


def _transmission(front: np.ndarray, back_shifted: np.ndarray) -> np.ndarray:
    return (1.0 - front) * (1.0 - back_shifted)


def composite_parallax(
    front: Image.Image | np.ndarray,
    back: Image.Image | np.ndarray,
    dx_um: float,
    dy_um: float,
    pixel_pitch_um: float,
    illum: str = "ambient",
    laser_color: tuple[float, float, float] = DEFAULT_LASER,
) -> Image.Image:
    """Composite front + parallax-shifted back into an RGB preview image.

    Implements the three plate.frag illumination modes on flat masks (no
    Lambert/specular terms — this is the head-on-light simplification, with
    0.85 / 0.12 / 0.25 standing in for the shader's head-on light factors):
      ambient  : (GOLD * reflected * 0.85 + 0.04 * transmission)
                 * (1 - 0.35 * overlap)
      laser    : laser_color * transmission + GOLD * 0.12 * reflected
      backlight: white * transmission + GOLD_BACK * reflected * 0.25

    The ambient overlap darkening is the shader's ONLY back-layer dependence
    wherever the front mask is gold (reflected saturates at 1 and transmission
    is 0 there), so dropping it makes every ambient metric over a front-gold
    figure tilt-blind. Laser and backlight have no overlap factor in the
    shader; do not add one here.

    Masks may be PIL 'L' images or 2D arrays (see _as_unit_mask) — callers that
    area-average a big raster down before compositing hand in floats.
    """
    f = _as_unit_mask(front)
    b = _as_unit_mask(back)
    if f.shape != b.shape:
        raise ValueError(f"front/back size mismatch: {f.shape} vs {b.shape}")

    dx_px, dy_px = _shift_px(dx_um, dy_um, pixel_pitch_um)
    b_shifted = _shift_back(b, dx_px, dy_px)
    transmission = _transmission(f, b_shifted)
    reflected = np.maximum(f, 0.55 * b_shifted)

    def _tint(color: tuple[float, float, float], field: np.ndarray) -> np.ndarray:
        return np.asarray(color, dtype=np.float32)[None, None, :] * field[..., None]

    if illum == "ambient":
        # plate.frag darkens the whole ambient color (gold shade AND the
        # transmission floor) where both layers are gold.
        overlap_dark = 1.0 - 0.35 * f * b_shifted
        rgb = (
            _tint(GOLD, reflected * 0.85) + 0.04 * transmission[..., None]
        ) * overlap_dark[..., None]
    elif illum == "laser":
        rgb = _tint(laser_color, transmission) + _tint(GOLD, 0.12 * reflected)
    elif illum == "backlight":
        rgb = transmission[..., None].repeat(3, axis=2) + _tint(
            GOLD_BACK, 0.25 * reflected
        )
    else:
        raise ValueError(f"Unknown illum {illum!r}; expected one of {ILLUMINATIONS}")

    out = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def transmission_contrast(
    front: Image.Image | np.ndarray,
    back: Image.Image | np.ndarray,
    dx_um: float,
    dy_um: float,
    pixel_pitch_um: float,
) -> float:
    """Mean transmission over the frame at a given parallax shift.

    This is the brightness a backlit viewer sees; sweeping it against tilt is
    the pattern-development metric (see contrast_curve).
    """
    f = _as_unit_mask(front)
    b = _as_unit_mask(back)
    if f.shape != b.shape:
        raise ValueError(f"front/back size mismatch: {f.shape} vs {b.shape}")
    dx_px, dy_px = _shift_px(dx_um, dy_um, pixel_pitch_um)
    return float(_transmission(f, _shift_back(b, dx_px, dy_px)).mean())


def contrast_curve(
    front: Image.Image | np.ndarray,
    back: Image.Image | np.ndarray,
    tilts_deg: list[float],
    thickness_um: float,
    n: float,
    pixel_pitch_um: float,
    axis: str = "x",
) -> list[dict]:
    """Sweep tilt along one axis and report mean transmission at each step.

    A working moiré pair shows strong modulation (the fringes sweep through
    the frame as the back layer slides); a blank or degenerate pair shows a
    flat curve. Each row is {tilt_deg, dx_um, transmission} where dx_um is
    the shift along the swept axis.

    The sweep is one full-frame composite per tilt, so it scales with the mask
    area: bound the grid (area-average) before calling on a plate-sized raster.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"Unknown axis {axis!r}; expected 'x' or 'y'")
    f = _as_unit_mask(front)
    b = _as_unit_mask(back)
    if f.shape != b.shape:
        raise ValueError(f"front/back size mismatch: {f.shape} vs {b.shape}")

    rows: list[dict] = []
    for tilt in tilts_deg:
        d_um = _axis_shift_um(float(tilt), thickness_um, n)
        dx_um, dy_um = (d_um, 0.0) if axis == "x" else (0.0, d_um)
        dx_px, dy_px = _shift_px(dx_um, dy_um, pixel_pitch_um)
        t = float(_transmission(f, _shift_back(b, dx_px, dy_px)).mean())
        rows.append({"tilt_deg": float(tilt), "dx_um": d_um, "transmission": t})
    return rows


# ---------------------------------------------------------------------------
# Pattern-type evaluation layer
#
# Zone math (all periodic dual-layer constructions): the composite depends on
# the back-layer shift s only through s mod p (carrier period p), so the view
# REPEATS — "zones" — every p of shift. The landmark views are half-period
# multiples k * p / 2 (shift_for_zone): zone 0 = head-on registration, odd
# zones = the de-registered / switched view, even zones re-register and repeat
# zone 0 (up to frame-edge loss). Exterior tilt for a back-layer shift s is
#     theta(s) = asin(n * sin(atan(s / t)))          (tilt_for_shift_um)
# at t=500 um, n=1.46: s = 10 / 20 / 40 / 80 um -> 1.67 / 3.35 / 6.69 / 13.34
# deg. A 14 deg demo tilt is an 84 um shift = 2.1 periods of a p=40 um
# construction — deep into zone aliasing.
#
# All metric functions are pure numpy on grayscale masks (PIL 'L' images or
# 2D arrays, 1 = gold); no file I/O, no GEOS, tiny-grid friendly.
# ---------------------------------------------------------------------------


def shift_for_zone(period_um: float, k: int) -> float:
    """Back-layer shift landing at the center of parallax zone ``k``.

    Zone k sits at k * period / 2 of shift: zone 0 = head-on registration,
    zone 1 = p/2 = the switched / de-registered view, zone 2 = p = a full
    period, which re-registers and repeats zone 0 (aliasing). Example: for
    p=40 um at t=500 um / n=1.46, zone 1 is 20 um = 3.35 deg of tilt and the
    pattern realiases every 6.69 deg.
    """
    return float(k) * float(period_um) / 2.0


def tilt_for_shift_um(shift_um: float, thickness_um: float, n: float) -> float:
    """Exterior tilt (deg) producing a given back-layer shift — the exact
    inverse of the Snell shift in parallax_shift_um (per axis).

        theta_sub = atan(shift / thickness);  theta = asin(n * sin(theta_sub))

    Anchors at t=500 um, n=1.46: 20 um -> 3.345 deg, 40 -> 6.69, 84 -> 14.00.
    Odd-symmetric in shift. Raises ValueError when no exterior angle can
    produce the shift (n * sin(atan(s/t)) >= 1, beyond grazing exit; at
    t=500 um / n=1.46 that is |s| >= ~470 um).
    """
    if thickness_um <= 0:
        raise ValueError("thickness_um must be positive")
    sin_v = n * math.sin(math.atan(shift_um / thickness_um))
    if abs(sin_v) >= 1.0:
        raise ValueError(
            f"shift {shift_um} um at t={thickness_um} um, n={n} exceeds the "
            "grazing-exit limit; no exterior tilt produces it"
        )
    return math.degrees(math.asin(sin_v))


def _as_unit_mask(mask: Image.Image | np.ndarray) -> np.ndarray:
    """Accept a PIL 'L' image or a 2D array as a unit gold mask.

    Integer arrays follow the PIL convention (0..255 -> /255); float and
    bool arrays are taken as already 0..1 (1 = gold).
    """
    if isinstance(mask, Image.Image):
        return _to_unit(mask)
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _mask_pair(
    front: Image.Image | np.ndarray,
    back: Image.Image | np.ndarray,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a front/back pair; transpose so the swept axis is x."""
    if axis not in ("x", "y"):
        raise ValueError(f"Unknown axis {axis!r}; expected 'x' or 'y'")
    f = _as_unit_mask(front)
    b = _as_unit_mask(back)
    if f.shape != b.shape:
        raise ValueError(f"front/back size mismatch: {f.shape} vs {b.shape}")
    if axis == "y":
        f = f.T
        b = b.T
    return f, b


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two fields; constant fields count as 1.0
    (an unchanging view is treated as perfectly self-similar, so identical
    gratings report zero decorrelation rather than 0/0 noise)."""
    av = a.astype(np.float64).ravel()
    bv = b.astype(np.float64).ravel()
    av -= av.mean()
    bv -= bv.mean()
    norm_a = math.sqrt(float((av * av).sum()))
    norm_b = math.sqrt(float((bv * bv).sum()))
    # Per-field std guard (not the norm product): fields flat to within
    # float rounding are "constant", never correlated against noise.
    n = math.sqrt(av.size)
    if norm_a / n < 1e-6 or norm_b / n < 1e-6:
        return 1.0
    return float((av * bv).sum() / (norm_a * norm_b))


def _box_blur_axis(field: np.ndarray, k: int, axis: int) -> np.ndarray:
    """One axis of the separable box filter, as a prefix-sum difference.

    Reproduces ``np.convolve(v, np.ones(k) / k, mode='same')`` term for term:
    that trims the length-(n + k - 1) full convolution to its middle n samples,
    so output i is the sum over the input window ``[i + off - k + 1, i + off]``
    with ``off = (k - 1) // 2`` and every out-of-frame tap counted as ZERO (not
    edge-replicated). Clamping the window ENDS (not the sampled values) against
    a prefix sum sums exactly those taps, one vectorized pass instead of
    ``np.apply_along_axis``'s per-row Python ``np.convolve`` call.

    Accumulated in float64 because a prefix sum runs the length of the axis
    while the window sum it replaces was only k terms long.
    """
    n = field.shape[axis]
    a = np.moveaxis(field, axis, 0)
    cs = np.empty((n + 1, *a.shape[1:]), dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(a, axis=0, dtype=np.float64, out=cs[1:])
    off = (k - 1) // 2
    ends = np.arange(n) + off + 1
    out = cs[np.clip(ends, 0, n)]
    out -= cs[np.clip(ends - k, 0, n)]
    out /= float(k)
    return np.moveaxis(out, 0, axis)


def _box_blur(field: np.ndarray, k: int) -> np.ndarray:
    """Separable k-pixel box filter ('same' edges — crop >= k/2 margins
    before using the result quantitatively)."""
    if k <= 1:
        return field.astype(np.float32)
    return _box_blur_axis(_box_blur_axis(field, k, 1), k, 0).astype(np.float32)


def sweep_transmission(
    front: Image.Image | np.ndarray,
    back: Image.Image | np.ndarray,
    pixel_pitch_um: float,
    shifts_um: list[float],
    axis: str = "x",
) -> list[dict]:
    """Mean backlit transmission vs back-layer shift along one axis.

    contrast_curve reparameterized by shift instead of tilt (translate with
    tilt_for_shift_um for demo calibration). Rows are
    {"shift_um", "transmission"} using the shader transmission contract
    T = (1 - front) * (1 - back_shifted); shifts round to whole pixels.
    """
    f, b = _mask_pair(front, back, axis)
    rows: list[dict] = []
    for s in shifts_um:
        px = int(round(float(s) / pixel_pitch_um))
        t = float(_transmission(f, _shift_back(b, px, 0)).mean())
        rows.append({"shift_um": float(s), "transmission": t})
    return rows


def switch_metrics(
    front: Image.Image | np.ndarray,
    back: Image.Image | np.ndarray,
    pixel_pitch_um: float,
    period_um: float,
    *,
    shift_um: float | None = None,
    axis: str = "x",
) -> dict:
    """Two-image switch quality for barrier-type (T1) constructions.

    The back layer is resampled into its two half-period column-phase
    channels — channel A = columns with (x mod p) < p/2, channel B = the
    rest — and each channel is composited ALONE through the front. Channel
    visibility is the normalized aperture correlation
        vis = sum((1 - front) * shifted_channel) / sum(channel)
    (the audit metric: 0.819 / 0.000 measured on the working
    colibri-globe-lenticular barrier — total extinction of the hidden
    channel).

    Composites are evaluated at +shift and -shift; the default shift is
    period/2 (zone 1, the first de-registered view; exterior tilt
    tilt_for_shift_um(p/2, t, n) = 3.35 deg at p=40, t=500, n=1.46).
    Registration note: a slit centered over channel A shows the hidden
    channel at BOTH signs of a p/2 shift (sign-symmetric); the straddle
    registration (slit offset p/4) switches A <-> B symmetrically at
    +/- p/4 — pass shift_um=period_um/4 for that geometry.

    Returns a dict:
      separation   min over the two tilt signs of
                   (dominant vis + 1e-3) / (suppressed vis + 1e-3). Forced
                   to 1.0 when the back is not actually interlaced (either
                   channel holds < 1% of the back gold): a construction with
                   all image content in one column phase — e.g. the borked
                   front-image / back-image anti-phase pairs — structurally
                   cannot switch, whatever its raw ratios say. Clean barrier
                   at first zone: ~1000; borked phase pair: 1.0.
      overlap_frac lit-pixel IoU (T > 0.5) between the two signs; ~1.0 means
                   both tilt directions show the same view (measured 0.99 on
                   the borked jp-monogram-phase pair). Empty union -> 1.0.
      corr         Pearson correlation of the two signs' darkness fields over
                   front-open pixels (front < 0.5; constant fields count as
                   1.0, < 16 open pixels -> 1.0). A true left/right switch
                   with distinct channel images scores near 0; identical
                   views score ~1.
      vis_a_plus / vis_b_plus / vis_a_minus / vis_b_minus, shift_um.
    """
    f, b = _mask_pair(front, back, axis)
    if pixel_pitch_um <= 0 or period_um <= 0:
        raise ValueError("pixel_pitch_um and period_um must be positive")
    sigma = period_um / 2.0 if shift_um is None else float(shift_um)
    s_px = int(round(sigma / pixel_pitch_um))

    w = f.shape[1]
    phase = np.mod(np.arange(w, dtype=np.float64) * pixel_pitch_um, period_um)
    in_a = (phase < period_um / 2.0)[None, :]
    chan_a = b * in_a
    chan_b = b * ~in_a
    mass_a = float(chan_a.sum())
    mass_b = float(chan_b.sum())
    total = mass_a + mass_b
    interlaced = total > 0.0 and min(mass_a, mass_b) >= 0.01 * total

    open_front = 1.0 - f
    eps = 1e-3
    vis: dict[str, float] = {}
    dark: dict[str, np.ndarray] = {}
    lit: dict[str, np.ndarray] = {}
    for label, sign in (("plus", 1), ("minus", -1)):
        t_full = _transmission(f, _shift_back(b, sign * s_px, 0))
        dark[label] = 1.0 - t_full
        lit[label] = t_full > 0.5
        for cname, chan, mass in (("a", chan_a, mass_a), ("b", chan_b, mass_b)):
            shifted = _shift_back(chan, sign * s_px, 0)
            vis[f"vis_{cname}_{label}"] = float(
                (open_front * shifted).sum() / max(mass, 1e-9)
            )

    if interlaced:
        dominances = []
        for label in ("plus", "minus"):
            va = vis[f"vis_a_{label}"]
            vb = vis[f"vis_b_{label}"]
            dominances.append((max(va, vb) + eps) / (min(va, vb) + eps))
        separation = float(min(dominances))
    else:
        separation = 1.0

    union = lit["plus"] | lit["minus"]
    inter = lit["plus"] & lit["minus"]
    n_union = int(union.sum())
    overlap_frac = 1.0 if n_union == 0 else float(inter.sum()) / n_union

    open_mask = f < 0.5
    if int(open_mask.sum()) < 16:
        corr = 1.0
    else:
        corr = _pearson(dark["plus"][open_mask], dark["minus"][open_mask])

    return {
        "separation": separation,
        "overlap_frac": overlap_frac,
        "corr": corr,
        "shift_um": sigma,
        **vis,
    }


def reveal_metrics(
    front: Image.Image | np.ndarray,
    back: Image.Image | np.ndarray,
    pixel_pitch_um: float,
    period_um: float,
    *,
    axis: str = "x",
) -> dict:
    """Contrast-reveal quality (T5 carrier-phase-reveal / T6 complement).

    Sweeps the back-layer shift 0 .. p/2 in whole-pixel steps and measures
    mean transmission on the frame cropped by the half-period (so the
    zero-filled entry strip never biases the sweep). Reveal math: an ideal
    anti-phase 50%-duty pair rises linearly, T(s) = s/p — extinction at
    registration and the full 0.5 exactly at s = p/2 (exterior tilt
    tilt_for_shift_um(p/2, t, n) = 3.35 deg at p=40, t=500, n=1.46, per the
    zone table above); an in-phase pair is the mirror image (max at 0,
    extinction at p/2). A static pair sweeps flat -> ratio ~1.

    Returns {"modulation_ratio" (t_max / max(t_min, 1e-3)), "t_max",
    "t_min", "shift_at_max_um", "shift_at_min_um",
    "curve": [{"shift_um", "transmission"}, ...]}.
    """
    f, b = _mask_pair(front, back, axis)
    if pixel_pitch_um <= 0 or period_um <= 0:
        raise ValueError("pixel_pitch_um and period_um must be positive")
    half_px = int(round(period_um / 2.0 / pixel_pitch_um))
    if half_px < 1:
        raise ValueError("period_um too small for pixel_pitch_um")
    h, w = f.shape
    m = half_px
    if 2 * m >= h or 2 * m >= w:
        raise ValueError(
            f"frame {f.shape} too small for a {half_px} px half-period crop"
        )

    curve: list[dict] = []
    for px in range(half_px + 1):
        t = _transmission(f, _shift_back(b, px, 0))[m : h - m, m : w - m]
        curve.append(
            {"shift_um": px * pixel_pitch_um, "transmission": float(t.mean())}
        )
    values = [row["transmission"] for row in curve]
    i_max = int(np.argmax(values))
    i_min = int(np.argmin(values))
    t_max = values[i_max]
    t_min = values[i_min]
    return {
        "modulation_ratio": t_max / max(t_min, 1e-3),
        "t_max": t_max,
        "t_min": t_min,
        "shift_at_max_um": curve[i_max]["shift_um"],
        "shift_at_min_um": curve[i_min]["shift_um"],
        "curve": curve,
    }


def fringe_metrics(
    front: Image.Image | np.ndarray,
    back: Image.Image | np.ndarray,
    pixel_pitch_um: float,
    period_um: float,
    *,
    axis: str = "x",
) -> dict:
    """Moire fringe flow (T3): decorrelation of the low-passed transmission
    field per quarter-period step of back-layer shift.

    T(s) is box-blurred over 2 carrier periods (removes the carrier, keeps
    the beat envelope), the frame is cropped by (max shift + blur
    half-width), and Pearson correlation is taken between consecutive fields
    at shifts 0, p/4, p/2, 3p/4, p:

        decorrelation_per_quarter = 1 - mean(consecutive correlations)

    A detuned pair's envelope glides at the moire magnification
    p_f / (p_f - p_b) per unit shift, so quarter-period steps decorrelate
    strongly (wayuu-kanasu measured corr 0.28 at p/4 and -0.36 at p/2 —
    decorrelation ~0.7); identical gratings blur to a constant field
    (constant fields count as corr 1.0) -> decorrelation 0 and near-zero
    fringe_std. fringe_std is the spatial std of the head-on blurred field
    (fringe visibility; a working moire shows > 0.05).

    Returns {"decorrelation_per_quarter", "fringe_std", "corr_quarter",
    "corr_half", "corr_full"}.
    """
    f, b = _mask_pair(front, back, axis)
    if pixel_pitch_um <= 0 or period_um <= 0:
        raise ValueError("pixel_pitch_um and period_um must be positive")
    period_px = period_um / pixel_pitch_um
    q_px = max(1, int(round(period_px / 4.0)))
    shifts = [0, q_px, 2 * q_px, 3 * q_px, 4 * q_px]
    k = max(2, int(round(2.0 * period_px)))
    margin = shifts[-1] + (k + 1) // 2
    h, w = f.shape
    if 2 * margin + 4 >= h or 2 * margin + 4 >= w:
        raise ValueError(
            f"frame {f.shape} too small for period {period_um} um at pitch "
            f"{pixel_pitch_um} um (needs > {2 * margin + 4} px per side)"
        )

    fields = []
    for px in shifts:
        t = _transmission(f, _shift_back(b, px, 0))
        fields.append(_box_blur(t, k)[margin : h - margin, margin : w - margin])
    consecutive = [
        _pearson(fields[i], fields[i + 1]) for i in range(len(fields) - 1)
    ]
    return {
        "decorrelation_per_quarter": float(1.0 - float(np.mean(consecutive))),
        "fringe_std": float(fields[0].std()),
        "corr_quarter": consecutive[0],
        "corr_half": _pearson(fields[0], fields[2]),
        "corr_full": _pearson(fields[0], fields[4]),
    }
