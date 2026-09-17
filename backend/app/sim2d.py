"""Barrier-switch METRICS: does a two-image parallax switch actually switch?

What is left of the 2D parallax lab. The lab itself — the tilt compositor, the
contrast-vs-tilt curves, the three illumination models, the reveal and fringe
metrics — scored the two-ply constructions the box abandoned on 2026-09-15, and
went with them (git history, 22d1634). ONE construction survives as a hidden
exemplar, ``globe-duo-phase``, and one question about it still has to be
answered by measurement rather than by arithmetic: with the real generated
geometry in front of you, does the off-channel image extinguish?

So this is the measurement, and nothing else. ``switch_metrics`` resamples the
back layer into its two half-period column channels, composites each ALONE
through the front comb at +/- a back shift, and reports how far apart the two
channels come out. ``tests/test_barrier_registration.py`` runs it on the
exemplar's own ``generate()`` output; CLAUDE.md's image-switch rule is what it
enforces.

Shift contract (must match frontend/src/shaders/lib/parallax.glsl):
    sinV = sin(tilt); sinSub = sinV / n; cosSub = sqrt(max(0, 1 - sinSub^2))
    shift_um = thickness_um * sinSub / max(0.05, cosSub)
applied along the tilt axis. Composite contract (simplified from plate.frag;
gold masks are grayscale 0..1 where 1 = gold):
    back_shifted = back sampled at (uv - shift)   [zero outside the frame]
    transmission = (1 - front) * (1 - back_shifted)

Pure functions only: no file I/O — callers hand in PIL "L" images or 2D arrays
(see _as_unit_mask).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image


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
def _transmission(front: np.ndarray, back_shifted: np.ndarray) -> np.ndarray:
    return (1.0 - front) * (1.0 - back_shifted)
def _as_unit_mask(mask: Image.Image | np.ndarray) -> np.ndarray:
    """Accept a PIL 'L' image or a 2D array as a unit gold mask.

    Integer arrays follow the PIL convention (0..255 -> /255); float and
    bool arrays are taken as already 0..1 (1 = gold).
    """
    if isinstance(mask, Image.Image):
        # an 'L' image: 0..255 -> 0..1 (the helper this used to call went with
        # the catalogue trim; the conversion is one line)
        return np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
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