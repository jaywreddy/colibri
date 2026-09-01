"""Prepare a photograph for a binary line-screen halftone on gold-on-glass.

Resampling an image and screening it is not enough. A line screen throws away
everything below its own pitch, cannot print 0% or 100%, and quantizes what is
left into a handful of levels — so a photo that goes in unprepared comes out
flat, dark in the midtones, clipped at both ends and banded through the smooth
areas. Every step here is undoing one of those, and each one is measurable at
face scale against the source:

  1. LOCAL CONTRAST (unsharp). The screen discards detail finer than its pitch.
     Putting some back first is the single biggest visible gain: +25% local RMS
     contrast on the reference portrait.
  2. SUBJECT FALLOFF. Optional. A busy background (a flower carpet, a crowd)
     competes with the subject once everything is one colour; a gentle radial
     falloff keeps it as texture.
  3. LINEARIZATION. Gold coverage is linear in AREA, so reflected light is
     linear in coverage, and the eye re-encodes it. The printed appearance is
     therefore ``encode(coverage)``, and matching a source stored in sRGB needs
     ``coverage = linearize(source)``. Feeding sRGB values in as coverage
     directly is the classic error: it renders a 0.25 midtone as 0.54, washing
     the whole image out. (Mapping through L* instead is very nearly a no-op —
     L* and the sRGB transfer curve agree to within 0.02 — which is why it
     looks like it helps and does almost nothing.)
  4. PRINTABLE WINDOW. The finest gold band and the finest gap are both bounded
     by the litho floor, so duty really lives in [1/steps, 1-1/steps].
     Compressing into that window keeps highlight and shadow detail that
     clipping would throw away.
  5. DITHER. Triangular-PDF noise at one quantisation step, which breaks the
     banding a ~22-level screen otherwise puts through skin and sky.

A sixth step only becomes available once a plate exists: DOT-GAIN COMPENSATION.
The duty ladder on the witness plate measures how far process bias moved the
realized duty, and that curve inverts directly into ``gain`` below — which is
why the second run of any image is better than the first.

The output is a DARKNESS map in [0, 1] on the halftone convention used by
``halftone._load_darkness``: larger means more gold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Duty cannot reach 0 or 1: both the gold band and the gap are bounded below by
# the process. One quantisation step at each end is the honest window.
_MIN_EDGE_STEPS = 1.0


@dataclass(frozen=True)
class PrepSpec:
    """Knobs for :func:`prep_darkness`. Defaults are the reference portrait's."""

    tone_steps: int = 22
    """Grey levels the target screen can hold — sets the printable window."""

    unsharp_amount: float = 0.75
    """0 disables. ~0.75 restores what a 44 um screen discards."""
    unsharp_radius_frac: float = 0.0064
    """Blur radius as a fraction of the image side, so it is resolution-free."""

    falloff: float = 0.30
    """0 disables. Radial darkening of a busy background, at the frame edge."""
    falloff_center: tuple[float, float] = (0.52, 0.44)
    falloff_start: float = 0.34
    falloff_span: float = 0.40

    linearize: bool = True
    """Convert sRGB to LINEAR light before it becomes coverage. Off only if the
    source is already linear."""

    clip_percentiles: tuple[float, float] = (1.0, 99.0)
    """Robust black/white points; (0, 100) uses the true extremes."""

    dither: bool = True
    seed: int = 11

    gain: float = 0.0
    """Dot-gain compensation. The fraction of a period the process ADDS to every
    gold band; the prep subtracts it back out so the printed tone lands where it
    was designed. Measure it from the witness plate's duty ladder — a positive
    value means the plate came out darker than asked."""

    def __post_init__(self) -> None:
        if self.tone_steps < 2:
            raise ValueError(f"tone_steps must be >= 2 (got {self.tone_steps})")
        lo, hi = self.clip_percentiles
        if not 0.0 <= lo < hi <= 100.0:
            raise ValueError(f"clip_percentiles must be 0 <= lo < hi <= 100 (got {lo},{hi})")
        if not -0.5 < self.gain < 0.5:
            raise ValueError(f"gain must be in (-0.5, 0.5) (got {self.gain})")


def srgb_to_linear(g: np.ndarray) -> np.ndarray:
    """sRGB grey in [0,1] -> LINEAR light in [0,1]. This is what coverage is."""
    g = np.clip(g, 0.0, 1.0)
    return np.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(y: np.ndarray) -> np.ndarray:
    """LINEAR light -> sRGB. Use it to PREVIEW a coverage map: the plate
    reflects light in proportion to coverage and the eye re-encodes, so a
    coverage array shown raw looks far darker than the finished plate."""
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055)


def printable_window(tone_steps: int) -> tuple[float, float]:
    """(lo, hi) duty the screen can actually realize."""
    edge = _MIN_EDGE_STEPS / float(tone_steps)
    return edge, 1.0 - edge


def prep_darkness(gray: np.ndarray, spec: PrepSpec = PrepSpec()) -> np.ndarray:
    """Run the pipeline on a float grey image in [0,1]; return darkness in [0,1].

    Order matters. Local contrast runs on the ORIGINAL tones (sharpening after
    the L* lift would amplify what the lift already stretched), the window is
    applied after L* (it is a duty range, not a luminance range), and the dither
    is last so nothing smooths it back out.
    """
    x = np.clip(np.asarray(gray, dtype=np.float32), 0.0, 1.0)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D grey image, got shape {x.shape}")

    # 1. local contrast
    if spec.unsharp_amount > 0.0:
        radius = max(1.0, spec.unsharp_radius_frac * max(x.shape))
        blur = np.asarray(
            Image.fromarray((x * 255).astype(np.uint8), "L").filter(
                ImageFilter.GaussianBlur(radius)
            ),
            dtype=np.float32,
        ) / 255.0
        x = np.clip(x + spec.unsharp_amount * (x - blur), 0.0, 1.0)

    # 2. subject falloff
    if spec.falloff > 0.0:
        h, w = x.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / float(max(h, w))
        cx, cy = spec.falloff_center
        r = np.hypot(xx - cx, yy - cy)
        t = np.clip((r - spec.falloff_start) / max(1e-6, spec.falloff_span), 0.0, 1.0)
        x = np.clip(x * (1.0 - spec.falloff * t), 0.0, 1.0)

    # 3. linearization -- from here on x is LIGHT, not an sRGB code value, and
    # every remaining step is an operation on coverage.
    if spec.linearize:
        x = srgb_to_linear(x)

    # 4. printable window
    lo_p, hi_p = spec.clip_percentiles
    p_lo, p_hi = np.percentile(x, (lo_p, hi_p))
    x = np.clip((x - p_lo) / max(1e-6, float(p_hi - p_lo)), 0.0, 1.0)
    lo, hi = printable_window(spec.tone_steps)
    x = x * (hi - lo) + lo

    # 6. dot gain (before the dither, so the correction is on the signal)
    if spec.gain:
        x = np.clip(x - spec.gain, lo, hi)

    # 5. dither -- clipped back into the WINDOW, not into [0,1]. Dithering last
    # and clipping to [0,1] would hand the screen duty values it cannot realize,
    # undoing step 4 on exactly the highlights and shadows that step existed to
    # protect.
    if spec.dither:
        rng = np.random.default_rng(spec.seed)
        n = (rng.random(x.shape, dtype=np.float32)
             - rng.random(x.shape, dtype=np.float32)) / float(spec.tone_steps)
        x = np.clip(x + n, lo, hi)

    return x.astype(np.float32)


def load_gray(
    path: str | Path,
    *,
    crop: tuple[float, float, float] | None = None,
    size: int = 1400,
) -> np.ndarray:
    """Load an image as a square float grey array.

    ``crop`` is ``(x_frac, y_frac, side_frac)`` of the SOURCE WIDTH, so a crop
    chosen on one copy of a photo survives a re-export at another resolution.
    """
    im = Image.open(Path(path))
    if crop is not None:
        w, h = im.size
        fx, fy, fs = crop
        x0, y0 = int(fx * w), int(fy * h)
        side = int(fs * w)
        im = im.crop((x0, y0, min(w, x0 + side), min(h, y0 + side)))
    g = ImageOps.grayscale(im).resize((size, size), Image.LANCZOS)
    return np.asarray(g, dtype=np.float32) / 255.0


def prepare_asset(
    src: str | Path,
    dst: str | Path,
    *,
    crop: tuple[float, float, float] | None = None,
    size: int = 1400,
    spec: PrepSpec = PrepSpec(),
) -> dict[str, float]:
    """Prep ``src`` and write it as a halftone-ready 8-bit asset at ``dst``.

    Returns the tone statistics, so a caller can see what the prep did without
    reopening the file.
    """
    gray = load_gray(src, crop=crop, size=size)
    dark = prep_darkness(gray, spec)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((dark * 255).astype(np.uint8), "L").save(Path(dst))
    lo, hi = printable_window(spec.tone_steps)
    return {
        "mean": float(dark.mean()),
        "std": float(dark.std()),
        "min": float(dark.min()),
        "max": float(dark.max()),
        "window_lo": lo,
        "window_hi": hi,
        "tone_steps": float(spec.tone_steps),
    }


def screen_period_for(tone_steps: int, min_band_um: float = 2.0) -> float:
    """Coarsest useful line period for a given depth — the inverse of the cap.

    Tone depth is bounded by ``steps <= period / min_band``, and the eye's
    integration cell (about 87 um at 300 mm) must span at least two periods or
    the screen stops averaging away. Both meet near 44 um / 22 steps.
    """
    return float(tone_steps) * min_band_um


def local_contrast(a: np.ndarray, radius: float = 3.0) -> float:
    """RMS deviation from a local mean — the 'pop' metric the prep optimizes."""
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L")
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    return float(np.sqrt(((np.clip(a, 0, 1) - blur) ** 2).mean()))
