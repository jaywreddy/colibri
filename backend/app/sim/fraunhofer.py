"""Tier 2: Fraunhofer far-field approximation of the combined front+back aperture.

This is intentionally simple and fast. For each requested wavelength we:
  1. Load both layer PNGs.
  2. Form the aperture transmission: t(x,y) = (1 - front) * (1 - back)
     (gold is opaque, so transmission is 1 minus the gold fraction on each side).
     We ignore the 500 μm gap phase at this tier — it's a pure Fraunhofer
     intensity pattern, useful for predicting iridescence and CGH
     reconstructions. The Tier 3 angular-spectrum engine handles the gap.
  3. Take |FFT|² and log-stretch for display.
  4. Stack per-wavelength slabs into a horizontal atlas PNG and save.

One FFT serves every wavelength: |FFT{t}|² lives in spatial-frequency (bin)
space, which is wavelength-independent. Only the bin -> angle mapping carries λ,
via sin θ = λ·f, so each slab is a *different angular window* of the same
diffraction pattern. The window is sized so no slab exceeds the requested
``max_angle_deg``; the achieved per-λ half-angles and the (constant) sin θ per
pixel come back in the result so a caller can put a real angular scale on the
atlas instead of guessing.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import MAX_ATLAS_CELLS, check_fft_budget
from .cache import key as cache_key


def _load_binary(path: Path, h: int, w: int) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return arr[:h, :w]


def _block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Area-average `arr` by `factor` (same construction as angular_spectrum)."""
    if factor <= 1:
        return arr
    h = (arr.shape[0] // factor) * factor
    w = (arr.shape[1] // factor) * factor
    return (
        arr[:h, :w]
        .reshape(h // factor, factor, w // factor, factor)
        .mean(axis=(1, 3))
    )


def fraunhofer_far_field(
    variant_dir: Path,
    wavelengths_um: list[float],
    pixel_pitch_um: float,
    n_angles: int = 256,
    max_angle_deg: float = 30.0,
    downsample: int = 4,
) -> dict:
    """|FFT|² of the combined aperture, cropped to a real angular window.

    ``pixel_pitch_um`` is the variant's raster pitch (from its manifest) —
    without it there is no bin -> angle mapping and ``max_angle_deg`` cannot be
    honored. ``downsample`` area-averages the fine transmission before the FFT;
    it is what keeps the padded grid inside the memory budget, at the cost of
    angular range (Nyquist is sin θ = λ / 2Δx with Δx = pitch·downsample).
    """
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1 (got {downsample})")
    if pixel_pitch_um <= 0:
        raise ValueError(f"pixel_pitch_um must be > 0 (got {pixel_pitch_um})")
    if not wavelengths_um:
        raise ValueError("wavelengths_um must not be empty")
    if min(wavelengths_um) <= 0:
        raise ValueError(f"wavelengths_um must all be > 0 (got {wavelengths_um})")
    if not 0.0 < max_angle_deg < 90.0:
        raise ValueError(f"max_angle_deg must be in (0, 90) (got {max_angle_deg})")

    # Geometry first, from the PNG headers only — the budget check and the cache
    # short-circuit must both land before we decode or allocate anything.
    with Image.open(variant_dir / "front.png") as img:
        fw, fh = img.size
    with Image.open(variant_dir / "back.png") as img:
        bw, bh = img.size
    h_fine, w_fine = min(fh, bh), min(fw, bw)
    h, w = h_fine // downsample, w_fine // downsample
    if h < 1 or w < 1:
        raise ValueError(
            f"downsample={downsample} collapses the {w_fine}x{h_fine} raster to nothing"
        )

    # Pad for resolution in angle space (zero-padding -> finer k sampling).
    pad = max(n_angles, h, w)
    check_fft_budget(
        pad * pad,
        "Fraunhofer far field",
        n_angles=n_angles,
        raster=max(h, w),
        downsample=downsample,
    )

    dx = pixel_pitch_um * downsample
    # Bin k of the shifted FFT sits at spatial frequency k/(pad·Δx), i.e. at
    # sin θ = k·λ/(pad·Δx). Size the crop off the LONGEST wavelength so no slab
    # runs past max_angle_deg; shorter λ then cover proportionally less angle.
    sin_per_px_ref = max(wavelengths_um) / (pad * dx)
    half = int(math.sin(math.radians(max_angle_deg)) / sin_per_px_ref)
    half = max(1, min(pad // 2, half))  # Nyquist clamp: |sin θ| <= λ/(2Δx)
    check_fft_budget(
        (2 * half) ** 2 * len(wavelengths_um),
        "Fraunhofer atlas",
        cap=MAX_ATLAS_CELLS,
        wavelengths=len(wavelengths_um),
        slab=2 * half,
    )

    scale = {
        "shape": [2 * half, 2 * half],
        "requested_max_angle_deg": max_angle_deg,
        # Achieved half-angle per wavelength (the reference λ hits the request
        # unless Nyquist clamped it); sin θ per pixel is exactly constant.
        "half_angle_deg": [
            math.degrees(math.asin(min(1.0, half * lam / (pad * dx))))
            for lam in wavelengths_um
        ],
        "sin_theta_per_px": [lam / (pad * dx) for lam in wavelengths_um],
        "nyquist_half_angle_deg": [
            math.degrees(math.asin(min(1.0, lam / (2 * dx)))) for lam in wavelengths_um
        ],
        "pad": pad,
        "downsample": downsample,
        "pixel_pitch_um": pixel_pitch_um,
    }

    out_name = "fft_atlas_{}.png".format(
        cache_key(
            {
                "wl": list(wavelengths_um),
                "na": n_angles,
                "ma": max_angle_deg,
                "ds": downsample,
                "px": pixel_pitch_um,
            }
        )
    )
    out_path = variant_dir / out_name
    if out_path.exists():
        return {"atlas_name": out_name, "cached": True, **scale}

    aperture = _block_mean(
        (1.0 - _load_binary(variant_dir / "front.png", h_fine, w_fine))
        * (1.0 - _load_binary(variant_dir / "back.png", h_fine, w_fine)),
        downsample,
    )
    field = np.zeros((pad, pad), dtype=np.complex64)
    y0 = (pad - h) // 2
    x0 = (pad - w) // 2
    field[y0 : y0 + h, x0 : x0 + w] = aperture
    del aperture

    # Fraunhofer: E(kx, ky) ~ FFT{aperture}
    ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    del field
    c = pad // 2
    # The DC bin is the global maximum for a real non-negative transmission
    # (it is the integral of it), so normalizing by |ft[c, c]|² avoids a
    # full-array abs() temporary just to find the peak.
    peak = float(np.abs(ft[c, c])) ** 2
    slab = np.abs(ft[c - half : c + half, c - half : c + half]).astype(np.float32) ** 2
    del ft
    slab /= peak + 1e-12

    # Log-stretch for visibility
    slab_db = 10.0 * np.log10(slab + 1e-8)
    slab_db = np.clip((slab_db + 80) / 80, 0, 1)  # -80 dB..0 dB -> 0..1

    # Per-wavelength colored composite (RGB channels weighted by wavelength)
    atlas = np.zeros((slab.shape[0], slab.shape[1] * len(wavelengths_um), 3), dtype=np.float32)
    for i, lam in enumerate(wavelengths_um):
        rgb = _wavelength_to_rgb(lam)
        tinted = slab_db[:, :, None] * np.asarray(rgb, dtype=np.float32)
        atlas[:, i * slab.shape[1] : (i + 1) * slab.shape[1], :] = tinted

    atlas8 = (atlas * 255).astype(np.uint8)
    Image.fromarray(atlas8, "RGB").save(out_path)
    return {"atlas_name": out_name, "cached": False, **scale}


def _wavelength_to_rgb(lam_um: float) -> tuple[float, float, float]:
    """Rough visible-spectrum color for a given wavelength in μm."""
    lam_nm = lam_um * 1000.0
    if lam_nm < 380:
        return (0.3, 0.0, 0.5)
    if lam_nm < 440:
        t = (lam_nm - 380) / 60
        return (0.3 * (1 - t), 0.0, 0.5 + 0.5 * t)
    if lam_nm < 490:
        t = (lam_nm - 440) / 50
        return (0.0, t, 1.0)
    if lam_nm < 510:
        t = (lam_nm - 490) / 20
        return (0.0, 1.0, 1.0 - t)
    if lam_nm < 580:
        t = (lam_nm - 510) / 70
        return (t, 1.0, 0.0)
    if lam_nm < 645:
        t = (lam_nm - 580) / 65
        return (1.0, 1.0 - t, 0.0)
    if lam_nm < 750:
        return (1.0, 0.0, 0.0)
    return (0.5, 0.0, 0.0)
