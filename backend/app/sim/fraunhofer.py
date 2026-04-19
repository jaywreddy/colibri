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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _load_binary(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def fraunhofer_far_field(
    variant_dir: Path,
    wavelengths_um: list[float],
    n_angles: int = 256,
    max_angle_deg: float = 30.0,
) -> dict:
    front = _load_binary(variant_dir / "front.png")
    back = _load_binary(variant_dir / "back.png")
    # Align shapes (rasterize already made them identical, but be defensive)
    h = min(front.shape[0], back.shape[0])
    w = min(front.shape[1], back.shape[1])
    front = front[:h, :w]
    back = back[:h, :w]
    aperture = (1.0 - front) * (1.0 - back)

    # Pad for resolution in angle space (zero-padding -> finer k sampling).
    pad = max(n_angles, max(h, w))
    field = np.zeros((pad, pad), dtype=np.complex64)
    y0 = (pad - h) // 2
    x0 = (pad - w) // 2
    field[y0 : y0 + h, x0 : x0 + w] = aperture

    # Fraunhofer: E(kx, ky) ~ FFT{aperture}
    ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    intensity = np.abs(ft) ** 2
    intensity /= intensity.max() + 1e-12

    # Crop to a window corresponding to max_angle_deg.
    # Spatial sampling of `intensity` is in angle units: Δθ ≈ λ / (pad · pitch).
    # We don't know pitch from the manifest here — caller scales via atlas width.
    # Just take the central 50% as a reasonable default window.
    c = pad // 2
    half = pad // 4
    slab = intensity[c - half : c + half, c - half : c + half]

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
    out_name = f"fft_atlas_{'_'.join(f'{l:.3f}' for l in wavelengths_um)}.png"
    Image.fromarray(atlas8, "RGB").save(variant_dir / out_name)
    return {"atlas_name": out_name, "shape": list(slab.shape)}


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
