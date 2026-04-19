"""Tier 3: Angular-spectrum wave propagation through the 500 um quartz substrate.

Pure-numpy implementation (no LightPipes dependency). For a complex field
U(x,y) sampled on a grid with pitch dx, propagation by distance z in a medium
of refractive index n is a single FFT sandwich:

    U(z) = IFFT{ FFT{U(0)} * exp(i * kz * z) }

where kz = sqrt((n*k0)^2 - kx^2 - ky^2) and k0 = 2*pi/lam.

Pipeline for our plate:
  front_mask (amplitude)  -- propagate 500 um in n=1.46 --> U_before_back
  U_before_back * back_mask                           --> U_after_back
  propagate to observer plane (e.g. 1 mm from back)   --> U_observer

We take |U_observer|^2 to get irradiance. We sweep a set of view angles by
launching a tilted plane wave through front_mask (equivalently, multiplying
the source by exp(i * (kx*x + ky*y))) and capturing the on-axis patch.

Outputs a view-conditioned tiled atlas: one tile per (view_angle, wavelength).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PropagateParams:
    wavelengths_um: tuple[float, ...]
    view_angles_deg: tuple[float, ...]
    observer_distance_um: float
    downsample: int  # downsample front/back by this factor before FFTs
    substrate_thickness_um: float
    substrate_n: float

    def hash(self) -> str:
        payload = {
            "w": self.wavelengths_um,
            "v": self.view_angles_deg,
            "od": self.observer_distance_um,
            "ds": self.downsample,
            "t": self.substrate_thickness_um,
            "n": self.substrate_n,
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:10]


def _load_mask(path: Path, downsample: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if downsample > 1:
        h, w = arr.shape
        h2 = (h // downsample) * downsample
        w2 = (w // downsample) * downsample
        arr = arr[:h2, :w2].reshape(h2 // downsample, downsample, w2 // downsample, downsample).mean(axis=(1, 3))
    return arr


def _propagate(field: np.ndarray, dx_um: float, z_um: float, lam_um: float, n: float) -> np.ndarray:
    """Angular-spectrum propagation of a scalar complex field."""
    h, w = field.shape
    k0 = 2 * np.pi / lam_um
    kx = np.fft.fftfreq(w, d=dx_um) * 2 * np.pi
    ky = np.fft.fftfreq(h, d=dx_um) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    kz_sq = (n * k0) ** 2 - KX**2 - KY**2
    kz = np.sqrt(np.maximum(kz_sq, 0)).astype(np.complex64)
    # Evanescent waves for k^2 < 0 -> imaginary kz, exp decays
    evanescent = kz_sq < 0
    kz = np.where(evanescent, 1j * np.sqrt(np.abs(kz_sq)), kz)
    H = np.exp(1j * kz * z_um).astype(np.complex64)
    F = np.fft.fft2(field)
    return np.fft.ifft2(F * H)


def propagate(
    variant_dir: Path,
    pixel_pitch_um: float,
    wavelengths_um: list[float],
    view_angles_deg: list[float],
    observer_distance_um: float = 1000.0,
    downsample: int = 4,
    substrate_thickness_um: float = 500.0,
    substrate_n: float = 1.46,
) -> dict:
    """Run angular-spectrum propagation and save a tiled atlas.

    The atlas is arranged as (len(view_angles) rows) x (len(wavelengths) cols),
    each tile being a center crop of the observer-plane irradiance.
    """
    params = PropagateParams(
        wavelengths_um=tuple(wavelengths_um),
        view_angles_deg=tuple(view_angles_deg),
        observer_distance_um=observer_distance_um,
        downsample=downsample,
        substrate_thickness_um=substrate_thickness_um,
        substrate_n=substrate_n,
    )
    out_name = f"asm_atlas_{params.hash()}.png"
    out_path = variant_dir / out_name
    if out_path.exists():
        return {"atlas_name": out_name, "cached": True}

    front = _load_mask(variant_dir / "front.png", downsample)
    back = _load_mask(variant_dir / "back.png", downsample)
    h = min(front.shape[0], back.shape[0])
    w = min(front.shape[1], back.shape[1])
    front = front[:h, :w]
    back = back[:h, :w]
    # Gold is opaque -> aperture is (1 - gold)
    front_t = (1.0 - front).astype(np.complex64)
    back_t = (1.0 - back).astype(np.complex64)

    dx = pixel_pitch_um * downsample

    # Crop to central window for display (faster, avoids edge ringing dominating)
    crop = min(h, w) // 2
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2

    tiles: list[list[np.ndarray]] = []
    for theta_deg in view_angles_deg:
        row = []
        theta = np.deg2rad(theta_deg)
        for lam in wavelengths_um:
            k0 = 2 * np.pi / lam
            # Tilted plane wave entering front mask
            xs = (np.arange(w) - w / 2) * dx
            ys = (np.arange(h) - h / 2) * dx
            X, Y = np.meshgrid(xs, ys)
            tilt = np.exp(1j * k0 * np.sin(theta) * X).astype(np.complex64)

            u0 = front_t * tilt
            u1 = _propagate(u0, dx, substrate_thickness_um, lam, substrate_n)
            u2 = u1 * back_t
            u3 = _propagate(u2, dx, observer_distance_um, lam, 1.0)
            irr = np.abs(u3) ** 2
            patch = irr[y0 : y0 + crop, x0 : x0 + crop]
            # Normalize per-tile for visibility
            m = patch.max()
            if m > 0:
                patch = patch / m
            # Log stretch
            patch = np.clip(np.log10(patch + 1e-4) / 4 + 1, 0, 1)
            row.append(patch.astype(np.float32))
        tiles.append(row)

    rows = len(view_angles_deg)
    cols = len(wavelengths_um)
    tile_h, tile_w = tiles[0][0].shape
    atlas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.float32)
    for ri, row in enumerate(tiles):
        for ci, patch in enumerate(row):
            rgb = _wavelength_to_rgb(wavelengths_um[ci])
            tinted = patch[:, :, None] * np.asarray(rgb, dtype=np.float32)
            atlas[ri * tile_h : (ri + 1) * tile_h, ci * tile_w : (ci + 1) * tile_w, :] = tinted

    atlas8 = (np.clip(atlas, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(atlas8, "RGB").save(out_path)

    return {
        "atlas_name": out_name,
        "cached": False,
        "rows": rows,
        "cols": cols,
        "tile": [int(tile_h), int(tile_w)],
    }


def _wavelength_to_rgb(lam_um: float) -> tuple[float, float, float]:
    lam_nm = lam_um * 1000.0
    if lam_nm < 440:
        return (0.3, 0.0, 0.8)
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
    return (1.0, 0.0, 0.0)
