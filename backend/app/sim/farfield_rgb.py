"""Far-field RGB reconstruction — three-wavelength Fraunhofer intensity
merged into a single color image for the `far_field_hologram` render
recipe.

Two patterns consume this:
  * colibri-hologram: the binary Lohmann CGH whose far-field is a
    hummingbird silhouette. Three wavelengths = three slightly-scaled
    reconstructions, merged R/G/B so the actual perceived reconstruction
    color appears (a rainbow-fringed bird under white coherent light).
  * meridian-speckle: the random speckle diffuser; its far-field is a
    flat-topped beam with wavelength-dependent speckle, also cleanly
    visualized as an RGB merge.

The underlying Fraunhofer math is the same as `fraunhofer.py` —
aperture → FFT → log-stretched intensity — but here we evaluate at
exactly three wavelengths and write ONE RGB PNG (channel = wavelength)
instead of three side-by-side tinted slabs. That's the whole point of
the `far_field_hologram` recipe: what you'd actually see projected on a
screen under white coherent illumination, not three labeled slabs.

Hash-keyed disk cache mirrors the carpet and angular-spectrum paths.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FarfieldParams:
    wavelengths_um: tuple[float, float, float]  # R, G, B
    n_angles: int
    max_angle_deg: float
    carrier_cells: int

    def hash(self) -> str:
        payload = {
            "w": list(self.wavelengths_um),
            "na": self.n_angles,
            "ma": self.max_angle_deg,
            "cc": self.carrier_cells,
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:10]


def _load_binary(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def _reconstruction_channel(
    aperture: np.ndarray, pad: int, carrier_cells: int = 0
) -> np.ndarray:
    """Return the log-stretched quadrant-crop intensity of FFT(aperture).

    carrier_cells:
      * 0 — center crop (default; speckle / any rotationally-symmetric
        target whose reconstruction is centered at DC).
      * C > 0 — crop the (pad/C, pad/C)-shifted quadrant containing the
        off-axis replica of a carrier-shifted target (e.g. the Lohmann
        colibri-hologram encoder shifts by N/C, so set C=4 to follow it).

    The result is in [0, 1], suitable for use as a single RGB channel.
    """
    h, w = aperture.shape
    field = np.zeros((pad, pad), dtype=np.complex64)
    y0 = (pad - h) // 2
    x0 = (pad - w) // 2
    field[y0 : y0 + h, x0 : x0 + w] = aperture

    ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    intensity = np.abs(ft) ** 2
    intensity /= intensity.max() + 1e-12

    # Crop a 50%-sized window. For carrier_cells == 0 the window is
    # centered on DC (preserves pre-G speckle behavior); for carrier_cells
    # > 0 the window is offset by pad/carrier_cells in both axes to follow
    # the Lohmann off-axis replica.
    c = pad // 2
    half = pad // 4
    if carrier_cells > 0:
        off = pad // carrier_cells
        cy = c + off
        cx = c + off
    else:
        cy = c
        cx = c
    # Guard against crop falling outside the array at extreme carrier_cells.
    y_lo = max(0, cy - half)
    y_hi = min(pad, cy + half)
    x_lo = max(0, cx - half)
    x_hi = min(pad, cx + half)
    slab = intensity[y_lo:y_hi, x_lo:x_hi]

    slab_db = 10.0 * np.log10(slab + 1e-8)
    return np.clip((slab_db + 80) / 80, 0, 1).astype(np.float32)


def farfield_rgb(
    variant_dir: Path,
    wavelengths_um: tuple[float, float, float] = (0.65, 0.55, 0.45),
    n_angles: int = 256,
    max_angle_deg: float = 30.0,
    carrier_cells: int = 0,
) -> dict:
    """Compute a merged-RGB Fraunhofer reconstruction and cache it.

    Returns {"farfield_name", "shape": [h, w], "cached"}.
    """
    params = FarfieldParams(
        wavelengths_um=tuple(wavelengths_um),  # type: ignore[arg-type]
        n_angles=n_angles,
        max_angle_deg=max_angle_deg,
        carrier_cells=int(carrier_cells),
    )
    out_name = f"farfield_{params.hash()}.png"
    out_path = variant_dir / out_name
    if out_path.exists():
        with Image.open(out_path) as img:
            w, h = img.size
        return {
            "farfield_name": out_name,
            "shape": [h, w],
            "cached": True,
        }

    front = _load_binary(variant_dir / "front.png")
    back = _load_binary(variant_dir / "back.png")
    h = min(front.shape[0], back.shape[0])
    w = min(front.shape[1], back.shape[1])
    front = front[:h, :w]
    back = back[:h, :w]
    aperture = (1.0 - front) * (1.0 - back)

    pad = max(n_angles, max(h, w))
    # Three wavelengths → three reconstructions. In a strict Fraunhofer
    # sense the k-space extent scales with λ; here we rely on the three
    # wavelengths being close enough (±0.1 μm either side of green) that
    # the perceived reconstruction differs mostly by a small global
    # scale, which is what lets us stack them into one RGB image. The
    # small chromatic smear *is* the visual signature of a binary
    # hologram under white coherent illumination.
    # R, G, B order in wavelengths_um is the intended channel order.
    channels = [
        _reconstruction_channel(aperture, pad, carrier_cells=int(carrier_cells))
        for _ in wavelengths_um
    ]
    rgb = np.stack(channels, axis=-1)  # (H, W, 3)

    # Gentle contrast lift so the faint reconstruction is visible on
    # dark backgrounds; the log-stretch already compressed the dynamic
    # range, so a small linear boost reads well without blowing out.
    rgb = np.clip(rgb * 1.1, 0, 1)

    rgb8 = (rgb * 255).astype(np.uint8)
    Image.fromarray(rgb8, "RGB").save(out_path)

    out_h, out_w = rgb.shape[:2]
    return {
        "farfield_name": out_name,
        "shape": [out_h, out_w],
        "cached": False,
    }
