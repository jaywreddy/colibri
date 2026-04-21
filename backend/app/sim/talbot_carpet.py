"""Near-field propagation carpet — the (x, y, z) intensity volume past the
back face of the plate, packed as one PNG for the `near_field_carpet`
shader recipe.

Two patterns consume this:
  * tairona-talbot: z_min=0, z_max≈2·z_T, where z_T = 2·Λ²·n/λ is the Talbot
    distance. The mid-z slice is a self-image; the revival structure at
    intermediate z is the "carpet" you see in textbook Talbot diagrams.
  * muzo-emerald-zone: z sweeps past the zone-plate focal length; the slice
    near z = f shows a tight focal spot (six-pointed, shaped by the hex
    crystal aperture).

The atlas layout is a vertical stack of `n_slices` tiles, each `tile_size`
square — the frontend shader picks a row based on `uZSlice` and samples
the tile as a regular 2D texture. One RGB tint per call (single
wavelength), because both target patterns have a physically meaningful
design wavelength and the interesting physics is per-wavelength.

Reuses `angular_spectrum._propagate` verbatim — no new FFT math.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .angular_spectrum import _load_mask, _propagate, _wavelength_to_rgb


@dataclass(frozen=True)
class CarpetParams:
    wavelength_um: float
    z_min_um: float
    z_max_um: float
    n_slices: int
    downsample: int
    tile_size: int
    substrate_thickness_um: float
    substrate_n: float
    layout: str  # "tiles" (2D snapshots per z) or "stripe" (1D x-cut per z)

    def hash(self) -> str:
        payload = {
            "w": self.wavelength_um,
            "zmin": self.z_min_um,
            "zmax": self.z_max_um,
            "n": self.n_slices,
            "ds": self.downsample,
            "ts": self.tile_size,
            "t": self.substrate_thickness_um,
            "ns": self.substrate_n,
            "lo": self.layout,
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:10]


def propagate_carpet(
    variant_dir: Path,
    pixel_pitch_um: float,
    wavelength_um: float,
    z_min_um: float,
    z_max_um: float,
    n_slices: int = 64,
    downsample: int = 2,
    tile_size: int = 128,
    substrate_thickness_um: float = 500.0,
    substrate_n: float = 1.46,
    layout: str = "tiles",
) -> dict:
    """Compute the near-field intensity carpet and save it as an atlas.

    Layouts:
      * "tiles"  — classic 2D snapshots per z, stacked vertically as
        n_slices × tile_size × tile_size. Used by muzo-emerald-zone, whose
        signature is a 2D focal spot.
      * "stripe" — 1D x-line-cut at the centerline of each z-slice, stacked
        vertically as n_slices × W. The output is a single 2D image of the
        (x, z) plane — the canonical Talbot-carpet diagram. Used by
        tairona-talbot.

    Returns {"atlas_name", "rows", "cols", "tile": [h, w], "layout", "cached"}.
    """
    if layout not in {"tiles", "stripe"}:
        raise ValueError(f"layout must be 'tiles' or 'stripe', got {layout!r}")

    params = CarpetParams(
        wavelength_um=wavelength_um,
        z_min_um=z_min_um,
        z_max_um=z_max_um,
        n_slices=n_slices,
        downsample=downsample,
        tile_size=tile_size,
        substrate_thickness_um=substrate_thickness_um,
        substrate_n=substrate_n,
        layout=layout,
    )
    out_name = f"carpet_{params.hash()}.png"
    out_path = variant_dir / out_name

    front = _load_mask(variant_dir / "front.png", downsample)
    back = _load_mask(variant_dir / "back.png", downsample)
    h = min(front.shape[0], back.shape[0])
    w = min(front.shape[1], back.shape[1])
    front = front[:h, :w]
    back = back[:h, :w]

    if out_path.exists():
        if layout == "stripe":
            return {
                "atlas_name": out_name,
                "rows": n_slices,
                "cols": w,
                "tile": [1, w],
                "layout": layout,
                "cached": True,
            }
        return {
            "atlas_name": out_name,
            "rows": n_slices,
            "cols": 1,
            "tile": [tile_size, tile_size],
            "layout": layout,
            "cached": True,
        }
    # Gold is opaque -> aperture amplitude is (1 - gold)
    front_t = (1.0 - front).astype(np.complex64)
    back_t = (1.0 - back).astype(np.complex64)

    dx = pixel_pitch_um * downsample

    # 1) Light crosses the front mask, propagates through the glass substrate,
    # then gets multiplied by the back mask. That's the field *at* the back
    # face; we start the z-sweep from there.
    u_front = front_t
    u_behind_substrate = _propagate(
        u_front, dx, substrate_thickness_um, wavelength_um, substrate_n
    )
    u_back = u_behind_substrate * back_t

    # 2) For each target z past the back face, propagate in air (n=1) from
    # z=0 to that absolute z.
    zs = np.linspace(z_min_um, z_max_um, n_slices)
    rgb = np.asarray(_wavelength_to_rgb(wavelength_um), dtype=np.float32)

    if layout == "stripe":
        # 1D x-cut at the centerline of each z-slice, stacked to form a
        # single (n_slices × W) image of the (x, z) plane — the textbook
        # Talbot carpet diagram. Per-slice 1/99 percentile normalization so
        # every z row has full dynamic range regardless of absolute brightness.
        stripes = np.zeros((n_slices, w), dtype=np.float32)
        ymid = h // 2
        for i, z_um in enumerate(zs):
            if z_um <= 1e-6:
                u_z = u_back
            else:
                u_z = _propagate(u_back, dx, float(z_um), wavelength_um, 1.0)
            irr = np.abs(u_z) ** 2
            row = irr[ymid, :]
            lo, hi = np.percentile(row, [1, 99])
            stripes[i] = np.clip((row - lo) / max(hi - lo, 1e-9), 0, 1)
        # Tint by wavelength for visual aesthetic; the signature is already
        # in the grayscale structure.
        rgb_img = stripes[:, :, None] * rgb
        atlas8 = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(atlas8, "RGB").save(out_path)
        return {
            "atlas_name": out_name,
            "rows": n_slices,
            "cols": w,
            "tile": [1, w],
            "layout": layout,
            "cached": False,
        }

    # layout == "tiles": classic 2D snapshots per z (muzo zone plate).
    crop = min(h, w)
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2

    atlas_h = n_slices * tile_size
    atlas = np.zeros((atlas_h, tile_size, 3), dtype=np.float32)

    slices: list[np.ndarray] = []
    for z_um in zs:
        if z_um <= 1e-6:
            u_z = u_back
        else:
            u_z = _propagate(u_back, dx, float(z_um), wavelength_um, 1.0)
        irr = np.abs(u_z) ** 2
        patch = irr[y0 : y0 + crop, x0 : x0 + crop]
        # Resize to tile_size via block-mean (crop is square, tile_size square).
        if patch.shape[0] != tile_size:
            factor = patch.shape[0] / tile_size
            pil = Image.fromarray((patch / (patch.max() + 1e-12) * 255).astype(np.uint8))
            pil = pil.resize((tile_size, tile_size), Image.Resampling.BOX)
            patch = np.asarray(pil, dtype=np.float32) / 255.0
            patch = patch * factor
        slices.append(patch.astype(np.float32))

    # For muzo: a focal spot is O(100×) brighter than background, so log
    # stretch across a global scale is the correct normalization (unchanged
    # from pre-G behavior).
    g_max = max((s.max() for s in slices), default=1.0)
    if g_max <= 0:
        g_max = 1.0
    log_slices = [np.clip(np.log10(s / g_max + 1e-4) / 4 + 1, 0, 1) for s in slices]

    for i, patch in enumerate(log_slices):
        tinted = patch[:, :, None] * rgb
        atlas[i * tile_size : (i + 1) * tile_size, :, :] = tinted

    atlas8 = (np.clip(atlas, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(atlas8, "RGB").save(out_path)

    return {
        "atlas_name": out_name,
        "rows": n_slices,
        "cols": 1,
        "tile": [tile_size, tile_size],
        "layout": layout,
        "cached": False,
    }
