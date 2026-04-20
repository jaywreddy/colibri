from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..sim.fraunhofer import fraunhofer_far_field
from ..sim.angular_spectrum import propagate as asm_propagate
from ..sim.talbot_carpet import propagate_carpet
from ..sim.farfield_rgb import farfield_rgb
from ..service import DATA_ROOT

router = APIRouter(prefix="/sim", tags=["sim"])
_log = logging.getLogger("optics.sim")


class FftRequest(BaseModel):
    slug: str
    variant: str
    wavelengths_um: list[float] = [0.65, 0.55, 0.45]
    n_angles: int = 256
    max_angle_deg: float = 30.0


@router.post("/fft")
def fft_sim(req: FftRequest) -> dict:
    root = DATA_ROOT / req.slug / req.variant
    if not root.exists():
        raise HTTPException(404, f"Variant not found: {req.slug}/{req.variant}")
    t0 = time.perf_counter()
    _log.info("fft start slug=%s variant=%s wl=%s", req.slug, req.variant, req.wavelengths_um)
    try:
        out = fraunhofer_far_field(
            root,
            wavelengths_um=req.wavelengths_um,
            n_angles=req.n_angles,
            max_angle_deg=req.max_angle_deg,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(
            "fft failed slug=%s variant=%s err=%r (%dms)",
            req.slug,
            req.variant,
            e,
            int((time.perf_counter() - t0) * 1000),
        )
        raise HTTPException(400, f"FFT failed: {e!r}") from e
    _log.info(
        "fft done slug=%s variant=%s cached=%s %dms",
        req.slug,
        req.variant,
        out.get("cached", False),
        int((time.perf_counter() - t0) * 1000),
    )
    return {
        "slug": req.slug,
        "variant": req.variant,
        "wavelengths_um": req.wavelengths_um,
        "atlas_png": f"/data/{req.slug}/{req.variant}/{out['atlas_name']}",
        "max_angle_deg": req.max_angle_deg,
    }


class PropagateRequest(BaseModel):
    slug: str
    variant: str
    wavelengths_um: list[float] = [0.65, 0.55, 0.45]
    view_angles_deg: list[float] = [-15.0, 0.0, 15.0]
    observer_distance_um: float = 1000.0
    downsample: int = 4


@router.post("/propagate")
def propagate(req: PropagateRequest) -> dict:
    root = DATA_ROOT / req.slug / req.variant
    if not root.exists():
        raise HTTPException(404, f"Variant not found: {req.slug}/{req.variant}")
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Variant missing manifest")
    manifest = json.loads(manifest_path.read_text())
    t0 = time.perf_counter()
    _log.info(
        "propagate start slug=%s variant=%s wl=%s angles=%s",
        req.slug,
        req.variant,
        req.wavelengths_um,
        req.view_angles_deg,
    )
    try:
        out = asm_propagate(
            root,
            pixel_pitch_um=float(manifest["pixel_pitch_um"]),
            wavelengths_um=req.wavelengths_um,
            view_angles_deg=req.view_angles_deg,
            observer_distance_um=req.observer_distance_um,
            downsample=req.downsample,
            substrate_thickness_um=float(manifest["substrate"]["thickness_um"]),
            substrate_n=float(manifest["substrate"]["n"]),
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(
            "propagate failed slug=%s variant=%s err=%r (%dms)",
            req.slug,
            req.variant,
            e,
            int((time.perf_counter() - t0) * 1000),
        )
        raise HTTPException(400, f"Propagate failed: {e!r}") from e
    _log.info(
        "propagate done slug=%s variant=%s cached=%s %dms",
        req.slug,
        req.variant,
        out.get("cached", False),
        int((time.perf_counter() - t0) * 1000),
    )
    return {
        "slug": req.slug,
        "variant": req.variant,
        "wavelengths_um": req.wavelengths_um,
        "view_angles_deg": req.view_angles_deg,
        "atlas_png": f"/data/{req.slug}/{req.variant}/{out['atlas_name']}",
        "rows": out.get("rows"),
        "cols": out.get("cols"),
        "tile": out.get("tile"),
        "cached": out.get("cached", False),
    }


class CarpetRequest(BaseModel):
    slug: str
    variant: str
    wavelength_um: float = 0.55
    z_min_um: float = 0.0
    z_max_um: float = 4000.0
    n_slices: int = 64
    downsample: int = 8
    tile_size: int = 128


@router.post("/carpet")
def carpet(req: CarpetRequest) -> dict:
    """Near-field propagation carpet past the back face of the plate.

    Used by the `near_field_carpet` render recipe (tairona-talbot,
    muzo-emerald-zone). Returns a vertical atlas of `n_slices` 2D tiles
    showing the intensity distribution at each z. The frontend shader
    picks a row based on `uZSlice` and the SecondaryView panel displays
    the full carpet with a slider indicator.
    """
    root = DATA_ROOT / req.slug / req.variant
    if not root.exists():
        raise HTTPException(404, f"Variant not found: {req.slug}/{req.variant}")
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Variant missing manifest")
    manifest = json.loads(manifest_path.read_text())
    if req.z_max_um <= req.z_min_um:
        raise HTTPException(400, "z_max_um must exceed z_min_um")
    if req.n_slices < 2:
        raise HTTPException(400, "n_slices must be >= 2")
    t0 = time.perf_counter()
    _log.info(
        "carpet start slug=%s variant=%s wl=%s z=[%s,%s] n=%d",
        req.slug,
        req.variant,
        req.wavelength_um,
        req.z_min_um,
        req.z_max_um,
        req.n_slices,
    )
    try:
        out = propagate_carpet(
            root,
            pixel_pitch_um=float(manifest["pixel_pitch_um"]),
            wavelength_um=req.wavelength_um,
            z_min_um=req.z_min_um,
            z_max_um=req.z_max_um,
            n_slices=req.n_slices,
            downsample=req.downsample,
            tile_size=req.tile_size,
            substrate_thickness_um=float(manifest["substrate"]["thickness_um"]),
            substrate_n=float(manifest["substrate"]["n"]),
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(
            "carpet failed slug=%s variant=%s err=%r (%dms)",
            req.slug,
            req.variant,
            e,
            int((time.perf_counter() - t0) * 1000),
        )
        raise HTTPException(400, f"Carpet failed: {e!r}") from e
    _log.info(
        "carpet done slug=%s variant=%s cached=%s %dms",
        req.slug,
        req.variant,
        out.get("cached", False),
        int((time.perf_counter() - t0) * 1000),
    )
    return {
        "slug": req.slug,
        "variant": req.variant,
        "wavelength_um": req.wavelength_um,
        "z_min_um": req.z_min_um,
        "z_max_um": req.z_max_um,
        "atlas_png": f"/data/{req.slug}/{req.variant}/{out['atlas_name']}",
        "rows": out.get("rows"),
        "cols": out.get("cols"),
        "tile": out.get("tile"),
        "cached": out.get("cached", False),
    }


class FarfieldRequest(BaseModel):
    slug: str
    variant: str
    # R, G, B wavelengths. Default picks standard laser-printing trichromat
    # so the reconstruction looks natural under "white coherent" illumination.
    wavelengths_um: list[float] = [0.65, 0.55, 0.45]
    n_angles: int = 256
    max_angle_deg: float = 30.0


@router.post("/farfield")
def farfield(req: FarfieldRequest) -> dict:
    """Merged-RGB Fraunhofer reconstruction for `far_field_hologram`.

    Used by colibri-hologram and meridian-speckle. Returns a single RGB
    PNG (channel = wavelength), suitable for direct <img> display in
    SecondaryView — the visualization of "what you'd see on a screen"
    for a binary CGH under white coherent light.
    """
    root = DATA_ROOT / req.slug / req.variant
    if not root.exists():
        raise HTTPException(404, f"Variant not found: {req.slug}/{req.variant}")
    if len(req.wavelengths_um) != 3:
        raise HTTPException(400, "wavelengths_um must have exactly 3 entries (R, G, B)")
    t0 = time.perf_counter()
    _log.info(
        "farfield start slug=%s variant=%s wl=%s",
        req.slug,
        req.variant,
        req.wavelengths_um,
    )
    try:
        out = farfield_rgb(
            root,
            wavelengths_um=tuple(req.wavelengths_um),  # type: ignore[arg-type]
            n_angles=req.n_angles,
            max_angle_deg=req.max_angle_deg,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(
            "farfield failed slug=%s variant=%s err=%r (%dms)",
            req.slug,
            req.variant,
            e,
            int((time.perf_counter() - t0) * 1000),
        )
        raise HTTPException(400, f"Farfield failed: {e!r}") from e
    _log.info(
        "farfield done slug=%s variant=%s cached=%s %dms",
        req.slug,
        req.variant,
        out.get("cached", False),
        int((time.perf_counter() - t0) * 1000),
    )
    return {
        "slug": req.slug,
        "variant": req.variant,
        "wavelengths_um": req.wavelengths_um,
        "farfield_png": f"/data/{req.slug}/{req.variant}/{out['farfield_name']}",
        "shape": out.get("shape"),
        "cached": out.get("cached", False),
    }
