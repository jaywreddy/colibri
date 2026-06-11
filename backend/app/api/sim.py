from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..service import DATA_ROOT

# NOTE: the wave-optics sim modules (fraunhofer / angular_spectrum) are
# imported lazily inside the endpoint handlers — they drag in scipy/FFT
# machinery that costs ~0.3 s at import and is never needed by the moiré
# pattern/plate/box hot path.

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
    from ..sim.fraunhofer import fraunhofer_far_field

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
    from ..sim.angular_spectrum import propagate as asm_propagate

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
