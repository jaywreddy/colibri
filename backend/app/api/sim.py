from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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


# ---------------------------------------------------------------------------
# 2D parallax lab (sim2d) — headless composite + contrast-vs-tilt curves.
# Cheap numpy shifts on the already-rasterized masks; no FFT, no GPU.
# ---------------------------------------------------------------------------


def _load_parallax_pair(slug: str, variant: str) -> tuple[Path, dict]:
    """Resolve a variant dir holding front.png/back.png/manifest.json.

    Only the literal variant 'default' may trigger generation, and only when
    the files are missing — any other miss is a 404. The 2D lab must never
    kick off expensive pattern materialization for arbitrary variant hashes.
    """
    root = DATA_ROOT / slug / variant
    complete = all(
        (root / name).exists() for name in ("manifest.json", "front.png", "back.png")
    )
    if complete:
        return root, json.loads((root / "manifest.json").read_text())
    if variant != "default":
        raise HTTPException(404, f"Variant not found: {slug}/{variant}")

    from .. import service

    try:
        manifest = service.materialize(slug)
    except KeyError as e:
        raise HTTPException(404, f"Unknown pattern: {slug}") from e
    return DATA_ROOT / slug / manifest["variant"], manifest


def _open_masks(root: Path):  # -> tuple[Image, Image]
    from PIL import Image

    front = Image.open(root / "front.png").convert("L")
    back = Image.open(root / "back.png").convert("L")
    return front, back


def _substrate_defaults(
    manifest: dict, thickness_um: float | None, n: float | None
) -> tuple[float, float, float]:
    """(thickness_um, n, pixel_pitch_um) with manifest values as fallbacks."""
    sub = manifest.get("substrate", {})
    t = thickness_um if thickness_um is not None else float(sub["thickness_um"])
    n_val = n if n is not None else float(sub["n"])
    return t, n_val, float(manifest["pixel_pitch_um"])


@router.get("/parallax2d/{slug}/{variant}")
def parallax2d(
    slug: str,
    variant: str,
    tilt_x_deg: float = 0.0,
    tilt_y_deg: float = 0.0,
    illum: str = "ambient",
    thickness_um: float | None = None,
    n: float | None = None,
) -> StreamingResponse:
    from .. import sim2d

    if illum not in sim2d.ILLUMINATIONS:
        raise HTTPException(400, f"Unknown illum: {illum}")
    root, manifest = _load_parallax_pair(slug, variant)
    t, n_val, pitch = _substrate_defaults(manifest, thickness_um, n)
    front, back = _open_masks(root)

    dx_um, dy_um = sim2d.parallax_shift_um(tilt_x_deg, tilt_y_deg, t, n_val)
    img = sim2d.composite_parallax(front, back, dx_um, dy_um, pitch, illum=illum)
    _log.info(
        "parallax2d slug=%s variant=%s tilt=(%.1f,%.1f) illum=%s shift_um=(%.2f,%.2f)",
        slug,
        root.name,
        tilt_x_deg,
        tilt_y_deg,
        illum,
        dx_um,
        dy_um,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/parallax2d/{slug}/{variant}/curve")
def parallax2d_curve(
    slug: str,
    variant: str,
    axis: str = "x",
    thickness_um: float | None = None,
    n: float | None = None,
    points: int = 21,
) -> dict:
    from .. import sim2d

    if axis not in ("x", "y"):
        raise HTTPException(400, f"Unknown axis: {axis}")
    root, manifest = _load_parallax_pair(slug, variant)
    t, n_val, pitch = _substrate_defaults(manifest, thickness_um, n)
    front, back = _open_masks(root)

    points = max(2, min(int(points), 41))
    tilts = [-30.0 + 60.0 * i / (points - 1) for i in range(points)]
    curve = sim2d.contrast_curve(front, back, tilts, t, n_val, pitch, axis=axis)
    return {
        "slug": slug,
        "variant": root.name,
        "axis": axis,
        "thickness_um": t,
        "n": n_val,
        "pixel_pitch_um": pitch,
        "curve": curve,
    }
