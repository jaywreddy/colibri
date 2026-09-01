from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..service import DATA_ROOT

# NOTE: the wave-optics sim modules (fraunhofer / angular_spectrum) are
# imported lazily inside the endpoint handlers — they drag in scipy/FFT
# machinery that costs ~0.3 s at import and is never needed by the moiré
# pattern/plate/box hot path.

router = APIRouter(prefix="/sim", tags=["sim"])
_log = logging.getLogger("optics.sim")

# Every sim dial that multiplies into an FFT allocation is bounded here, not
# just budgeted downstream: an unbounded n_angles alone can ask for a
# multi-gigabyte square complex field, and this host bugchecks before the
# allocation ever fails. The kernels still carry their own byte-budget guards
# (sim.check_fft_budget) because the variant raster feeds the grid size too.
Wavelength = Annotated[float, Field(ge=0.2, le=2.0)]
ViewAngle = Annotated[float, Field(ge=-89.0, le=89.0)]
MAX_SIM_TILES = 24  # (view angles x wavelengths) tiles per /sim/propagate call


def _load_manifest(slug: str, variant: str) -> tuple[Path, dict]:
    root = DATA_ROOT / slug / variant
    if not root.exists():
        raise HTTPException(404, f"Variant not found: {slug}/{variant}")
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Variant missing manifest")
    return root, json.loads(manifest_path.read_text())


class FftRequest(BaseModel):
    slug: str
    variant: str
    wavelengths_um: list[Wavelength] = Field(
        default=[0.65, 0.55, 0.45], min_length=1, max_length=8
    )
    n_angles: int = Field(256, ge=64, le=2048)
    max_angle_deg: float = Field(30.0, gt=0.0, lt=90.0)
    downsample: int = Field(4, ge=1, le=16)


@router.post("/fft")
def fft_sim(req: FftRequest) -> dict:
    from ..sim.fraunhofer import fraunhofer_far_field

    root, manifest = _load_manifest(req.slug, req.variant)
    t0 = time.perf_counter()
    _log.info("fft start slug=%s variant=%s wl=%s", req.slug, req.variant, req.wavelengths_um)
    try:
        out = fraunhofer_far_field(
            root,
            wavelengths_um=req.wavelengths_um,
            pixel_pitch_um=float(manifest["pixel_pitch_um"]),
            n_angles=req.n_angles,
            max_angle_deg=req.max_angle_deg,
            downsample=req.downsample,
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
    # The atlas is one shared |FFT|² in bin space, so each wavelength slab spans
    # a different angular window (sin θ = λ·f). We report the half-angle each
    # slab actually achieved instead of echoing the request as if it were law.
    return {
        "slug": req.slug,
        "variant": req.variant,
        "wavelengths_um": req.wavelengths_um,
        "atlas_png": f"/data/{req.slug}/{req.variant}/{out['atlas_name']}",
        "requested_max_angle_deg": req.max_angle_deg,
        "half_angle_deg": out["half_angle_deg"],
        "sin_theta_per_px": out["sin_theta_per_px"],
        "nyquist_half_angle_deg": out["nyquist_half_angle_deg"],
        "shape": out["shape"],
        "downsample": out["downsample"],
        "cached": out["cached"],
    }


class PropagateRequest(BaseModel):
    slug: str
    variant: str
    wavelengths_um: list[Wavelength] = Field(
        default=[0.65, 0.55, 0.45], min_length=1, max_length=8
    )
    view_angles_deg: list[ViewAngle] = Field(
        default=[-15.0, 0.0, 15.0], min_length=1, max_length=16
    )
    observer_distance_um: float = Field(1000.0, gt=0.0, le=100_000.0)
    downsample: int = Field(4, ge=1, le=16)

    @model_validator(mode="after")
    def _bound_tile_count(self) -> PropagateRequest:
        tiles = len(self.view_angles_deg) * len(self.wavelengths_um)
        if tiles > MAX_SIM_TILES:
            raise ValueError(
                f"{len(self.view_angles_deg)} view angles x "
                f"{len(self.wavelengths_um)} wavelengths = {tiles} tiles, each a "
                f"four-FFT round over the whole grid; the cap is {MAX_SIM_TILES}."
            )
        return self


@router.post("/propagate")
def propagate(req: PropagateRequest) -> dict:
    from ..sim.angular_spectrum import propagate as asm_propagate

    root, manifest = _load_manifest(req.slug, req.variant)
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


# A variant raster runs to the rasterizer's 16 Mpx cap, and the curve endpoint
# composites the whole frame once per tilt point (up to 41). Area-average both
# masks to at most this many pixels per side before sweeping: the back-layer
# shift is a whole-pixel roll, so a coarser grid costs shift RESOLUTION
# (pitch * factor), not correctness, and 512 px still resolves the beat envelope
# every metric here reads.
SIM2D_MAX_SWEEP_SIDE = 512


def _mask_arrays(
    root: Path,
    pixel_pitch_um: float,
    downsample: int = 1,
    max_side: int | None = None,
):  # -> tuple[ndarray, ndarray, float, int]
    """Both masks as 0..1 float arrays, area-averaged, plus the grid pitch.

    The applied factor is ``downsample`` widened to whatever also brings the
    longest side under ``max_side`` (None = honor the request only). Same
    reduction as ``sim.angular_spectrum._load_mask`` — crop to a whole multiple
    of the factor, then block-mean — so the 2D lab and the wave sims coarsen a
    raster identically, and the grid pitch is pitch * factor exactly as there.
    """
    import numpy as np

    front, back = _open_masks(root)
    arrays = [np.asarray(img, dtype=np.float32) / 255.0 for img in (front, back)]
    factor = max(1, int(downsample))
    if max_side is not None:
        side = max(max(a.shape) for a in arrays)
        factor = max(factor, -(-side // max_side))
    if factor > 1:
        reduced = []
        for arr in arrays:
            h = (arr.shape[0] // factor) * factor
            w = (arr.shape[1] // factor) * factor
            if not h or not w:  # factor bigger than the raster — leave it alone
                factor = 1
                reduced = arrays
                break
            reduced.append(
                arr[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(
                    axis=(1, 3)
                )
            )
        arrays = reduced
    return arrays[0], arrays[1], pixel_pitch_um * factor, factor


def _substrate_defaults(
    manifest: dict, thickness_um: float | None, n: float | None
) -> tuple[float, float, float]:
    """(thickness_um, n, pixel_pitch_um) with manifest values as fallbacks."""
    sub = manifest.get("substrate", {})
    t = thickness_um if thickness_um is not None else float(sub["thickness_um"])
    n_val = n if n is not None else float(sub["n"])
    return t, n_val, float(manifest["pixel_pitch_um"])


@router.get("/readability/{box_id}")
def readability(
    box_id: str,
    distance_mm: float = 300.0,
    pupil_mm: float = 3.0,
) -> dict:
    """Will this box's faces READ to a human eye, face by face?

    Computed from each face's FABRICATED periods (the manifest's recipe_data)
    and the box's real glass — never from the render, which systematically
    understates the effect (it filters the two layers independently, so it
    loses the correlation term the physical part keeps). Every barrier face
    gets the switch budget, every face gets the frame-moire budget.
    """
    from ..boxes import get_box
    from ..readability import moire_readability, switch_readability, verdict

    box = get_box(box_id)
    if box is None:
        raise HTTPException(404, f"Unknown box: {box_id}")
    glass = (box.get("spec") or {}).get("glass") or {}
    t = float(glass.get("thickness_um", 500.0))
    n = float(glass.get("n", 1.46)) or 1.46

    reports = []
    for fid, fm in (box.get("faces") or {}).items():
        rd = fm.get("recipe_data") or {}
        num = lambda k, d: float(rd.get(k, d) or d)  # noqa: E731
        # Barrier faces: the A/B interlace and the scanimation both switch by
        # walking a barrier, so both take the switch budget.
        is_barrier = bool(rd.get("switch_interlace")) or num("water_scan_n", 0) > 0
        if is_barrier:
            pitch = num("switch_interlace_period_um", 0) or num("fab_center_period_um", 60.0)
            reports.append(
                switch_readability(
                    face=fid, barrier_pitch_um=pitch, thickness_um=t, n=n,
                    distance_mm=distance_mm, pupil_mm=pupil_mm,
                )
            )
        # The frame encodes a DIFFERENT louvre orientation per motif species
        # (plate.frag: frameAngle = slitAngle + (bucket - (count-1)/2) * span),
        # so one face fabricates a FAN of crossing angles. Evaluating only the
        # base offset reported "passes" for every face while individual leaf
        # species sat below acuity — the gate has to judge every direction it
        # actually writes.
        base = num("fab_angle_offset_deg", 3.0)
        span = num("frame_angle_span_deg", 3.5)
        count = int(num("frame_bucket_count", 6)) or 6
        offsets = [base + (b - (count - 1) / 2.0) * span for b in range(count)]
        reports.append(
            moire_readability(
                face=fid,
                back_pitch_um=num("fab_back_period_um", 22.0),
                front_pitch_um=num("fab_front_period_um", 23.98),
                angle_offset_deg=base,
                thickness_um=t, n=n, distance_mm=distance_mm, pupil_mm=pupil_mm,
                angle_offsets_deg=offsets,
            )
        )
    out = verdict(reports)
    out["viewing"] = {"distance_mm": distance_mm, "pupil_mm": pupil_mm}
    out["glass"] = {"thickness_um": t, "n": n}
    return out


@router.get("/diffraction/lut")
def diffraction_lut(
    duty: float = 0.5,
    u_max_um: float = 10.0,
    size: int = 1024,
    orders: int = 16,
) -> dict:
    """Baked diffraction colour table for the renderer's spectral accent.

    The physics (grating equation, square-wave order series, CIE integration)
    runs in ``app.diffraction`` where it is unit-tested against closed forms;
    the shader does one texture fetch. Indexed by the optical path term
    ``u = period * (V.g + L.g)`` in um, so ONE table serves every pitch — see
    the module docstring. Values are LINEAR sRGB.
    """
    from ..diffraction import lut_payload

    if not (0.0 < duty < 1.0):
        raise HTTPException(400, f"duty must be in (0, 1) (got {duty})")
    if not (2 <= size <= 4096):
        raise HTTPException(400, f"size must be 2..4096 (got {size})")
    if not (1 <= orders <= 64):
        raise HTTPException(400, f"orders must be 1..64 (got {orders})")
    if not (0.1 <= u_max_um <= 100.0):
        raise HTTPException(400, f"u_max_um must be 0.1..100 (got {u_max_um})")
    return lut_payload(duty=duty, u_max_um=u_max_um, size=size, orders=orders)


@router.get("/parallax2d/{slug}/{variant}")
def parallax2d(
    slug: str,
    variant: str,
    tilt_x_deg: float = 0.0,
    tilt_y_deg: float = 0.0,
    illum: str = "ambient",
    thickness_um: float | None = None,
    n: float | None = None,
    downsample: int = 1,
) -> StreamingResponse:
    """One composite at a given tilt.

    ``downsample`` area-averages both masks by that factor first (grid pitch
    becomes pitch * downsample, mirroring PropagateRequest.downsample) — the way
    to keep a plate-sized raster off this host's memory. It defaults to 1, so
    the default response is the full-resolution composite.
    """
    from .. import sim2d

    if illum not in sim2d.ILLUMINATIONS:
        raise HTTPException(400, f"Unknown illum: {illum}")
    if not 1 <= downsample <= 16:
        raise HTTPException(400, f"downsample must be 1..16, got {downsample}")
    root, manifest = _load_parallax_pair(slug, variant)
    t, n_val, pitch = _substrate_defaults(manifest, thickness_um, n)
    front, back, grid_pitch, factor = _mask_arrays(root, pitch, downsample=downsample)

    dx_um, dy_um = sim2d.parallax_shift_um(tilt_x_deg, tilt_y_deg, t, n_val)
    img = sim2d.composite_parallax(front, back, dx_um, dy_um, grid_pitch, illum=illum)
    _log.info(
        "parallax2d slug=%s variant=%s tilt=(%.1f,%.1f) illum=%s shift_um=(%.2f,%.2f) "
        "grid=%dx%d ds=%d",
        slug,
        root.name,
        tilt_x_deg,
        tilt_y_deg,
        illum,
        dx_um,
        dy_um,
        img.width,
        img.height,
        factor,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------------------------------------------------------------------------
# Tilt-sweep collage — one sheet per pattern, every angle side by side.
# ---------------------------------------------------------------------------

# The sweep composites the full raster once per angle, so the tile count is the
# multiplier on the work. 13 angles at a 512-px working grid is ~0.4 s and a
# few tens of MB; these caps keep a URL from turning into a heavy compute.
MAX_COLLAGE_ANGLES = 25
COLLAGE_MAX_SIDE = 512


def _collage_angles(span_deg: float, steps: int) -> list[float]:
    if not 0.2 <= span_deg <= 45.0:
        raise HTTPException(400, f"span_deg must be 0.2..45, got {span_deg}")
    if not 3 <= steps <= MAX_COLLAGE_ANGLES:
        raise HTTPException(400, f"steps must be 3..{MAX_COLLAGE_ANGLES}, got {steps}")
    half = (steps - 1) / 2.0
    return [round(span_deg * (i - half) / half, 3) for i in range(steps)]


def _collage_for(
    slug: str,
    variant: str,
    *,
    span_deg: float,
    steps: int,
    illum: str,
    axis: str,
    tile_px: int,
    thickness_um: float | None,
    n: float | None,
    cols: int | None = None,
    show_title: bool = True,
):
    from .. import collage as collage_mod
    from .. import sim2d

    if illum not in sim2d.ILLUMINATIONS:
        raise HTTPException(400, f"Unknown illum: {illum}")
    if axis not in ("x", "y", "auto"):
        raise HTTPException(400, f"axis must be 'x', 'y' or 'auto', got {axis!r}")
    if not 48 <= tile_px <= 400:
        raise HTTPException(400, f"tile_px must be 48..400, got {tile_px}")

    angles = _collage_angles(span_deg, steps)
    root, manifest = _load_parallax_pair(slug, variant)
    t, n_val, pitch = _substrate_defaults(manifest, thickness_um, n)
    front, back, grid_pitch, _ = _mask_arrays(
        root, pitch, max_side=COLLAGE_MAX_SIDE
    )

    # 'auto' sweeps BOTH axes and keeps the stronger. This is not a nicety: a
    # layer whose lines run horizontally is invariant under a horizontal shift,
    # so tilting about the wrong axis walks it in the one direction where it has
    # no structure and the pattern reads as perfectly dead. bitmap-halftone does
    # exactly that — 0.000 on x, 0.195 on y — and monogram-carrier-reveal is the
    # mirror image at 0.585 / 0.000. A fixed default is wrong for one of them.
    candidates = ("x", "y") if axis == "auto" else (axis,)
    best = None
    for cand in candidates:
        frames = collage_mod.sweep_frames(
            front, back,
            pixel_pitch_um=grid_pitch, thickness_um=t, n=n_val,
            angles_deg=angles, illum=illum, axis=cand, tile_px=tile_px,
        )
        crop = collage_mod.border_crop_frac(
            front.shape, grid_pitch, t, n_val, angles, cand
        )
        m = collage_mod.tile_metrics(frames, crop_frac=crop)
        if best is None or m["effect_strength"] > best[1]["effect_strength"]:
            best = (frames, m, cand)
    frames, metrics, chosen = best
    metrics = {**metrics, "axis": chosen, "axis_mode": axis}
    sheet = collage_mod.compose_grid(
        frames, cols=cols,
        title=f"{slug}  {illum}  tilt-{chosen}" if show_title else None,
    )
    return sheet, metrics, angles, root, chosen


@router.get("/collage/{slug}/{variant}")
def collage_sheet(
    slug: str,
    variant: str = "default",
    span_deg: float = 6.0,
    steps: int = 13,
    illum: str = "ambient",
    axis: str = "auto",
    tile_px: int = 160,
    cols: int | None = None,
    show_title: bool = True,
    thickness_um: float | None = None,
    n: float | None = None,
) -> StreamingResponse:
    """One PNG sheet: this pattern composited across a fan of view angles.

    This is the honest preview. The WebGL renderer filters each layer
    separately and so computes <front>*<back>, losing the cross term that IS
    the effect once the lattice goes sub-pixel; here the product is formed at
    raster resolution and only then area-averaged into a tile, which is the
    order the eye integrates in. A pattern that does nothing across this sheet
    will do nothing in glass.

    Metrics for the same sweep come back on the ``X-Collage-Metrics`` header so
    a caller showing the image does not have to composite it twice.
    """
    if cols is not None and not 1 <= cols <= MAX_COLLAGE_ANGLES:
        raise HTTPException(400, f"cols must be 1..{MAX_COLLAGE_ANGLES}, got {cols}")
    sheet, metrics, angles, root, chosen = _collage_for(
        slug, variant, span_deg=span_deg, steps=steps, illum=illum, axis=axis,
        tile_px=tile_px, thickness_um=thickness_um, n=n, cols=cols,
        show_title=show_title,
    )
    _log.info(
        "collage slug=%s variant=%s angles=%d illum=%s axis=%s strength=%.3f peak=%s",
        slug, root.name, len(angles), illum, chosen,
        metrics["effect_strength"], metrics.get("peak_pair_deg"),
    )
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"X-Collage-Metrics": json.dumps(metrics)},
    )


@router.get("/collage/{slug}/{variant}/metrics")
def collage_metrics(
    slug: str,
    variant: str = "default",
    span_deg: float = 6.0,
    steps: int = 13,
    illum: str = "ambient",
    axis: str = "auto",
    thickness_um: float | None = None,
    n: float | None = None,
) -> dict:
    """The same sweep's numbers without the image."""
    _, metrics, angles, root, _chosen = _collage_for(
        slug, variant, span_deg=span_deg, steps=steps, illum=illum, axis=axis,
        tile_px=96, thickness_um=thickness_um, n=n,
    )
    return {
        "slug": slug,
        "variant": root.name,
        "angles_deg": angles,
        "illum": illum,
        **metrics,
    }


@router.get("/parallax2d/{slug}/{variant}/curve")
def parallax2d_curve(
    slug: str,
    variant: str,
    axis: str = "x",
    thickness_um: float | None = None,
    n: float | None = None,
    points: int = 21,
) -> dict:
    """Mean transmission vs tilt, up to 41 points.

    Every point is a full-frame composite, so the masks are area-averaged to
    SIM2D_MAX_SWEEP_SIDE per side first and the sweep runs on that grid at
    pitch * factor; ``pixel_pitch_um`` still reports the raster's own pitch,
    with the swept grid in ``sim_pixel_pitch_um`` / ``sim_grid``.
    """
    from .. import sim2d

    if axis not in ("x", "y"):
        raise HTTPException(400, f"Unknown axis: {axis}")
    root, manifest = _load_parallax_pair(slug, variant)
    t, n_val, pitch = _substrate_defaults(manifest, thickness_um, n)
    front, back, grid_pitch, factor = _mask_arrays(
        root, pitch, max_side=SIM2D_MAX_SWEEP_SIDE
    )

    points = max(2, min(int(points), 41))
    tilts = [-30.0 + 60.0 * i / (points - 1) for i in range(points)]
    curve = sim2d.contrast_curve(front, back, tilts, t, n_val, grid_pitch, axis=axis)
    _log.info(
        "parallax2d curve slug=%s variant=%s axis=%s points=%d grid=%dx%d ds=%d",
        slug, root.name, axis, points, front.shape[1], front.shape[0], factor,
    )
    return {
        "slug": slug,
        "variant": root.name,
        "axis": axis,
        "thickness_um": t,
        "n": n_val,
        "pixel_pitch_um": pitch,
        "sim_pixel_pitch_um": grid_pitch,
        "sim_grid": [int(front.shape[1]), int(front.shape[0])],
        "downsample": factor,
        "curve": curve,
    }
