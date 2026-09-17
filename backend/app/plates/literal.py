"""LITERAL rasters: the fabricated chrome, sampled as coverage.

What the renderer binds for the honest optical path — not a stylised preview
but the real written geometry, area-averaged into a texture, plus the period
map that tells the shader which grating sits under each texel.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from PIL import Image

from .. import region_art as _RA
from ..service import save_png_atomic
from .photo import photo_colour_band_periods
from .recipe import CENTERPIECE_FILL, single_ply_leaf_period_um
from .spec import FaceKind, PlateSpec, _aperture

_log = logging.getLogger("optics.plates")

# --- LITERAL rasters: the fabricated chrome, sampled as coverage -------------
#
# ``front.png`` / ``back.png`` are SHADER masks: graylevel ZONE CODES
# (FRAME_LEVEL, ART_LEVEL, RAINBOW_LEVEL, the FRAME_BUCKET angle ladder) that
# tell plate.frag which procedural effect to run where, at a pitch far too
# coarse to carry a grating without aliasing rings. The literal rasters are the
# opposite kind of image: no codes and no zones, just the actual DRC-healed
# litho polygons ``export_fine.build_plate_fine`` hands the GDS writer, sampled
# as area coverage. 255 = chrome/metal present, 0 = bare glass.
#
# The renderer samples these on the outer and inner pattern planes instead of
# synthesising gratings, which is what makes the preview a picture of the plate
# rather than an impression of it — and it is the same geometry the mask writer
# prints, so a moiré on screen is a moiré the fab will make.
#
# Same extent, orientation and centre as front.png/back.png (whole plate
# INCLUDING the weld margin, unmirrored, plate µm with the origin at the plate
# centre and y UP mapped to row 0 = top) so the two overlay pixel for pixel
# after a uniform scale. An EMPTY layer is a valid literal raster and is
# written as all zeros — a blank face, or the back of a single-ply one — never
# left as a missing file.
LITERAL_RASTER_PX = 2048
# ``period_front.png`` packs the sub-grating period as period_um × 25, so the
# whole diffractive family fits an 8-bit channel (0-10.2 µm at 0.04 µm steps)
# and 0 keeps its meaning: no sub-grating here.
LITERAL_PERIOD_SCALE = 25.0


def _literal_raster_dims(spec: PlateSpec) -> tuple[int, int]:
    """``(w_px, h_px)`` for the literal rasters: plate aspect, 2048 on the long side."""
    w_um = max(1e-6, float(spec.width_um))
    h_um = max(1e-6, float(spec.height_um))
    if w_um >= h_um:
        return LITERAL_RASTER_PX, max(1, int(round(LITERAL_RASTER_PX * h_um / w_um)))
    return max(1, int(round(LITERAL_RASTER_PX * w_um / h_um))), LITERAL_RASTER_PX


def _literal_layer_raster(
    polys: list["np.ndarray"], spec: PlateSpec, w_px: int, h_px: int
) -> Image.Image:
    """One layer's plate-frame polygon rings → an ``L`` coverage raster.

    ``polys`` are ``build_plate_fine``'s healed EXTERIOR rings (holes already
    dropped by ``drc_clean_region`` — a fill-only gold mask), so filling every
    ring is the whole rasterization. Plate µm (origin centre, y up) map to
    pixels exactly as ``_raster_compose_plate`` maps them: x + W/2 scaled
    across the width, H/2 - y scaled down the height.

    The coverage is computed ANALYTICALLY (``literal_raster.layer_coverage``),
    not by supersampling a fill. Every feature here is far below a texel — a
    27.5 mm face is ~13.4 µm/texel while the colour sub-gratings are 2.5 µm
    lines — so a boundary-inclusive fill biases the whole raster bright
    (measured +6% on a 50%-duty carrier, and the 5 µm colour bands read 0.71
    with 46% of their texels pinned at 255). See that module for why no
    supersample/shrink combination fixes it and how the exact accumulation
    works. 255 = chrome over the whole texel, 0 = bare glass.
    """
    from ..literal_raster import layer_coverage

    import numpy as np

    cov = layer_coverage(polys or [], spec.width_um, spec.height_um, w_px, h_px)
    return Image.fromarray(np.round(cov * 255.0).astype(np.uint8), mode="L")


def _literal_period_raster(spec: PlateSpec, w_px: int, h_px: int) -> Image.Image | None:
    """``period_front.png``: the sub-grating PERIOD field, or None if there is none.

    Only a photo-halftone face with a colour plan has one. Its diffraction
    sub-grating is ~5 µm lines on a few-µm period — three orders of magnitude
    below a 2048 px plate raster — so ``literal_front`` can only carry the
    stripes' COVERAGE (a grey inside each coloured band). The colour they
    diffract comes from their PERIOD, which is what this map publishes:
    ``period_um × LITERAL_PERIOD_SCALE`` inside every band that carries one,
    0 everywhere else.

    Bands are painted at final resolution and outset to whole pixels: a band is
    thinner than a pixel at this scale, and a label field that rounds its thin
    bands away publishes "no sub-grating here" — a worse lie than a band one
    pixel too wide.

    A texel therefore lands inside several stacked bands at once, so the one
    that WINS it is the one whose exact overlap area is largest — not whichever
    was painted last. Bands with no sub-grating (period 0) never enter the
    contest, so the map stays 0 wherever there is nothing to diffract.
    """
    import numpy as np

    from ..literal_raster import rect_texel_overlaps

    out = np.zeros(h_px * w_px, dtype=np.uint8)
    if spec.kind is FaceKind.REGION:
        # SINGLE-LAYER DIFFRACTION centrepiece: the region map's periods, at
        # the art box, nearest-neighbour onto the texel grid.
        side_um = CENTERPIECE_FILL * _aperture(spec)
        sx_, sy_ = w_px / max(1e-6, spec.width_um), h_px / max(1e-6, spec.height_um)
        px0 = int(round((spec.width_um / 2 - side_um / 2) * sx_))
        py0 = int(round((spec.height_um / 2 - side_um / 2) * sy_))
        side_px = max(8, int(round(side_um * sx_)))
        ra = _RA.centerpiece_regions(spec.pattern_slug, side_px, spec.pattern_params)
        if ra is not None:
            per = ra.period_lut()[ra.labels]
            vals = np.minimum(255, np.round(per * LITERAL_PERIOD_SCALE)).astype(np.uint8)
            grid = out.reshape(h_px, w_px)
            xa, ya = max(0, px0), max(0, py0)
            xb, yb = min(w_px, px0 + side_px), min(h_px, py0 + side_px)
            if xb > xa and yb > ya:
                tile = vals[ya - py0 : yb - py0, xa - px0 : xb - px0]
                sub_ = grid[ya:yb, xa:xb]
                sub_[tile > 0] = tile[tile > 0]
    rects, periods = photo_colour_band_periods(spec)
    if rects.shape[0] == 0:
        if out.any():
            return Image.fromarray(out.reshape(h_px, w_px), mode="L")
        return None
    values = np.minimum(
        255, np.round(np.asarray(periods, dtype=float) * LITERAL_PERIOD_SCALE)
    ).astype(np.int64)
    has = values > 0
    if not has.any():
        if out.any():
            return Image.fromarray(out.reshape(h_px, w_px), mode="L")
        return None
    rects, values = rects[has], values[has]

    w_um = max(1e-6, float(spec.width_um))
    h_um = max(1e-6, float(spec.height_um))
    sx, sy = w_px / w_um, h_px / h_um
    hx, hy = 0.5 * w_um, 0.5 * h_um
    flat, band, area = rect_texel_overlaps(
        (rects[:, 0] + hx) * sx,
        (rects[:, 1] + hx) * sx,
        (hy - rects[:, 3]) * sy,
        (hy - rects[:, 2]) * sy,
        w_px,
        h_px,
    )
    if flat.size == 0:
        return Image.fromarray(out.reshape(h_px, w_px), mode="L") if out.any() else None
    # Largest overlap wins each texel: sort by (texel, area) and keep the last
    # entry of every texel's run.
    order = np.lexsort((area, flat))
    flat, band = flat[order], band[order]
    last = np.nonzero(np.diff(flat, append=flat[-1] + 1))[0]
    out[flat[last]] = values[band[last]].astype(np.uint8)
    if not out.any():
        return None
    return Image.fromarray(out.reshape(h_px, w_px), mode="L")


def _single_ply_leaf_period_raster(
    spec: PlateSpec, front: Image.Image, period: Image.Image | None
) -> Image.Image | None:
    """Add the leaf grating's period to the period map of a single-ply face.

    On one ply the garland is fine diffractive gratings
    (``single_ply_leaf_period_um`` — the same value the manifest advertises)
    that a 2048 px raster cannot resolve — ``literal_front`` carries their 50%
    coverage. Every front texel OUTSIDE the centerpiece art box that carries
    metal is a leaf (nothing else is written there on a single ply), so it gets
    the leaf period; texels the photo bands already claimed keep theirs. The
    per-leaf ORIENTATION is not in this map — the preview's sheen is the same
    for every leaf, which is the one approximation the flag in the shader names.
    """
    import numpy as np

    cov = np.asarray(front.convert("L"), dtype=np.uint8)
    h_px, w_px = cov.shape
    out = np.zeros((h_px, w_px), dtype=np.uint8) if period is None else np.asarray(period.convert("L"), dtype=np.uint8).copy()
    side = CENTERPIECE_FILL * _aperture(spec)
    sx, sy = w_px / max(1e-6, spec.width_um), h_px / max(1e-6, spec.height_um)
    x0 = int(np.floor((spec.width_um / 2 - side / 2) * sx)); x1 = int(np.ceil((spec.width_um / 2 + side / 2) * sx))
    y0 = int(np.floor((spec.height_um / 2 - side / 2) * sy)); y1 = int(np.ceil((spec.height_um / 2 + side / 2) * sy))
    outside = np.ones((h_px, w_px), dtype=bool)
    outside[max(0, y0):min(h_px, y1), max(0, x0):min(w_px, x1)] = False
    leaf = outside & (cov > 5) & (out == 0)
    value = int(min(255, round(single_ply_leaf_period_um(spec) * LITERAL_PERIOD_SCALE)))
    out[leaf] = value
    if not out.any():
        return None
    return Image.fromarray(out, mode="L")


def _write_literal_rasters(spec: PlateSpec, out_dir: Path, pid: str) -> dict[str, str]:
    """Publish the plate's literal rasters; return the ``files`` entries for them.

    ``literal_front`` / ``literal_back`` always, ``period_front`` only when the
    face actually has a sub-grating period field (an all-zero period map is
    noise — the absent key IS "no sub-gratings on this face"). Any period map
    left by an earlier compose of this slot is removed, so the manifest and the
    directory cannot disagree.

    ONE ``build_plate_fine`` per face: it is the expensive call on this path
    (seconds to ~40 s), and it already reuses this plate's frame-scene sidecar
    rather than regrowing the band — see ``frame_scene_for_plate`` and the
    fresh-compose window ``_materialize_plate_locked`` opens around it.
    """
    from ..export_fine import build_plate_fine

    w_px, h_px = _literal_raster_dims(spec)
    # ``face`` is informational on this path (it labels PlateFine.face for the
    # wafer report's per-face stats and picks nothing geometric); a plate
    # materialized on its own has no box face id, so it is labelled by what it
    # actually is.
    t_fine = time.perf_counter()
    fine = build_plate_fine(spec, spec.label or spec.pattern_slug)
    fine_ms = int((time.perf_counter() - t_fine) * 1000)

    t_raster = time.perf_counter()
    files: dict[str, str] = {}
    front_img: Image.Image | None = None
    for name, layer in (("literal_front", fine.front_polys), ("literal_back", fine.back_polys)):
        img = _literal_layer_raster(layer, spec, w_px, h_px)
        if name == "literal_front":
            front_img = img
        save_png_atomic(img, out_dir / f"{name}.png")
        files[name] = f"/data/plates/{pid}/{name}.png"

    period = _literal_period_raster(spec, w_px, h_px)
    if getattr(spec, "single_ply", False) and front_img is not None:
        period = _single_ply_leaf_period_raster(spec, front_img, period)
    period_path = out_dir / "period_front.png"
    if period is None:
        period_path.unlink(missing_ok=True)
    else:
        save_png_atomic(period, period_path)
        files["period_front"] = f"/data/plates/{pid}/period_front.png"
    _log.info(
        "literal_rasters id=%s slug=%s %dx%d fine=%dms raster=%dms polys=%d/%d period=%s",
        pid,
        spec.pattern_slug,
        w_px,
        h_px,
        fine_ms,
        int((time.perf_counter() - t_raster) * 1000),
        len(fine.front_polys),
        len(fine.back_polys),
        period is not None,
    )
    return files
