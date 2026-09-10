"""Fine-pitch wafer GDS writer — TRUE optical periods, per-face dispatch, BSA marks.

The coarse SVG/GDS path (``plates.ensure_plate_svg`` + ``export_wafer``) rasters
every layer at a *budget pitch* (72-82 µm on a mini plate) and, when the ideal
4-samples/period pitch would blow the 400k-cell cap, RESCALES the grating periods
13-15× to fit (see F2 in the audit). That kills the physics: the centerpiece
switch period jumps 60 µm → ~800 µm (a dead ~65° switch), the leaf carrier and
rainbow accent alias into noise.

This module is the fab-grade alternative. It writes the wafer's gold layers as
TRUE geometry at the design periods —

    * back leaf carrier      22.000 µm      (BACK_CARRIER_PERIOD_UM)
    * front leaf grating      23.980 µm     (22 × 1.09) at +3.0° offset
    * centerpiece switch      60.000 µm     front/back EXACTLY 30 µm (½-period) out of phase
    * scanimation slit         60.000 µm    open slot 15 µm / bar 45 µm, N=4 interleave
    * rainbow accent           4.400 µm     (2.2 µm line + 2.2 µm gap) at 45°

— by generating each grating as RECTANGLES in its own grating-local frame
(where lines are axis-aligned and one line is ONE full-span rectangle, so the
polygon count scales with LINE count, not pixel count), clipping each line to
its zone by a numpy row-span pass against a freshly-regenerated SOURCE mask, and
rotating the whole set to placement as polygons only when the grating is angled.

Only the zone *boundaries* (which pixels are frame / centerpiece / water / accent)
are quantized to the source-mask raster (~20-30 µm silhouette-edge steps, which
are unavoidable and visually irrelevant); the PERIODS inside every zone are exact
vector geometry.

Per-face dispatch reads the real six-face plan from ``boxes.default_box_spec``
(front colibrí-globe, back capybara-scanimation, left food-pair, right
gear-quill, top monogram-jp, bottom inscription) — one source of truth, no
duplicated slug table. The plate footprints come from the same packer
``export_wafer.solve_max_scale`` uses, minus keep-out around the two BSA marks.

BSA fiducials (F4, user constraint, EXACT): one pair on the wafer horizontal
centerline, centers at (-30.000, 0.0) mm and (+30.000, 0.0) mm — 60.000 mm
apart. Front layer (10,0) gets a SOLID cross in a clear field; back layer (20,0)
gets the COMPLEMENTARY window target (open cross slot in a solid pad) so the
aligner overlays cross-in-slot. A dedicated fiducial datalayer (60,0)/(61,0)
carries copies so the marks are findable independent of pattern gold, and a
small vernier pair sits beside each mark on both layers.

CLI (writes the WHOLE wafer — do NOT run casually; one compute process at a
time on the 13.7 GB host):

    uv run python -m app.export_fine --out data/wafer/wafer_fine.gds

Validate cheaply with the built-in test cell (one grating of each type in a
5×5 mm cell, written + read back + measured):

    uv run python -m app.export_fine --test-cell --out data/wafer/testcell.gds

All lengths µm unless a ``_mm`` suffix says otherwise.
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .patterns.effects.gratings import band_select

_log = logging.getLogger("optics.export_fine")

# --- GDS layer map (layer, datatype) ---------------------------------------
# Pattern gold matches export_wafer's map so the two exports overlay.
LAYER_FRONT = (10, 0)      # front-face gold (viewer side)
LAYER_BACK = (20, 0)       # back-face gold (far side)
LAYER_OUTLINE = (1, 0)     # plate outlines + dicing streets
LAYER_WAFER = (99, 0)      # wafer usable-region outline
LAYER_LABEL = (3, 0)       # text
# Dedicated fiducial datalayers — a copy of each BSA mark lives here so the
# aligner/QA can find the marks independent of pattern gold.
LAYER_FIDUCIAL_FRONT = (60, 0)
LAYER_FIDUCIAL_BACK = (61, 0)

# --- BSA fiducial geometry (F4 — EXACT, do not tune) ------------------------
# User constraint: one pair on the wafer horizontal centerline, EXACTLY 60 mm
# apart → centers at (∓30.000 mm, 0.0).
FIDUCIAL_CENTERS_UM: tuple[tuple[float, float], tuple[float, float]] = (
    (-30_000.0, 0.0),
    (+30_000.0, 0.0),
)
FIDUCIAL_KEEPOUT_R_UM = 2_500.0     # r = 2.5 mm packer keep-out disc per mark
# Front mark: solid cross, 400 µm arm length (half-arm 200 µm from center),
# 20 µm arm width, inside a clear field.
FID_CROSS_ARM_UM = 400.0
FID_CROSS_WIDTH_UM = 20.0
# Back mark: complementary WINDOW target — a 500×500 µm solid pad with an open
# cross slot 30 µm wide cut through it (the front solid cross seats in the slot).
FID_PAD_UM = 500.0
FID_SLOT_WIDTH_UM = 30.0
# Vernier pair beside each mark (rotation / registration readout): two short
# rulings whose pitch differs by the vernier step so a sub-pitch offset shows up
# as a coincidence line shift.
FID_VERNIER_PITCH_A_UM = 20.0
FID_VERNIER_PITCH_B_UM = 25.0       # 5 µm pitch difference (the scale)
FID_VERNIER_N = 10                  # rulings per comb
FID_VERNIER_LEN_UM = 120.0
FID_VERNIER_LINE_UM = 4.0
FID_VERNIER_OFFSET_UM = 400.0       # comb center offset from the fiducial center

# --- litho floor ------------------------------------------------------------
LITHO_FLOOR_UM = 2.0   # 2 µm line / 2 µm gap, HARD

# GDS database unit: 1 nm. Coordinates are written as round(um / DBU_UM).
DBU_UM = 0.001


# =========================================================================== #
#  DRC bridge (hard dependency — wired in at Integrate)                         #
# =========================================================================== #
# The DRC module lives at backend/app/patterns/effects/drc.py with the fixed
# interface (drc_clean_rects / drc_clean_polys / drc_report). It is now a HARD
# dependency of the fine writer: the whole point of this export is fab-grade
# geometry, and shipping a wafer whose sub-floor slivers were silently passed
# through would defeat it. If the import fails we fail loud rather than write an
# un-DRC'd layout.
from .patterns.effects.drc import (  # noqa: E402
    drc_clean_polys,
    drc_clean_rects,
    drc_clean_region,
    drc_report,
    drc_report_region,
)

_DRC_AVAILABLE = True


# =========================================================================== #
#  geometry primitives — native-pitch grating rectangles                       #
# =========================================================================== #

def _line_rects_local(
    zone_bbox_um: tuple[float, float, float, float],
    period_um: float,
    duty: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Vertical line grating over a bbox, one FULL-SPAN rectangle per gold line.

    Returns an ``(N, 4)`` ``[x0, x1, y0, y1]`` array (µm) in the SAME frame as
    ``zone_bbox_um`` (``x0,y0,x1,y1``). Lines are constant-x gold stripes of
    width ``duty·period``, spaced ``period``, phase-shifted by ``phase`` periods.
    Each line spans the full bbox height as ONE rectangle — the count is the
    number of lines across the width, NOT the pixel area.

    Phase sign is the ONE shared convention: gold line ``k``'s left edge is at
    ``(k − phase)·period``, matching ``plates._grating_grid`` (gold where
    ``fract(x/p + phase) < duty``) and :func:`_angled_grating_local_rects`. It
    must stay that way — the interlace front comb runs at ``phase = −0.25``, so
    the opposite sign would shift the fabricated barrier bar by p/2 onto the
    designed open slot and invert the tilt→image mapping. Pinned by
    ``tests/test_grating_phase.py``.
    """
    x0b, y0b, x1b, y1b = zone_bbox_um
    line_w = duty * period_um
    # Enumerate one period either side of the bbox so no partially-overlapping
    # edge line is dropped for any phase/duty; the x-clip below drops the misses.
    k0 = math.floor((x0b / period_um) + phase) - 1
    k1 = math.ceil((x1b / period_um) + phase) + 1
    rects = []
    for k in range(k0, k1):
        lx0 = (k - phase) * period_um
        lx1 = lx0 + line_w
        # Clip to bbox in x.
        cx0 = max(lx0, x0b)
        cx1 = min(lx1, x1b)
        if cx1 - cx0 <= 1e-9:
            continue
        rects.append((cx0, cx1, y0b, y1b))
    if not rects:
        return np.empty((0, 4), dtype=float)
    return np.asarray(rects, dtype=float)


def _clip_rects_to_row_spans(
    line_rects: np.ndarray,
    zone_mask: np.ndarray,
    pitch_um: float,
    extent_um: tuple[float, float],
) -> np.ndarray:
    """Clip vertical-line rectangles to a zone mask by numpy ROW-SPANS.

    ``zone_mask`` is a ``(h_px, w_px)`` bool grid (True = inside the zone) at
    ``pitch_um`` per cell, centered at the grid origin, y up (matching
    ``gratings._linear_frac``). For each gold line (a vertical rect) we walk the
    mask rows the line covers and emit one rectangle per contiguous run of True
    rows within the line's x-column — MERGING colinear vertical runs so an
    unbroken line stays ONE rectangle. This is the "clip by row-span against the
    SOURCE mask" pass (F2): periods stay exact; only the zone BOUNDARY is
    quantized to ``pitch_um``.

    Returns ``(M, 4)`` ``[x0, x1, y0, y1]`` µm rectangles (y up).
    """
    h_px, w_px = zone_mask.shape
    hx = w_px * pitch_um / 2.0
    hy = h_px * pitch_um / 2.0
    if line_rects.shape[0] == 0:
        return np.empty((0, 4), dtype=float)
    n_lines = line_rects.shape[0]
    # Per-line column band in mask pixels.
    c0 = np.floor((line_rects[:, 0] + hx) / pitch_um).astype(int)
    c1 = np.ceil((line_rects[:, 1] + hx) / pitch_um).astype(int)
    c0 = np.clip(c0, 0, w_px)
    c1 = np.clip(c1, 0, w_px)
    # Build a (n_lines × h_px) "line covered in this row" bool by OR-ing each
    # line's column band. Loop is over LINES (few) but each op is vectorised; the
    # per-line run extraction below is done ONCE on the whole stacked array.
    cover = np.zeros((n_lines, h_px), dtype=bool)
    for i in range(n_lines):
        if c1[i] > c0[i]:
            cover[i] = zone_mask[:, c0[i] : c1[i]].any(axis=1)
    # Vectorised run extraction over the stacked array: pad each row, diff along
    # the h_px axis → run starts/ends per line, all at once.
    padded = np.zeros((n_lines, h_px + 2), dtype=np.int8)
    padded[:, 1:-1] = cover
    d = np.diff(padded, axis=1)
    li_s, r_starts = np.nonzero(d == 1)   # li_s = line index, r_starts = top row
    _, r_ends = np.nonzero(d == -1)       # exclusive bottom row
    if li_s.size == 0:
        return np.empty((0, 4), dtype=float)
    x0 = line_rects[li_s, 0]
    x1 = line_rects[li_s, 1]
    yy1 = hy - r_starts * pitch_um        # y up: top row → higher y
    yy0 = hy - r_ends * pitch_um
    return np.stack([x0, x1, yy0, yy1], axis=1).astype(float)


# =========================================================================== #
#  zone masks — regenerated at an adequate n_grid (boundary only)              #
# =========================================================================== #

@dataclass
class ZoneMasks:
    """Per-plate zone bool grids at ``pitch_um`` (origin center, y up).

    Every grid is ``(h_px, w_px)``. ``pitch_um`` sets the zone-BOUNDARY
    quantization only; the periods inside are exact vector geometry.
    """

    pitch_um: float
    extent_um: tuple[float, float]
    frame: np.ndarray                      # perimeter foliage band (front)
    frame_level: np.ndarray                # per-cell graylevel (angle bucket)
    back_window: np.ndarray                # whole exposed-face carrier window (back)
    front_art: np.ndarray                  # centerpiece front silhouette
    back_art: np.ndarray                   # centerpiece back silhouette
    front_accent: np.ndarray               # rainbow accent subset of front_art
    # Full centerpiece art box (the CENTERPIECE_FILL square, weld-rim clipped).
    # The barrier-interlace front comb spans this WHOLE square — never the
    # silhouette union (a union-gated comb is itself a static front image).
    art_box: np.ndarray | None = None
    # Scanimation (capybara back only); None otherwise. The barrier + back frames
    # are EMITTED as exact vector rects (see the scanimation vector builders) —
    # only ``capy_body`` (the dry silhouette that carries the shimmer grating) and
    # ``water_band`` (used to exclude the carrier from the animated region) are
    # rastered here, and both are boundary-only zones like the other masks.
    # ``water_band`` is the CARVED band — below the waterline MINUS the submerged
    # body (``capybara_scanimation._capybara_and_water``, the same algebra the
    # composed preview uses) — so the plain back carrier is KEPT over the animal
    # rather than cleared for a ripple field that must not print there.
    water_band: np.ndarray | None = None
    capy_body: np.ndarray | None = None      # dry capybara silhouette (above water)
    # FULL capybara silhouette in the SQUARE art-box grid (``side_px``², y-DOWN,
    # normalized 0..1 coords) — NOT placed on the plate grid like the fields above.
    # The exact vector builders work in art-box normalized coords, so they carve the
    # animal out of the barrier comb / ripple slots by sampling THIS mask (see
    # ``_sample_art_mask``); handing them the plate-grid placement would mean
    # re-deriving the normalization they already have. Same silhouette raster the
    # ``water_band`` / ``capy_body`` zones above were cut from, so the rastered
    # zones and the vector geometry cannot disagree about where the animal is.
    #
    # The FULL silhouette, not the submerged part, for the same reason
    # ``_capybara_and_water`` carves the band with ``~capy``: the builders only ever
    # emit INSIDE the water band, so above the waterline the mask is never consulted
    # — while a submerged-only mask would lose the body in the one raster row that
    # STRADDLES the waterline (the band's analytic edge sits partway into that row,
    # so ``capy & below`` calls it dry and a bar/crest would print on the animal's
    # back there).
    capy_art: np.ndarray | None = None
    # EFFECTIVE waterline (art-box normalized y, 0 = top) this plate's zones were
    # built at — resolved ONCE from the face's ``waterline`` pattern param by
    # ``plates._water_waterline_y``, the same resolver the composed preview mask,
    # the fab SVG bake and the shader's ``water_waterline_y`` read. Carried here
    # (rather than re-read from the module constant downstream) so the rastered
    # body/band zones and the EXACT vector barrier + back-frame builders in
    # ``build_plate_fine`` cannot be built at two different waterlines. None on a
    # non-scanimation plate.
    waterline_y: float | None = None


def _build_zone_masks(spec: Any, pitch_um: float) -> ZoneMasks:
    """Regenerate a plate's zone masks at ``pitch_um`` from the SOURCE motifs.

    Reuses plates.py's own helpers (`frame_scene_for_plate` →
    `render_scene_to_image` for the foliage band graylevels — the composed
    plate's own scene, not a regrown one — `_centerpiece_masks` for the silhouettes,
    `_front_accent_zone` for the rainbow patch, the capybara `_build` for the
    scanimation). Boundary quantization only — periods are added later as vector
    geometry.
    """
    from PIL import Image

    from . import plates as P

    W, H = spec.width_um, spec.height_um
    fw = max(1, int(round(W / pitch_um)))
    fh = max(1, int(round(H / pitch_um)))

    frame = np.zeros((fh, fw), dtype=bool)
    frame_level = np.zeros((fh, fw), dtype=np.uint8)
    back_window = np.zeros((fh, fw), dtype=bool)
    front_art = np.zeros((fh, fw), dtype=bool)
    back_art = np.zeros((fh, fw), dtype=bool)
    front_accent = np.zeros((fh, fw), dtype=bool)
    art_box = np.zeros((fh, fw), dtype=bool)

    if spec.pattern_slug == P.BLANK_SLUG:
        # BARE GLASS: every zone stays empty, so every ``.any()`` gate in
        # ``build_plate_fine`` falls through and the plate emits no geometry at
        # all. Returning here rather than special-casing each emitter means any
        # OTHER caller of the zone masks (the single-ply carrier block below,
        # the render tools) also sees a blank face as blank.
        return ZoneMasks(
            pitch_um=pitch_um,
            extent_um=(W, H),
            frame=frame,
            frame_level=frame_level,
            back_window=back_window,
            front_art=front_art,
            back_art=back_art,
            front_accent=front_accent,
            art_box=art_box,
        )

    # --- FRONT perimeter foliage band (graylevel = angle bucket) -----------
    active_w, active_h = spec.active_dims()
    if active_w > 0 and active_h > 0:
        from .patterns.frames import (
            RectFrame,
            render_scene_to_image,
        )

        rect = RectFrame(width_um=active_w, height_um=active_h)
        fp = spec.frame.to_frame_params()
        fp.fill_interior = False
        # Reuse the composed plate's scene.json sidecar — the fine bake must
        # carry the SAME foliage band the preview PNG and the fab SVG do, and
        # regrowing it here cost 1-3 s per face on every export.
        pid = P.plate_hash(spec)
        scene = P.frame_scene_for_plate(
            P.PLATES_ROOT / pid, P.get_plate(pid), rect, fp
        )
        sil_img = render_scene_to_image(
            scene, rect, fp, pitch_um, level_fn=P._frame_level_for
        )
        aw = min(fw, sil_img.size[0])
        ah = min(fh, sil_img.size[1])
        ox = (fw - aw) // 2
        oy = (fh - ah) // 2
        frame_level[oy : oy + ah, ox : ox + aw] = np.asarray(sil_img)[:ah, :aw]
        frame = frame_level > 0
        _mask_rim(frame, spec.weld_margin_um, pitch_um)
        frame_level[~frame] = 0

    # --- BACK carrier window (whole exposed face) --------------------------
    back_w, back_h = spec.back_dims()
    if back_w > 0 and back_h > 0:
        bw = max(1, int(round(back_w / pitch_um)))
        bh = max(1, int(round(back_h / pitch_um)))
        bx = (fw - bw) // 2
        by = (fh - bh) // 2
        back_window[by : by + bh, bx : bx + bw] = True
        back_margin = spec.weld_margin_um if spec.back_margin_um is None else spec.back_margin_um
        _mask_rim(back_window, back_margin, pitch_um)

    # --- CENTERPIECE silhouettes + accent ----------------------------------
    aperture = P._aperture(spec)
    side_px = max(8, int(round(P.CENTERPIECE_FILL * aperture / pitch_um))) if aperture > 0 else 0
    if side_px > 0:
        # Full art-box square (the CENTERPIECE_FILL footprint) — the zone the
        # barrier-interlace front comb spans, independent of the silhouettes.
        ax0 = max(0, fw // 2 - side_px // 2)
        ay0 = max(0, fh // 2 - side_px // 2)
        art_box[ay0 : min(fh, ay0 + side_px), ax0 : min(fw, ax0 + side_px)] = True
        _mask_rim(art_box, spec.weld_margin_um, pitch_um)
        masks = P._centerpiece_masks(spec.pattern_slug, side_px, spec.pattern_params)
        if masks is not None:
            def _place(art: np.ndarray) -> np.ndarray:
                if art.shape[0] != side_px or art.shape[1] != side_px:
                    im = Image.fromarray((art.astype(np.uint8) * 255), "L").resize(
                        (side_px, side_px), Image.NEAREST
                    )
                    art = np.asarray(im) > 127
                m = np.zeros((fh, fw), dtype=bool)
                x0 = fw // 2 - side_px // 2
                y0 = fh // 2 - side_px // 2
                xa, ya = max(0, x0), max(0, y0)
                xb, yb = min(fw, x0 + side_px), min(fh, y0 + side_px)
                m[ya:yb, xa:xb] = art[ya - y0 : yb - y0, xa - x0 : xb - x0]
                return m, art

            front_art, front_side = _place(masks[0])
            back_art, _ = _place(masks[1])
            _mask_rim(front_art, spec.weld_margin_um, pitch_um)
            accent_zone = P._front_accent_zone(spec.pattern_slug, front_side)
            if accent_zone is not None:
                front_accent, _ = _place(accent_zone)
                _mask_rim(front_accent, spec.weld_margin_um, pitch_um)

    zm = ZoneMasks(
        pitch_um=pitch_um,
        extent_um=(W, H),
        frame=frame,
        frame_level=frame_level,
        back_window=back_window,
        front_art=front_art,
        back_art=back_art,
        front_accent=front_accent,
        art_box=art_box,
    )

    # --- SCANIMATION (capybara back face) ----------------------------------
    # Only the SILHOUETTE zones are rastered here (boundary quantization is fine
    # for them): the dry capybara body that carries the shimmer grating, and the
    # water-band rectangle. The 60/15/45 barrier and the N=4 interleaved back
    # frames are NOT rastered — they are emitted as exact vector geometry in
    # ``build_plate_fine`` (blockers 1+2), so the coarse pitch never touches the
    # 15 µm slot structure.
    if spec.pattern_slug == P.WATER_SCAN_SLUG and side_px > 0:
        from .patterns.artistic import capybara_scanimation as capyscan

        # Honour the face's tunable ``waterline`` param instead of the module
        # constant: ONE resolver (plates._water_waterline_y) feeds the composed
        # preview mask, the fab SVG bake, the recipe_data the shader binds and —
        # via zm.waterline_y below — this fine-GDS bake, so a face that moves its
        # waterline gets a band/wake that matches its preview and its SVG. The
        # silhouettes above already honour it (``_centerpiece_masks`` takes
        # ``spec.pattern_params``), so hardwiring it here also split THIS module
        # against itself: the dry-body zone moved while the water band did not.
        waterline_y = P._water_waterline_y(spec.pattern_params)
        scene = capyscan._capybara_and_water(side_px, waterline_y)

        def _place_side(side_mask: np.ndarray, w_px: int | None = None) -> np.ndarray:
            w_px = side_px if w_px is None else w_px
            m = np.zeros((fh, fw), dtype=bool)
            x0 = fw // 2 - w_px // 2
            y0 = fh // 2 - side_px // 2
            xa, ya = max(0, x0), max(0, y0)
            xb, yb = min(fw, x0 + w_px), min(fh, y0 + side_px)
            m[ya:yb, xa:xb] = side_mask[ya - y0 : yb - y0, xa - x0 : xb - x0]
            return m

        # WATER FULL WIDTH: the water band spans the full WIDTH-axis aperture
        # (`_aperture_width_um`, edge-to-edge of the window — NOT the square
        # `_aperture` that sizes the body); its VERTICAL extent stays the body
        # square's so the waterline lines up with the half-submerged capybara.
        # The submerged body is carved only from the central square columns — the
        # wings are open water, and ``scene["water_band"]`` is already
        # ``below & ~capy``, the same carve `plates._paste_centerpiece` pastes into
        # the preview's full-width band.
        ap_w = max(side_px, int(round(P._aperture_width_um(spec) / pitch_um)))
        rows = np.arange(side_px)[:, None] / side_px  # 0..1 y-down over the square
        below = np.broadcast_to(rows >= waterline_y, (side_px, ap_w))
        water_full = np.array(below, dtype=bool)
        col_off = (ap_w - side_px) // 2
        water_full[:, col_off : col_off + side_px] = scene["water_band"]

        zm.water_band = _place_side(water_full, w_px=ap_w)
        zm.capy_body = _place_side(scene["capy_above"])
        # Art-box-grid silhouette for the EXACT vector builders (they carve the
        # barrier comb + ripple slots against it — see `_sample_art_mask`). Kept in
        # the square grid, unplaced, because those builders address the art box in
        # normalized coords; the FULL silhouette rather than ``capy_below`` for the
        # waterline-row reason in the field's docstring.
        zm.capy_art = scene["capy"]
        # Publish the resolved value so build_plate_fine's EXACT vector builders
        # bake at the SAME waterline these raster zones were carved at.
        zm.waterline_y = waterline_y
        _mask_rim(zm.water_band, spec.weld_margin_um, pitch_um)
        _mask_rim(zm.capy_body, spec.weld_margin_um, pitch_um)

    return zm


def _mask_rim(mask: np.ndarray, margin_um: float, pitch_um: float) -> None:
    """Zero the keep-out rim (foil overlap) in place — the fine writer must not
    lay gold under the copper foil."""
    if margin_um <= 0:
        return
    m = max(1, int(round(margin_um / pitch_um)))
    mask[:m, :] = False
    mask[-m:, :] = False
    mask[:, :m] = False
    mask[:, -m:] = False


# =========================================================================== #
#  SCANIMATION — EXACT vector geometry (F3/F7, blockers 1+2)                    #
# =========================================================================== #
# The coarse zone-mask path resampled the 60/15/45 barrier and the N=4
# interleaved ripple frames through the ~39.5 µm boundary raster, so the 15 µm
# slot (0.38 cell) was unreachable: the barrier aliased to a ~70-118 µm comb and
# the back frames came out empty (audit blockers 1+2). These builders instead
# emit the scanimation as EXACT vector rectangles in plate coords:
#
#   * the slit-barrier BARS are a 60 µm-pitch comb of 45 µm-wide vertical bars,
#     clipped to the water-band rectangle and CARVED around the submerged animal
#     — pure geometry, no raster;
#   * the N interleaved back frames put ripple phase k into slot k (a 15 µm-wide
#     vertical column) of every 60 µm period; each slot's crest y-runs come from
#     the SAME analytic flow field the shader/pattern use (``capyscan._flow_*``),
#     sampled finely in y at the slot's own columns and floored to the 2 µm
#     litho minimum — so the slot WIDTH is exactly 15 µm and the crest THICKNESS
#     is ≥ 2 µm, nothing between.
#
# The water band's OUTLINE is analytic in plate coords — the lower
# ``(1 - waterline_y)`` of the art-box square, full width — but it is NOT a plain
# half-plane: the SUBMERGED CAPYBARA IS CARVED OUT of it (the pattern's
# ``water_band`` is ``below & ~capy``, matching the composed preview). The ripple
# band covers the water AROUND the animal, so both builders take the capybara
# silhouette (``ZoneMasks.capy_art``, the same raster the zone masks were cut
# from) and drop every bar segment / crest run that lands on it. Two consequences
# worth keeping in mind:
#   * the body affects crest AMPLITUDE (the calm patch, analytic) AND the emitted
#     extent (the carve, silhouette-driven) — the old "no silhouette raster is
#     needed here" note was what let the fine GDS print ripples over the animal
#     while the preview showed clean carrier there;
#   * the carve is applied to the y-SAMPLE, before the run close/open, so every
#     surviving segment and every carved gap is still a whole number of ≥2 µm
#     samples — the carve cannot mint a sub-floor sliver.


def _art_box_um(spec: Any) -> tuple[float, float, float, float]:
    """Plate-coord bbox (x0,y0,x1,y1 µm, y up) of the centerpiece art square.

    The centerpiece is a square of side ``CENTERPIECE_FILL·aperture`` centered on
    the plate (see ``plates._aperture`` / ``CENTERPIECE_FILL``); the capybara +
    water live in it in normalized art-box coords (x,y in 0..1, y DOWN). This box
    is the BODY registration frame: the flow-field wake (body center, calm patch)
    is normalized against it so the current organizes around the capybara.
    """
    from . import plates as P

    aperture = P._aperture(spec)
    side = P.CENTERPIECE_FILL * aperture
    h = side / 2.0
    return (-h, -h, h, h)


def _water_box_um(spec: Any) -> tuple[float, float, float, float]:
    """Plate-coord bbox (x0,y0,x1,y1 µm, y up) of the WATER band's extent.

    WATER FULL WIDTH: the water band spans the whole window edge-to-edge
    horizontally (the full WIDTH-axis aperture ``_aperture_width_um``, NOT the
    square ``_aperture`` that sizes the body), while its VERTICAL extent stays
    the body square's (so the waterline lines up with the half-submerged
    capybara). The barrier comb + interleaved back frames iterate x over THIS
    box; the flow field's crest shape/wake stays normalized to the body square
    (``_art_box_um``) so the wings just continue the periodic streamline current
    past the animal.
    """
    from . import plates as P

    aperture = P._aperture(spec)
    side = P.CENTERPIECE_FILL * aperture
    hy = side / 2.0
    hx = P._aperture_width_um(spec) / 2.0
    return (-hx, -hy, hx, hy)


def _artbox_norm_to_plate(
    xn: np.ndarray | float, yn: np.ndarray | float, art_bbox: tuple[float, float, float, float]
):
    """Normalized art-box (xn,yn in 0..1, y DOWN) → plate µm (x right, y UP)."""
    x0, y0, x1, y1 = art_bbox
    side = x1 - x0
    px = x0 + xn * side
    py = y1 - yn * side   # y-down normalized → y-up plate
    return px, py


def _sample_art_mask(
    mask: np.ndarray, xn: np.ndarray, yn: np.ndarray
) -> np.ndarray:
    """Nearest-neighbour sample of a SQUARE art-box mask at normalized coords.

    ``mask`` is the ``(n, n)`` art-box raster (y DOWN, the Pillow order every motif
    silhouette uses); ``xn`` / ``yn`` are normalized art-box coordinates that
    BROADCAST against each other (the builders pass an ``(S,1)`` column of x's and
    a ``(1,n_y)`` row of y's). Samples outside the unit box read False — the
    full-width water band's wings lie outside the body square, so "off the box" and
    "not the animal" are the same answer there.

    Nearest sampling (not interpolation) is deliberate: the mask IS the silhouette
    the rastered zones and the composed preview were cut from, so sampling it this
    way keeps the vector geometry and the zone masks agreeing cell-for-cell about
    where the animal is.
    """
    n_r, n_c = mask.shape
    ci = np.floor(np.asarray(xn, dtype=np.float64) * n_c).astype(np.int64)
    ri = np.floor(np.asarray(yn, dtype=np.float64) * n_r).astype(np.int64)
    inside = (ci >= 0) & (ci < n_c) & (ri >= 0) & (ri < n_r)
    vals = mask[np.clip(ri, 0, n_r - 1), np.clip(ci, 0, n_c - 1)]
    return vals & inside


def _carve_submerged_from_bars(
    bars: np.ndarray,
    art_bbox: tuple[float, float, float, float],
    body_art: np.ndarray,
    y_samp_um: float,
) -> np.ndarray:
    """Split full-band vertical rects around the SUBMERGED capybara silhouette.

    The ripple band covers the water AROUND the animal, so the slit-barrier comb
    must stop at the body: a bar striping across the submerged capybara is gold
    printed ON the animal (and it is not even a barrier there — there is no back
    ripple lane left to gate). Each bar is sampled down its span on a ``y_samp_um``
    grid, samples that land on the silhouette are dropped, and every surviving
    contiguous run becomes one rect. Sampling at the 2 µm litho floor makes each
    emitted segment and each carved gap a whole number of ≥2 µm samples, so the
    carve cannot mint a sub-floor sliver.

    The probe is CONSERVATIVE: the body is sampled at the bar's left edge, center
    and right edge, and any hit clears the sample. A bar that straddles the
    silhouette outline is therefore cut back by up to its own 45 µm width rather
    than left half-printed over the animal — a hair more open water at the outline,
    never gold on the capybara.

    Bars outside the body square in x are returned untouched (they cannot overlap
    it), which is most of a full-width band.

    ``body_art`` is the FULL silhouette (see ``ZoneMasks.capy_art``); the bars only
    exist inside the water band, so everything it flags here is submerged.
    """
    x0b, _y0b, x1b, y1b = art_bbox
    side = x1b - x0b
    touch = (bars[:, 1] > x0b) & (bars[:, 0] < x1b)
    if not touch.any():
        return bars
    hit = bars[touch]
    # Every bar spans the same water band in y (see the builder), so one sample
    # ladder serves them all.
    band_y1 = float(hit[0, 3])
    band_y0 = float(hit[0, 2])
    band_h = band_y1 - band_y0
    if band_h <= 0.0:
        return bars
    n_y = max(2, int(round(band_h / y_samp_um)))
    dy = band_h / n_y
    ys = band_y1 - (np.arange(n_y) + 0.5) * dy      # plate y, top→bottom
    yn = ((y1b - ys) / side)[None, :]               # art-box normalized, y-down
    eps = 1e-6
    on_body = np.zeros((hit.shape[0], n_y), dtype=bool)
    for xs in (hit[:, 0] + eps, 0.5 * (hit[:, 0] + hit[:, 1]), hit[:, 1] - eps):
        on_body |= _sample_art_mask(body_art, ((xs - x0b) / side)[:, None], yn)
    keep = ~on_body
    # Same floor discipline as the back-frame builder: CLOSE bridges a carved gap
    # thinner than the floor (a printer would bridge it anyway), OPEN drops a
    # surviving segment thinner than the floor. A no-op at the 2 µm default sample,
    # and the guarantee if a caller samples finer.
    keep = _close_and_open_rows(keep, max(1, int(round(LITHO_FLOOR_UM / dy))))
    rows, r_start, r_end = _row_runs(keep)
    parts = [bars[~touch]]
    if rows.size:
        ry1 = ys[r_start] + dy / 2.0
        ry0 = ys[r_end - 1] - dy / 2.0
        parts.append(np.stack([hit[rows, 0], hit[rows, 1], ry0, ry1], axis=1))
    return _concat_rects(parts)


def _scanimation_barrier_bar_rects(
    spec: Any,
    waterline_y: float,
    frame_pitch_um: float,
    n_phases: int,
    body_art: np.ndarray | None = None,
    *,
    y_samp_um: float = 2.0,
) -> np.ndarray:
    """FRONT slit-barrier BARS as exact vector rects (60 µm pitch, 45 µm bar).

    The barrier's OPEN slot is ``frame_pitch/n_phases`` (15 µm) and the closed
    BAR is ``frame_pitch·(1−1/n_phases)`` (45 µm). Bars are vertical, span the
    water band in y, and repeat at exactly ``frame_pitch`` across the band width
    — one rect per period. Slot boundaries snap to the SAME period lattice the
    back interleave uses (both anchored at art-box x0), so a slot's open column
    lines up over its back ripple lane.

    ``body_art`` (``ZoneMasks.capy_art``) is the capybara silhouette in art-box
    coords; when given, the comb is CARVED around the animal so no bar prints over
    it (see :func:`_carve_submerged_from_bars`). A carved bar becomes several
    shorter rects; the period lattice is untouched.
    """
    # WATER FULL WIDTH: bars span the full aperture-width water box; the band's
    # vertical extent + waterline come from the body square (y matches the animal).
    art_bbox = _art_box_um(spec)
    x0, y0, x1, _y1 = _water_box_um(spec)
    # Water band: normalized y in [waterline_y, 1] → plate y in [y0, y_top].
    _, y_top = _artbox_norm_to_plate(0.0, waterline_y, art_bbox)
    band_y0, band_y1 = y0, y_top
    slot_um = frame_pitch_um / n_phases
    # Period 0 open slot starts at the band's left edge x0; the bar is the closed
    # remainder of the period. Emit the bar [x0 + slot, x0 + period] per period.
    n_periods = int(math.ceil((x1 - x0) / frame_pitch_um)) + 1
    rects = []
    for k in range(n_periods):
        bx0 = x0 + k * frame_pitch_um + slot_um
        bx1 = x0 + (k + 1) * frame_pitch_um
        cx0 = max(bx0, x0)
        cx1 = min(bx1, x1)
        if cx1 - cx0 <= 1e-9:
            continue
        rects.append((cx0, cx1, band_y0, band_y1))
    if not rects:
        return np.empty((0, 4), dtype=float)
    bars = np.asarray(rects, dtype=float)
    if body_art is None or not body_art.any():
        return bars
    return _carve_submerged_from_bars(bars, art_bbox, body_art, y_samp_um)


# Slots evaluated per vectorised chunk in _scanimation_back_frame_rects. The flow
# field is ELEMENTWISE, so a chunk of S slots is one (S,1)×(1,n_y) broadcast whose
# per-element operands (and their order) are exactly the old per-slot 1-D ones —
# and every Y-ONLY term (depth, shear, the wake tanh, the centerline gaussian) is
# then evaluated ONCE per chunk instead of once per slot. Chunked rather than done
# in one shot so the S×n_y temporaries stay in the low tens of MB.
_SCAN_SLOT_CHUNK = 256


def _scanimation_back_frame_rects(
    spec: Any,
    waterline_y: float,
    frame_pitch_um: float,
    n_phases: int,
    body_art: np.ndarray | None = None,
    *,
    y_samp_um: float = 2.0,
) -> np.ndarray:
    """BACK N-phase interleaved ripple frames as EXACT vector rects.

    For every 60 µm period across the water band, slot k (a 15 µm-wide vertical
    column, k = 0..N−1) carries ripple phase k. The crest shape for a slot is the
    analytic flow field (``capyscan._flow_streamline_field`` /
    ``_flow_amplitude``) evaluated at the slot's CENTER x across a y sample
    quantized to ``y_samp_um``. Setting this to the 2 µm floor is deliberate: it
    snaps every crest y-boundary to the floor grid, so the small y-offset between
    ADJACENT slots' crests (their flow field is sampled at different x) becomes a
    step of 0 or ≥ 2 µm — never a 1 µm sub-floor notch at the slot boundary. The
    column is then run-closed/opened so each crest and each gap within a slot is
    also ≥ 2 µm. Each surviving y-run becomes ONE rect spanning the slot's exact
    15 µm width; nothing sub-floor survives (F3).

    ``body_art`` (``ZoneMasks.capy_art``) is the capybara silhouette in art-box
    coords. When given, the crest samples that land on the
    animal are CARVED before the run close/open, so the interleave stops at the
    body outline — the ripple band is the water AROUND the capybara, and the
    submerged body keeps the plain back carrier (which `_build_zone_masks`' carved
    ``water_band`` leaves in place there). Carving on the sample ladder, ahead of
    the floor pass, is what keeps a carved crest tip from becoming a sub-floor
    sliver. The probe is conservative in x (slot edges AND center, any hit clears),
    so a slot straddling the outline loses that sample rather than half-printing
    over the animal.

    Memory: one CHUNK of ``_SCAN_SLOT_CHUNK`` slot-columns of ``band_h/y_samp``
    samples at a time, streamed chunk-by-chunk — never the whole band raster (the
    blocker's ~3.8M-cell trap). The rect count is bounded by
    (#slots × #crests-per-slot) ≈ a few ×10³.
    """
    from .patterns.artistic import capybara_scanimation as capyscan

    # WATER FULL WIDTH: slots tile the full aperture-width water box; the flow
    # field's crest shape/wake stays normalized to the body square (art box), so
    # the wings continue the periodic current past the animal.
    art_bbox = _art_box_um(spec)
    wx0, y0, wx1, _wy1 = _water_box_um(spec)
    _, y_top = _artbox_norm_to_plate(0.0, waterline_y, art_bbox)
    body_x0 = art_bbox[0]
    body_side = art_bbox[2] - art_bbox[0]   # body square side (flow normalization)
    water_w = wx1 - wx0                      # full water band width (x iteration)
    slot_um = frame_pitch_um / n_phases
    wavelength = capyscan.RIPPLE_WAVELENGTH_NORM

    # Fine y sample across the water band (normalized, y-down). Band spans
    # normalized y in [waterline_y, 1]. Sample at y_samp_um plate spacing.
    band_h_um = y_top - y0
    n_y = max(2, int(round(band_h_um / y_samp_um)))
    # Plate y samples top→bottom (descending) so a run index maps to a y-interval.
    ys_plate = y_top - (np.arange(n_y) + 0.5) * (band_h_um / n_y)
    # Normalized BODY-square y (y-down) for each sample (water box y == body y).
    yn = (art_bbox[3] - ys_plate) / body_side

    n_periods = int(math.ceil(water_w / frame_pitch_um)) + 1
    min_run = max(1, int(round(LITHO_FLOOR_UM / (band_h_um / n_y))))
    dy = band_h_um / n_y

    # Slot table flattened in (period, phase) order — the SAME order the per-slot
    # double loop emitted, so the rect SEQUENCE is unchanged. sx0 keeps the old
    # two-step form (period origin first, then k·slot) so the floats are identical.
    p_i = np.arange(n_periods, dtype=np.float64)[:, None]
    k_i = np.arange(n_phases, dtype=np.float64)[None, :]
    sx0 = (wx0 + p_i * frame_pitch_um) + k_i * slot_um
    sx1 = (sx0 + slot_um).ravel()
    sx0 = sx0.ravel()
    phase_step = np.broadcast_to(k_i, (n_periods, n_phases)).ravel()
    cx0 = np.maximum(sx0, wx0)
    cx1 = np.minimum(sx1, wx1)
    live = np.flatnonzero((cx1 - cx0) > 1e-9)

    # Sample the flow field for phase k at each slot's center x, normalized to the
    # BODY square (wings fall outside [0,1] → periodic continuation). X is one
    # column per slot, Y one row of band samples: the broadcast product is the
    # elementwise field the old per-slot calls computed, slot by slot.
    yn_row = yn[None, :]
    parts: list[np.ndarray] = []
    for c in range(0, live.size, _SCAN_SLOT_CHUNK):
        sel = live[c : c + _SCAN_SLOT_CHUNK]
        xc = 0.5 * (sx0[sel] + sx1[sel])
        xn = ((xc - body_x0) / body_side)[:, None]
        s = capyscan._flow_streamline_field(
            xn, yn_row, wavelength, n_phases, phase_step[sel][:, None], waterline_y
        )
        f = s - np.floor(s + 0.5)
        amp = capyscan._flow_amplitude(xn, yn_row, waterline_y)
        half = 0.16 * amp
        gold = (np.abs(f) < half) & (amp > 0.05)
        if body_art is not None:
            # Carve the animal out of the band (the pattern does the same with
            # ``crest & water_band``; these samples are all inside the band, so a
            # silhouette hit here IS the submerged body). Probe both slot edges and
            # the center so a slot straddling the outline is cleared, not
            # half-printed.
            on_body = np.zeros_like(gold)
            for xs in (sx0[sel], xc, sx1[sel]):
                on_body |= _sample_art_mask(
                    body_art, ((xs - body_x0) / body_side)[:, None], yn_row
                )
            gold &= ~on_body
        if not gold.any():
            continue
        # Floor BOTH the crest thickness and the gap between crests in each slot
        # to the 2 µm litho minimum: close any sub-floor y-gap (merge crests that
        # nearly touch — a printer would bridge them) then drop any run still
        # thinner than the floor. Guarantees every emitted run is ≥ 2 µm tall AND
        # every gap ≥ 2 µm, so the merged-geometry DRC finds nothing sub-floor in
        # the back frames. Row-wise: each slot column is closed/opened alone.
        gold = _close_and_open_rows(gold, min_run)
        # Contiguous y-runs, all slots at once. np.nonzero is C-order, so runs come
        # out slot-major / top→bottom — the old append order exactly.
        rows, r_start, r_end = _row_runs(gold)
        if rows.size == 0:
            continue
        long_enough = (r_end - r_start) >= min_run   # no-op after OPEN; kept as the gate
        rows = rows[long_enough]
        r_start = r_start[long_enough]
        r_end = r_end[long_enough]
        if rows.size == 0:
            continue
        # Run rows r_start..r_end-1 (top→bottom in plate y). Row i covers
        # [ys_plate[i]-dy/2, ys_plate[i]+dy/2].
        ry1 = ys_plate[r_start] + dy / 2.0
        ry0 = ys_plate[r_end - 1] - dy / 2.0
        parts.append(np.stack([cx0[sel][rows], cx1[sel][rows], ry0, ry1], axis=1))
    return _concat_rects(parts)


def _bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a 1-D bool array → list of (start, end) inclusive.

    Kept as the INDEPENDENT 1-D reference for :func:`_row_runs` (same runs, one
    row, exclusive→inclusive end): the scanimation builder runs on the row-wise
    vectorised pair, and this is what a parity test pins them against. Not on any
    hot path — do not route it through the vectorised form, or the reference stops
    being independent.
    """
    if not mask.any():
        return []
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _row_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-ROW contiguous True runs of a 2-D bool array, all rows at once.

    Returns ``(rows, starts, ends)`` — ``ends`` EXCLUSIVE — in C order, i.e. row
    ascending then column ascending, which is the order a per-row
    :func:`_bool_runs` loop would visit them. Padding a False column on each side
    keeps every run row-local, so the two ``np.nonzero`` results pair up
    one-for-one within each row.
    """
    s_px, l_px = mask.shape
    padded = np.zeros((s_px, l_px + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1)
    rows, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)
    return rows, starts, ends


def _fill_runs(shape: tuple[int, int], rows: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Bool mask of ``shape`` with ``[starts, ends)`` set on each row.

    Difference-array + cumsum instead of a Python slice loop. The runs handed in
    are disjoint WITHIN a row (they come from :func:`_row_runs`, where a gap of at
    least one cell separates consecutive runs), so the two scatter writes never
    collide and plain fancy indexing is exact. int16 accumulator: the running sum
    is only ever 0 or 1.
    """
    s_px, l_px = shape
    acc = np.zeros((s_px, l_px + 1), dtype=np.int16)
    acc[rows, starts] += 1
    acc[rows, ends] -= 1
    return np.cumsum(acc, axis=1, dtype=np.int16)[:, :l_px] > 0


def _close_and_open_rows(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Row-wise 1-D CLOSE then OPEN of a 2-D bool array by ``min_run`` samples.

    Same operator as :func:`_close_and_open_1d`, applied independently to every
    row with no Python run loop: CLOSE fills any False gap shorter than
    ``min_run`` that is flanked by True on BOTH sides (a run touching either end
    of the row is never filled), OPEN then drops any True run shorter than
    ``min_run``. Both passes read their runs from the mask as it stood BEFORE the
    pass — exactly what the sequential 1-D version did, since its in-place writes
    could not change the run list it had already extracted.
    """
    if min_run <= 1 or mask.size == 0:
        return mask
    s_px, l_px = mask.shape
    out = mask.copy()
    # CLOSE: short INTERIOR False runs (start > 0 and end < row length).
    rows, starts, ends = _row_runs(~out)
    sel = (starts > 0) & (ends < l_px) & ((ends - starts) < min_run)
    if sel.any():
        out |= _fill_runs((s_px, l_px), rows[sel], starts[sel], ends[sel])
    # OPEN: short True runs, including the ones touching a row end.
    rows, starts, ends = _row_runs(out)
    sel = (ends - starts) < min_run
    if sel.any():
        out &= ~_fill_runs((s_px, l_px), rows[sel], starts[sel], ends[sel])
    return out


def _close_and_open_1d(mask: np.ndarray, min_run: int) -> np.ndarray:
    """1-D CLOSE then OPEN of a bool array by ``min_run`` samples.

    CLOSE fills any False gap shorter than ``min_run`` (sub-floor gap between two
    crests → bridged), OPEN drops any True run shorter than ``min_run`` (sub-floor
    crest tip → removed). Leaves every surviving run and gap ≥ ``min_run`` samples,
    i.e. ≥ the 2 µm floor.

    Kept as the INDEPENDENT 1-D reference for :func:`_close_and_open_rows`, which
    is what the scanimation builder uses (every slot column at once) — a parity
    test pins the two. Not on any hot path; do not reimplement it in terms of the
    row-wise form, or the reference stops being independent."""
    if min_run <= 1 or mask.size == 0:
        return mask
    m = mask.copy()
    # CLOSE: fill short False runs that are flanked by True on both sides.
    inv = ~m
    for a, b in _bool_runs(inv):
        if a > 0 and (b + 1) < m.size and (b - a + 1) < min_run:
            m[a : b + 1] = True
    # OPEN: drop short True runs.
    for a, b in _bool_runs(m):
        if (b - a + 1) < min_run:
            m[a : b + 1] = False
    return m


def _erode_zone(mask: np.ndarray, cells: int = 1) -> np.ndarray:
    """Erode a bool zone by ``cells`` grid cells (4-neighbour), fully vectorised.

    Used to open a SEAM GUTTER between adjacent zones filled with DIFFERENT-angle
    gratings (frame angle buckets; the 45° accent vs the 0° centerpiece carrier).
    Where two such zones abut, their rotated grating lines cross into sub-floor
    wedge tips / gaps at the boundary (blocker 3's zone/bucket seam class); giving
    each zone a ≥1-cell (≈ pitch, tens of µm ≫ 2 µm floor) inset means the two
    gratings never touch, so no seam crossing is created in the first place. The
    inset is invisible (a fraction of the frame band) and costs nothing optically.
    """
    if cells <= 0 or not mask.any():
        return mask
    m = mask
    for _ in range(cells):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        m = e
    return m


# =========================================================================== #
#  per-plate fine geometry — front + back rect/poly sets                       #
# =========================================================================== #

@dataclass
class PlateFine:
    """One plate's fine geometry, plate-centered (origin = plate center, y up).

    ``front_rects`` / ``back_rects`` are axis-aligned ``(N,4)`` [x0,x1,y0,y1] µm
    rectangles in the PLATE frame (vertical gratings, scanimation slots) —
    cleaned by the fast ``drc_clean_rects``.

    ``front_angled`` / ``back_angled`` are angled gratings stored as
    ``(local_rects, angle_deg)`` groups: the rects are axis-aligned EXACT gold
    lines in the grating-LOCAL frame (so ``drc_clean_rects`` cleans them cheaply),
    and ``angle_deg`` is applied only at GDS placement — rotated to polygons then
    (per the DRC contract, and so a uniform grating never pays the per-polygon
    morphological DRC cost). The GDS writer composes ``angle_deg`` with any plate
    90° packing rotation when it emits.
    """

    face: str
    slug: str
    front_rects: np.ndarray
    back_rects: np.ndarray
    front_angled: list[tuple[np.ndarray, float]]
    back_angled: list[tuple[np.ndarray, float]]
    # PRINTED plate-frame geometry after the merged-region DRC heal (blocker 3):
    # a list of (K,2) µm vertex rings per layer, y up, in the PLATE frame (angled
    # groups already rotated in). This is what the GDS writer emits — the rects /
    # angled fields above are retained only for the per-group before/after DRC
    # report and stats. Empty list ⇒ that layer had no geometry.
    front_polys: list[np.ndarray] = field(default_factory=list)
    back_polys: list[np.ndarray] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def build_plate_fine(spec: Any, face: str, *, drc_before_report: bool = False) -> PlateFine:
    """Compose one plate's front+back fine geometry at native periods.

    ``drc_before_report``: also measure the PRE-heal merged layers
    (``*_merged_before`` blocks). Off by default — the heal runs
    unconditionally and the AFTER report is the printed-geometry gate; the
    before-report is audit narrative that doubles the check cost. The CLI
    wafer/audit paths pass True.

    ``spec`` is the face's ``PlateSpec`` (already normalized). Periods come from
    ``plates._carrier_recipe_data`` (the fab_* fields — the TRUE optical values,
    not the coarse budget rescale). Each zone is processed with vectorised numpy
    run-extraction, so the in-RAM rect set stays bounded (a few ×10⁴ per plate,
    the GDS writer streams one plate-layer at a time).
    """
    from . import plates as P

    rd = P._carrier_recipe_data(spec)
    back_period = float(rd["fab_back_period_um"])       # 22.0
    front_period = float(rd["fab_front_period_um"])     # 23.98
    angle_off = float(rd["fab_angle_offset_deg"])       # 3.0
    base_angle = float(rd["carrier_angle_deg"])
    duty = float(rd["grating_duty"])
    frame_span = float(rd["frame_angle_span_deg"])
    frame_count = int(round(float(rd["frame_bucket_count"])))
    center_period = float(rd.get("fab_center_period_um", rd["center_period_um"]))  # 60.0
    center_axis = float(rd["switch_axis_deg"])
    is_interlace = spec.pattern_slug in P.SWITCH_INTERLACE_SLUGS
    # PHOTOGRAPH: the centerpiece is a line screen, emitted as exact rectangles
    # from ``plates.photo_band_rects`` (front layer only) rather than a
    # silhouette filled with the switch carrier.
    is_photo = spec.pattern_slug == P.PHOTO_SLUG
    # SINGLE PLY: no inner ply exists, so the uniform carrier joins the leaves on
    # the FRONT layer and the back layer stays empty.
    single_ply = bool(getattr(spec, "single_ply", False))

    W, H = spec.width_um, spec.height_um

    # Zone-boundary raster pitch. Fine ENOUGH to resolve the finest zone edge
    # feature (scanimation 15 µm slot, accent), but this is BOUNDARY only — the
    # periods are exact vector geometry regardless. Cap the grid so the source
    # masks stay small (well under the lattice budget). ~4 samples on the finest
    # slot, floored so a huge plate does not explode the source raster.
    from .patterns._helpers import MAX_LATTICE_CELLS

    ideal = min(P.water_scan_fab_pitch_um(spec) / P.WATER_SCAN_N_PHASES, back_period) / 4.0
    budget_pitch = math.sqrt(W * H / (0.9 * MAX_LATTICE_CELLS))
    pitch = max(ideal, budget_pitch)

    zm = _build_zone_masks(spec, pitch)
    extent = (W, H)

    front_rects_parts: list[np.ndarray] = []
    back_rects_parts: list[np.ndarray] = []
    front_angled: list[tuple[np.ndarray, float]] = []
    back_angled: list[tuple[np.ndarray, float]] = []
    stats: dict[str, Any] = {"pitch_um": pitch}

    # --- FRONT frame band: per-motif angle-bucket gratings (angled) --------
    # Each bucket b fills its cells with a grating rotated to
    # base + angle_off + (b - (N-1)/2)·span. Emitted as grating-local rects +
    # angle (rotated to polys only at GDS placement).
    if zm.frame.any() and single_ply:
        # ONE PLY: no carrier to beat against, so the leaves are fine
        # diffractive gratings (plates.SINGLE_PLY_LEAF_PERIOD_UM) with one
        # orientation PER MOTIF FAMILY — the frame's angle buckets, fanned over
        # the half-turn instead of the two-ply 1 deg — so each family flashes
        # on its own under a lamp. The bucket seam gutter (1 cell) keeps the
        # differently angled gratings from crossing where families touch.
        # WHICH fill (line angles, a period ladder, pads) is the selectable part;
        # see app/leaf_fills.py. "lines" is the default and emits exactly what
        # this branch emitted before the knob existed.
        from . import leaf_fills as LF

        leaf_period = P.SINGLE_PLY_LEAF_PERIOD_UM
        fill = str(getattr(P, "SINGLE_PLY_LEAF_FILL", "lines"))
        hue_periods = tuple(getattr(P, "SINGLE_PLY_LEAF_HUE_PERIODS_UM", ()))
        dot_cov = float(getattr(P, "SINGLE_PLY_LEAF_DOT_COVERAGE", 0.5))
        floor_p = LF.min_period_um(fill, leaf_period, coverage=dot_cov)
        probe = min([p for p, _ in LF.bucket_layers(fill, 0, frame_count, 0.0,
                                                    leaf_period, hue_periods)] or [leaf_period])
        if probe < floor_p - 1e-9:
            raise ValueError(
                f"single-ply leaf fill {fill!r} at {probe} um breaks the 2 um litho "
                f"floor (needs p >= {floor_p:.2f} um)")
        rng = np.random.default_rng(int(getattr(spec.frame, "seed", 1)) * 7919 + 17)
        fan0 = float(rng.uniform(0.0, 180.0))          # the face's own starting angle
        n_used = 0
        for b in range(frame_count):
            lvl_lo = P.FRAME_BUCKET0 + b * P.FRAME_BUCKET_STEP - P.FRAME_BUCKET_STEP // 2
            lvl_hi = P.FRAME_BUCKET0 + b * P.FRAME_BUCKET_STEP + P.FRAME_BUCKET_STEP // 2
            in_bucket = (zm.frame_level > max(0, lvl_lo)) & (zm.frame_level <= lvl_hi)
            if not in_bucket.any():
                continue
            in_bucket = _erode_zone(in_bucket, 1)
            if not in_bucket.any():
                continue
            if fill == "dots":
                pads = LF.dot_rects(in_bucket, pitch, leaf_period, coverage=dot_cov)
                if pads.shape[0]:
                    front_rects_parts.append(pads)
            else:
                for per_um, ang in LF.bucket_layers(
                    fill, b, frame_count, fan0, leaf_period, hue_periods
                ):
                    _emit_grating(
                        in_bucket, pitch, extent, per_um, duty, ang,
                        0.0, front_rects_parts, front_angled,
                    )
            n_used += 1
        stats["single_ply_leaves"] = {
            "n_families": int(n_used), "duty": float(duty),
            "fan_start_deg": round(fan0, 2), "fan_step_deg": round(180.0 / max(1, frame_count), 2),
            **LF.describe(fill, leaf_period, hue_periods, coverage=dot_cov),
            # the ONE period the manifest advertises (ladder mean under "hue");
            # placed after describe() so it is not overwritten by the fill's knob
            "period_um": float(P.single_ply_leaf_period_um(spec)),
        }
    elif zm.frame.any():
        base_front = base_angle + angle_off
        for b in range(frame_count):
            lvl_lo = P.FRAME_BUCKET0 + b * P.FRAME_BUCKET_STEP - P.FRAME_BUCKET_STEP // 2
            lvl_hi = P.FRAME_BUCKET0 + b * P.FRAME_BUCKET_STEP + P.FRAME_BUCKET_STEP // 2
            in_bucket = (zm.frame_level > max(0, lvl_lo)) & (zm.frame_level <= lvl_hi)
            if not in_bucket.any():
                continue
            # Seam gutter: adjacent buckets carry gratings at DIFFERENT angles, so
            # inset each by 1 cell to keep their lines from crossing into sub-floor
            # wedges at the shared boundary (blocker 3 bucket-seam class).
            in_bucket = _erode_zone(in_bucket, 1)
            if not in_bucket.any():
                continue
            ang = base_front + (b - 0.5 * (frame_count - 1)) * frame_span
            _emit_grating(
                in_bucket, pitch, extent, front_period, duty, ang,
                0.0, front_rects_parts, front_angled,
            )

    # --- FRONT centerpiece (60 µm vertical) --------------------------------
    # Barrier-interlace faces (SWITCH_INTERLACE_SLUGS): the FRONT layer is a
    # NEUTRAL slit comb (open duty 0.5 → one 30 µm lane, phase −0.25) over the
    # FULL centerpiece art box — never the union of the silhouettes. A
    # union-gated comb is itself a static front image (its envelope is the
    # union, which cannot move with tilt — the measured ~0.31 front residual),
    # and the comb must cover every column any back lane can slide under
    # within the first zone (≥ p/2 beyond the union), so the whole
    # CENTERPIECE_FILL square gets the comb — the same treatment the capybara
    # water band already gets. The comb bar is a controlled 30 µm feature
    # (≫ 2 µm floor). Legacy phase-switch faces (front-only shimmer): the
    # FRONT silhouette filled with the phase-0 switch carrier. Both exclude
    # the accent zone (4.4 µm grating).
    if is_photo:
        # LINE SCREEN, front layer only, at the face's art box. Exact vector
        # rectangles: the bands themselves plus, inside every coloured band, its
        # vertical diffraction sub-grating (``screenrects.stripe_plan``'s whole
        # stripes). Built by ``plates.photo_band_rects``, the SAME function the
        # fab SVG bakes and (through ``photo_coverage``) the preview stamp reads,
        # so the picture in the GDS is the picture on screen.
        #
        # Nothing on the back: a halftone's tone IS its band height, and gold on
        # the inner ply would show through the gaps and lift every shadow.
        band_rects = P.photo_band_rects(spec)
        front_rects_parts.append(band_rects)
        stats["photo"] = {
            "n_band_rects": int(band_rects.shape[0]),
            "art_box_um": round(P.CENTERPIECE_FILL * P._aperture(spec), 1),
        }
    elif is_interlace:
        comb_box = zm.art_box if zm.art_box is not None else (zm.front_art | zm.back_art)
        comb_zone = _erode_zone(comb_box & ~zm.front_accent, 1)
        if comb_zone.any():
            _emit_grating(
                comb_zone, pitch, extent, center_period, 0.5, center_axis,
                -0.25, front_rects_parts, front_angled,
            )
    elif zm.front_art.any():
        # Exclude the accent zone (it gets the 4.4 µm grating instead). Inset the
        # carrier away from the accent so the 0° carrier and the 45° accent never
        # abut (their crossing would be a seam defect); the accent itself is inset
        # below so a 1-cell gutter sits between the two different-angle fields.
        art_no_accent = _erode_zone(zm.front_art & ~zm.front_accent, 1)
        if art_no_accent.any():
            _emit_grating(
                art_no_accent, pitch, extent, center_period, duty, center_axis,
                0.0, front_rects_parts, front_angled,
            )

    # --- FRONT accent: INTERLEAVED diffraction + moiré (both at 45°) -------
    # The accent used to be pure 4.4 µm diffraction: it could flash a rainbow
    # but never shimmer, because its only possible moiré partner (the 22 µm back
    # carrier) beats with 4.4 µm at 5.5 µm — far below anything an eye resolves.
    # It now carries BOTH effects, spatially interleaved in sub-acuity bands so
    # the eye integrates them as one surface (see gratings.band_select for why
    # interleaving beats nesting the fine grating inside the coarse one).
    #
    # Both gratings are written at the SAME 45° axis, so they share one
    # grating-local frame and the band lattice is a scalar test on local x —
    # no polygon clipping, and the two sets tile the zone exactly once.
    if zm.front_accent.any():
        # Inset the accent too so a 1-cell gutter sits between it and the 0°
        # carrier (accent zones are many cells wide, so the inset is negligible).
        accent_zone = _erode_zone(zm.front_accent, 1)
        if not accent_zone.any():
            accent_zone = zm.front_accent
        acc_angle = P._DIFFRACTION_ACCENT_ANGLE_DEG
        band_pitch = P._INTERLEAVE_BAND_PITCH_UM
        # Band A: the diffraction grating (rainbow).
        diff_local = _angled_grating_local_rects(
            accent_zone, pitch, extent,
            P._DIFFRACTION_ACCENT_PERIOD_UM, P._DIFFRACTION_ACCENT_DUTY, acc_angle,
        )
        diff_local = band_select(diff_local, band_pitch, want_odd=False)
        if diff_local.shape[0]:
            front_angled.append((diff_local, acc_angle))
        # Band B: a moiré louvre at the frame's front period, which beats
        # against the dedicated back patch emitted below.
        moire_local = _angled_grating_local_rects(
            accent_zone, pitch, extent, front_period, duty, acc_angle,
        )
        moire_local = band_select(moire_local, band_pitch, want_odd=True)
        if moire_local.shape[0]:
            front_angled.append((moire_local, acc_angle))

    is_scanimation = zm.water_band is not None

    # --- SINGLE PLY: the carrier moves to the FRONT layer -------------------
    # One sheet of glass, so there is no inner ply to carry the uniform carrier.
    # It joins the leaves on the outer ply over the back window MINUS the art
    # box, and every BACK emitter below is skipped. This IS the single-ply
    # carrier geometry — the production witness dies (witness_dies.build_face_die,
    # via build_plate_fine) are built from this block, not from a parallel one.
    # The 2-cell erosion is the same
    # seam gutter the two-ply carrier takes: an angled grating's rotated
    # rectangles end in slanted tips that would otherwise close the gap to the
    # abutting field below the litho floor.
    # 2026-09-10: a single ply carries NO carrier at all. The photograph and the
    # leaf gratings are the whole face; the picture's edge dissolves to bare
    # glass (photo.CARRIER_COV = 0) and the leaves stand on glass. A carrier on
    # the same ply as the leaves made a static union moire — bright bars where
    # the two gratings interleaved — which is not an effect one sheet can make
    # honestly, only a printed texture; Jay chose photo + frame without it.

    # --- BACK carrier grating (22 µm) over the whole window ----------------
    if not single_ply and zm.back_window.any():
        # Carrier fills the window MINUS the centerpiece art (art gets its own
        # phase-π switch carrier); on capybara the water band is scanimation.
        if is_scanimation:
            carrier_zone = zm.back_window & ~zm.water_band
        elif is_interlace:
            # Both silhouettes live on the back as interleaved lanes → clear the
            # plain carrier across their whole union so the lanes read.
            carrier_zone = zm.back_window & ~(zm.front_art | zm.back_art)
        else:
            carrier_zone = zm.back_window & ~zm.back_art
        # Seam gutter: the carrier (base_angle) abuts the 0° switch art (or, on
        # capybara, the exact-vector water-band crest field) at a different-angle
        # boundary. Inset it 2 cells, not 1: the carrier is an ANGLED grating and
        # its rotated-rectangle lines end in a SLANTED tip. A 1-cell gutter leaves
        # that tip descending to ~2 µm past the zone edge and raking toward the
        # abutting field, closing the seam to a sub-floor gap (measured: 15 gaps
        # 0.02–1.9 µm along the capybara waterline y≈1598.8 µm where crest tops sit
        # at the band edge, and 2 gaps 1.085 µm on the right plate where the 165°
        # carrier tip nears the 0° switch carrier). A 2-cell inset pulls the whole
        # slanted tip clear, opening a ≥2 µm gutter so no carrier line approaches
        # the neighbour field within the floor. The extra cell is one ~39 µm
        # boundary-raster step off a rim already inset by the weld margin —
        # optically invisible, and it only shrinks gold so it cannot add defects.
        # The accent zone gets its OWN back patch just below (at a small
        # crossing to the accent's 45° front bands), so keep the main carrier
        # out of it — two carriers at different angles in one place would beat
        # against each other as well as against the front.
        if zm.front_accent.any():
            carrier_zone = carrier_zone & ~zm.front_accent
        carrier_zone = _erode_zone(carrier_zone, 2)
        # base_angle is a per-face carrier rotation (seed-keyed); non-zero →
        # grating-local rects + angle, else exact axis rects.
        if carrier_zone.any():
            _emit_grating(
                carrier_zone, pitch, extent, back_period, duty, base_angle,
                0.0, back_rects_parts, back_angled,
            )

    # --- BACK patch under the accent (the accent moiré's partner) ----------
    # Laid at a SMALL crossing to the accent's 45° front bands so the pair beats
    # at a spacing the eye resolves; the frame's per-species angle fan does not
    # apply here because the accent is centerpiece geometry, not frame foliage.
    if not single_ply and zm.front_accent.any():
        acc_back = _erode_zone(zm.front_accent, 2)
        if acc_back.any():
            _emit_grating(
                acc_back, pitch, extent, back_period, duty,
                P._DIFFRACTION_ACCENT_ANGLE_DEG + P.ACCENT_MOIRE_OFFSET_DEG,
                0.0, back_rects_parts, back_angled,
            )

    # --- BACK centerpiece switch carrier (60 µm, phase 0.5 = EXACT 30 µm) --
    # The half-period offset makes the colibrí↔globe interlace. center_period is
    # EXACTLY 60 µm so phase 0.5 is EXACTLY a 30 µm shift in the shared frame.
    # switch_axis_deg is 0 in the confirmed plan → axis-aligned rects.
    if single_ply:
        # No back layer at all — see the single-ply block above.
        pass
    elif is_interlace:
        # Barrier-interlace BACK layer: both silhouettes interleaved in alternating
        # lanes (lane pitch = half the barrier period). A (front silhouette) fills
        # the even lanes — a 0.5-duty barrier-period grating at phase 0 gates the
        # first lane of each pair — and B (back silhouette) fills the odd lanes
        # (phase 0.5). Each lane is a controlled 30 µm gold column clipped to its
        # silhouette; nothing sub-lane survives the merged-region DRC heal below.
        a_zone = _erode_zone(zm.front_art, 1)
        if a_zone.any():
            _emit_grating(
                a_zone, pitch, extent, center_period, 0.5, center_axis,
                0.0, back_rects_parts, back_angled,
            )
        b_zone = _erode_zone(zm.back_art, 1)
        if b_zone.any():
            _emit_grating(
                b_zone, pitch, extent, center_period, 0.5, center_axis,
                0.5, back_rects_parts, back_angled,
            )
    elif zm.back_art.any() and not is_scanimation:
        # Inset 1 cell so the 0° switch carrier does not abut the base_angle back
        # carrier at the art silhouette boundary (different-angle seam → wedge
        # crossings). The carrier was already inset off ~back_art above, so this
        # opens a symmetric gutter between the two fields.
        switch_zone = _erode_zone(zm.back_art, 1)
        if switch_zone.any():
            _emit_grating(
                switch_zone, pitch, extent, center_period, duty, center_axis,
                0.5, back_rects_parts, back_angled,
            )

    # --- SCANIMATION (capybara) — EXACT vector geometry (blockers 1+2) ------
    # FRONT: capybara body shimmer (24 µm carrier grating, clipped to the dry
    #        silhouette via row-span) + slit-barrier BARS over the water band
    #        (exact 60 µm-pitch / 45 µm-bar vector comb).
    # BACK:  N=4 interleaved ripple frames — phase k in slot k of every 60 µm
    #        period, each slot an exact 15 µm-wide column, crest thickness sampled
    #        finely in y and floored to 2 µm. NONE of this passes through the
    #        coarse zone raster, so the 15 µm slot survives (was empty / aliased).
    if is_scanimation:
        # EFFECTIVE waterline, resolved ONCE in _build_zone_masks from the face's
        # ``waterline`` param (plates._water_waterline_y) — the same number the
        # composed preview mask, the fab SVG bake and recipe_data's
        # ``water_waterline_y`` carry. Taking it from the zone masks (not from
        # capyscan.WATERLINE_Y, which used to be hardwired here) is what keeps the
        # rastered body/band zones and these exact vector builders in one frame.
        # The fallback re-resolves rather than falling back to the constant, so a
        # future caller that builds ZoneMasks by hand still cannot drift.
        wl = (
            zm.waterline_y
            if zm.waterline_y is not None
            else P._water_waterline_y(spec.pattern_params)
        )
        fp = P.water_scan_fab_pitch_um(spec)
        nph = P.WATER_SCAN_N_PHASES
        carrier_um = P.WATER_SCAN_FAB_CARRIER_UM
        # Front body shimmer: native 24 µm vertical carrier clipped to the dry
        # capybara silhouette (row-span against the rastered body zone — the
        # PERIOD is exact vector, only the body boundary is quantized).
        if zm.capy_body is not None and zm.capy_body.any():
            # Inset 1 cell so the body shimmer does not abut the barrier bars at
            # the waterline (both 0° but distinct fields — a gutter avoids a
            # sub-floor gap where a body stripe nearly meets a bar).
            body_zone = _erode_zone(zm.capy_body, 1)
            if body_zone.any():
                _emit_grating(
                    body_zone, pitch, extent, carrier_um, duty, 0.0,
                    0.0, front_rects_parts, front_angled,
                )
        # SUBMERGED-BODY CARVE. Both vector builders take the capybara silhouette so
        # the comb and the interleave stop at the animal: the ripple band is the
        # water AROUND the capybara and the submerged body keeps the plain carrier
        # (which the carved ``zm.water_band`` above already leaves in place). Same
        # silhouette raster + same effective waterline as the zone masks, which is
        # the same algebra the composed preview carves its band with.
        body_art = zm.capy_art
        # Front slit-barrier bars (exact 60/45 vector comb over the water band).
        front_rects_parts.append(
            _scanimation_barrier_bar_rects(spec, wl, fp, nph, body_art)
        )
        # Back interleaved ripple frames (exact 15 µm slots, analytic crests).
        back_rects_parts.append(
            _scanimation_back_frame_rects(spec, wl, fp, nph, body_art)
        )
        stats["scanimation"] = {
            "frame_pitch_um": fp,
            "slot_um": fp / nph,
            "barrier_bar_um": fp * (1.0 - 1.0 / nph),
            "n_phases": nph,
            # Was the animal carved out of the band? False would mean the comb and
            # the ripple slots printed over the submerged body (the pre-carve bug).
            "submerged_body_carved": bool(body_art is not None and body_art.any()),
            # Recorded so a wafer run's stats show WHICH waterline was baked (it
            # is now a per-face param, not a module constant).
            "waterline_y": float(wl),
        }

    # --- concat + per-group rect DRC (fast) --------------------------------
    # drc_clean_rects is a cheap vectorised first pass on axis-aligned geometry
    # in each group's OWN frame — it snap-widens/drops per-feature sub-floor rects
    # before the merged gate. (The slow whole-array colinear-gap REPORT that used
    # to run here on the full concatenated rect set is dropped: the authoritative
    # measurement is the merged-region report below, which sees the actual printed
    # geometry; a per-group rect report cannot and just cost ~150 s/plate.)
    stats["drc"] = {}
    front_rects = _drc_rects(_concat_rects(front_rects_parts))
    back_rects = _drc_rects(_concat_rects(back_rects_parts))
    front_angled = [(_drc_rects(r), a) for (r, a) in front_angled]
    back_angled = [(_drc_rects(r), a) for (r, a) in back_angled]

    n_fa = sum(r.shape[0] for r, _ in front_angled)
    n_ba = sum(r.shape[0] for r, _ in back_angled)
    stats["front_rects"] = int(front_rects.shape[0])
    stats["back_rects"] = int(back_rects.shape[0])
    stats["front_angled"] = int(n_fa)
    stats["back_angled"] = int(n_ba)
    stats["front_total"] = int(front_rects.shape[0] + n_fa)
    stats["back_total"] = int(back_rects.shape[0] + n_ba)
    stats["periods"] = {
        "back_carrier_um": back_period,
        "front_leaf_um": front_period,
        "front_leaf_offset_deg": angle_off,
        "center_switch_um": center_period,
        "back_center_phase_offset_um": 0.5 * center_period,
    }

    # --- MERGED-geometry DRC (blocker 3): the PRINTED-layer gate -------------
    # The per-group drc_clean_rects above is a fast first pass on axis-aligned
    # geometry in each group's OWN frame; it cannot see the defects that appear
    # only once every group is UNIONED in the PLATE frame — sub-floor slivers
    # where two angled gratings overlap into wedge tips, and sub-floor gaps where
    # gratings from adjacent zones/buckets/slots nearly touch (the ~5000 sites the
    # audit found). We compose each layer's full plate-frame polygon set (angled
    # groups rotated in), build the merged klayout Region, MEASURE it, and heal it
    # (drop isolated sub-floor slivers non-destructively — see drc_clean_region).
    # The healed exterior rings are what the GDS writer emits, so the PRINTED
    # geometry is the measured one. before = merged Region of the composed layer;
    # after = merged Region of the heal.
    front_layer_polys = _compose_layer_polys(front_rects, front_angled)
    back_layer_polys = _compose_layer_polys(back_rects, back_angled)

    if drc_before_report:
        stats["drc"]["front_merged_before"] = drc_report_region(
            front_layer_polys, min_width_um=LITHO_FLOOR_UM, min_gap_um=LITHO_FLOOR_UM
        )
        stats["drc"]["back_merged_before"] = drc_report_region(
            back_layer_polys, min_width_um=LITHO_FLOOR_UM, min_gap_um=LITHO_FLOOR_UM
        )

    # ``report=True``: the AFTER report is the heal loop's own final check, not a
    # second pass over the same 200k polygons (see drc_clean_region) — same
    # measurement, one run of it instead of two.
    front_polys, stats["drc"]["front_merged_after"] = drc_clean_region(
        front_layer_polys, min_width_um=LITHO_FLOOR_UM, min_gap_um=LITHO_FLOOR_UM,
        report=True,
    )
    back_polys, stats["drc"]["back_merged_after"] = drc_clean_region(
        back_layer_polys, min_width_um=LITHO_FLOOR_UM, min_gap_um=LITHO_FLOOR_UM,
        report=True,
    )
    stats["front_polys"] = len(front_polys)
    stats["back_polys"] = len(back_polys)

    return PlateFine(
        face=face,
        slug=spec.pattern_slug,
        front_rects=front_rects,
        back_rects=back_rects,
        front_angled=front_angled,
        back_angled=back_angled,
        front_polys=front_polys,
        back_polys=back_polys,
        stats=stats,
    )


def _compose_layer_polys(
    rects: np.ndarray, angled: list[tuple[np.ndarray, float]]
) -> list[np.ndarray]:
    """One layer's full PLATE-frame polygon list: axis-aligned rects as 4-vertex
    rings + every angled grating-local group rotated into the plate frame. Feeds
    the merged-region DRC (blocker 3) which unions them and heals seam defects."""
    polys: list[np.ndarray] = []
    if rects.shape[0]:
        x0, x1, y0, y1 = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
        ring = np.stack(
            [
                np.stack([x0, y0], axis=1),
                np.stack([x1, y0], axis=1),
                np.stack([x1, y1], axis=1),
                np.stack([x0, y1], axis=1),
            ],
            axis=1,
        )  # (N,4,2)
        polys.extend(ring[i] for i in range(ring.shape[0]))
    for local_rects, angle_deg in angled:
        if local_rects.shape[0] == 0:
            continue
        verts = _rotate_rects_to_polys(local_rects, angle_deg)  # (N,4,2) plate frame
        polys.extend(verts[i] for i in range(verts.shape[0]))
    return polys


def _clip_axis_grating(
    zone: np.ndarray,
    pitch_um: float,
    extent_um: tuple[float, float],
    period_um: float,
    duty: float,
    angle_deg: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Axis-aligned (angle 0) grating clipped to a zone as column-merged PLATE-
    frame rects. angle 0 ⇒ vertical gold lines. For non-zero angles the caller
    routes through :func:`_angled_grating_local_rects` via :func:`_emit_grating`.
    """
    if abs(angle_deg) >= 1e-9:
        raise ValueError("_clip_axis_grating requires angle 0; use _emit_grating")
    line_rects = _line_rects_local(_zone_bbox(zone, pitch_um, extent_um), period_um, duty, phase)
    return _clip_rects_to_row_spans(line_rects, zone, pitch_um, extent_um)


def _emit_grating(
    zone: np.ndarray,
    pitch_um: float,
    extent_um: tuple[float, float],
    period_um: float,
    duty: float,
    angle_deg: float,
    phase: float,
    rects_out: list[np.ndarray],
    angled_out: list[tuple[np.ndarray, float]],
) -> None:
    """Dispatch a clipped grating to the right container by angle.

    angle ≈ 0 → exact axis-aligned PLATE-frame rects (appended to ``rects_out``).
    Non-zero angle → exact grating-LOCAL-frame rects + the angle, appended to
    ``angled_out`` as a ``(local_rects, angle_deg)`` group (rotated to polys only
    at GDS placement). Both routes clean via the fast ``drc_clean_rects``.
    """
    if abs(angle_deg) < 1e-9:
        rects_out.append(
            _clip_axis_grating(zone, pitch_um, extent_um, period_um, duty, 0.0, phase=phase)
        )
    else:
        local = _angled_grating_local_rects(
            zone, pitch_um, extent_um, period_um, duty, angle_deg, phase=phase
        )
        if local.shape[0]:
            angled_out.append((local, angle_deg))


def _angled_grating_local_rects(
    zone: np.ndarray,
    pitch_um: float,
    extent_um: tuple[float, float],
    period_um: float,
    duty: float,
    angle_deg: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Grating at ``angle_deg`` clipped to a zone → axis-aligned rects in the
    GRATING-LOCAL frame (the frame where the lines are vertical).

    Returns ``(N, 4)`` ``[x0, x1, y0, y1]`` µm rectangles that are EXACT gold
    lines (width = duty·period, y-runs merged) — but expressed in the rotated
    grating frame, NOT the plate frame. The caller keeps ``angle_deg`` alongside
    and rotates these to polygons only at GDS placement (per the DRC contract:
    "most output axis-aligned rects in grating-local frame, rotated as polygons
    only at placement"). Because they are axis-aligned here, the fast
    ``drc_clean_rects`` gate applies; a uniform grating's lines are floor-safe by
    construction (line = duty·period ≥ 2 µm for every period we use).

    Clip method: project each zone cell into the local frame and emit EVERY gold
    line whose stripe overlaps the cell's local-x SPAN — not just the line at the
    cell center (F/blocker 4). A single-index-per-cell bucketing fatally
    undersamples any period smaller than the boundary pitch: a 4.4 µm accent under
    a ~39.5 µm cell collapses ~9 lines to one, leaving no adjacent 4.4 µm pair. By
    expanding each cell to the index RANGE its projected width covers, a sub-cell
    period is realized in full while a super-cell period still emits exactly its
    one line. Then vectorised-merge the contiguous y-runs per line index. Only the
    zone BOUNDARY is quantized to ``pitch_um``; the period + line width are exact.

    A FILLED axis-aligned rectangle zone (the frame buckets' big blocks, the back
    carrier window, the art box) takes a separable fast path for the projection —
    see :func:`_zone_filled_rect`. It is bit-identical, not an approximation: the
    cell staircase this emits per line depends on WHICH cells project into the
    line's stripe, so an analytic rotate-the-bbox clip would move rect edges by up
    to a cell and change the boundary lines' x extents (the fine GDS is cached
    under ``api.export.FINE_GDS_VERSION``, so that is a version-bumping change,
    not a free one).
    """
    h_px, w_px = zone.shape
    hx = w_px * pitch_um / 2.0
    hy = h_px * pitch_um / 2.0
    # Rotate INTO the grating-local frame (rotate by -angle).
    a = math.radians(-angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    rect = _zone_filled_rect(zone)
    if rect is not None:
        r0, r1, c0, c1 = rect
        # Filled rectangle ⇒ the cell centers are a product grid, so lx/ly are
        # OUTER SUMS of a per-column and a per-row term: cx·ca, cx·sa, cy·ca, cy·sa
        # are evaluated (n_cols + n_rows) times instead of n_cells times, and
        # np.nonzero never has to enumerate the zone. Bit-identical to the general
        # path below: the same two products are summed per element in the same
        # order, and IEEE guarantees ``u - v == u + (-v)``. C-order ravel of the
        # (row, col) grid reproduces np.nonzero's row-major cell order.
        cx_c = np.arange(c0, c1 + 1) * pitch_um - hx + pitch_um / 2.0
        cy_r = hy - np.arange(r0, r1 + 1) * pitch_um - pitch_um / 2.0
        lx = ((cx_c * ca)[None, :] + (-(cy_r * sa))[:, None]).ravel()
        ly = ((cx_c * sa)[None, :] + (cy_r * ca)[:, None]).ravel()
    else:
        ys, xs = np.nonzero(zone)
        if xs.size == 0:
            return np.empty((0, 4), dtype=float)
        # Zone cell centers in µm (plate frame, y up).
        cx = xs * pitch_um - hx + pitch_um / 2.0
        cy = hy - ys * pitch_um - pitch_um / 2.0
        lx = cx * ca - cy * sa
        ly = cx * sa + cy * ca
    line_w = duty * period_um
    # A cell is a pitch×pitch square in the PLATE frame; its projection onto the
    # local-x axis spans ±hspan about lx (hspan = ½·pitch·(|cos|+|sin|)). Every
    # gold line whose stripe intersects [lx-hspan, lx+hspan] is present at this
    # cell's local-y. Enumerate the index range [k_lo, k_hi] per cell.
    hspan = 0.5 * pitch_um * (abs(ca) + abs(sa))
    k_lo = np.floor((lx - hspan) / period_um + phase).astype(int)
    k_hi = np.floor((lx + hspan) / period_um + phase).astype(int)
    span = k_hi - k_lo + 1                      # #period-cells each cell touches
    n_pairs = int(span.sum())
    if n_pairs <= 0:
        return np.empty((0, 4), dtype=float)
    # Build (cell, k) pairs by RAGGED indexing: cell c contributes lines
    # k_lo[c] .. k_hi[c]. Identical pairs, in the identical order, to the
    # (n_cells × max_span) bool mask + np.nonzero this replaces (C-order nonzero is
    # cell-major, offset-minor) — without materialising the mask. The pair count is
    # #cells · (avg period-cells per cell), bounded small for our zones (accent
    # ~10², carriers ~2-3× the cell count).
    cell_i = np.repeat(np.arange(span.size), span)
    k_all = k_lo[cell_i] + (np.arange(n_pairs) - np.repeat(np.cumsum(span) - span, span))
    # Keep only lines whose GOLD stripe actually overlaps this cell's x-span
    # (the period cell touches, but the gap half must not spuriously fill).
    gold_x0 = (k_all - phase) * period_um
    cell_lx = lx[cell_i]
    overlap = ((gold_x0 + line_w) > (cell_lx - hspan)) & (gold_x0 < (cell_lx + hspan))
    if not overlap.any():
        return np.empty((0, 4), dtype=float)
    idx_g = k_all[overlap]
    ly_g = ly[cell_i[overlap]]
    # Fully vectorised run extraction: sort by (line_idx, ly); a run breaks where
    # the line index changes OR the local-y gap exceeds ~1.5·pitch. Each run → one
    # local-frame rectangle. Duplicate (line, ly) pairs from adjacent cells are
    # harmless — they collapse into the same run.
    order = np.lexsort((ly_g, idx_g))
    idx_s = idx_g[order]
    ly_s = ly_g[order]
    n = idx_s.size
    brk = np.empty(n, dtype=bool)
    brk[0] = True
    brk[1:] = (idx_s[1:] != idx_s[:-1]) | (np.diff(ly_s) > pitch_um * 1.5)
    starts = np.flatnonzero(brk)
    ends = np.concatenate([starts[1:], [n]]) - 1  # inclusive last index per run
    y_lo = ly_s[starts] - pitch_um / 2.0
    y_hi = ly_s[ends] + pitch_um / 2.0
    k_run = idx_s[starts].astype(float)
    x_lo = (k_run - phase) * period_um
    x_hi = x_lo + line_w
    return np.stack([x_lo, x_hi, y_lo, y_hi], axis=1).astype(float)


def _rotate_rects_to_polys(local_rects: np.ndarray, angle_deg: float) -> np.ndarray:
    """(N,4) local-frame [x0,x1,y0,y1] rects → (N,4,2) vertex rings rotated CCW
    by ``angle_deg`` into the plate frame."""
    m = local_rects.shape[0]
    if m == 0:
        return np.empty((0, 4, 2), dtype=float)
    x_lo, x_hi, y_lo, y_hi = local_rects[:, 0], local_rects[:, 1], local_rects[:, 2], local_rects[:, 3]
    corners = np.empty((m, 4, 2), dtype=float)
    corners[:, 0, 0] = x_lo; corners[:, 0, 1] = y_lo
    corners[:, 1, 0] = x_hi; corners[:, 1, 1] = y_lo
    corners[:, 2, 0] = x_hi; corners[:, 2, 1] = y_hi
    corners[:, 3, 0] = x_lo; corners[:, 3, 1] = y_hi
    a_fwd = math.radians(angle_deg)
    caf, saf = math.cos(a_fwd), math.sin(a_fwd)
    rx = corners[:, :, 0] * caf - corners[:, :, 1] * saf
    ry = corners[:, :, 0] * saf + corners[:, :, 1] * caf
    return np.stack([rx, ry], axis=2)  # (m,4,2)


def _mask_to_row_span_rects(
    mask: np.ndarray, pitch_um: float, extent_um: tuple[float, float]
) -> np.ndarray:
    """A bool mask → column-merged vertical-run rectangles (for vertical-slot
    fields like the scanimation slots/bars, whose gold IS vertical stripes).

    Fully vectorised: a COLUMN run is a ROW run of the transpose, so we reuse
    the diff-based run finder on ``mask.T`` (one gold cell per column-run → one
    tall rectangle) then merge horizontally-adjacent same-band rects. No Python
    per-column loop.
    """
    if mask is None or not mask.any():
        return np.empty((0, 4), dtype=float)
    h_px, w_px = mask.shape
    hx = w_px * pitch_um / 2.0
    hy = h_px * pitch_um / 2.0
    # Vertical runs = horizontal runs of the transpose. In transpose space the
    # "row" index is the ORIGINAL column c, and the run "columns" are ORIGINAL
    # rows r (top→bottom). Pad + diff per transpose-row.
    g = mask.T.astype(np.int8)              # (w_px, h_px): [c, r]
    padded = np.zeros((w_px, h_px + 2), dtype=np.int8)
    padded[:, 1:-1] = g
    d = np.diff(padded, axis=1)
    cols, r_starts = np.nonzero(d == 1)     # cols = original column c
    _, r_ends = np.nonzero(d == -1)         # exclusive end (original row)
    if cols.size == 0:
        return np.empty((0, 4), dtype=float)
    x0 = cols * pitch_um - hx
    x1 = x0 + pitch_um
    # Original rows r_starts..r_ends-1 (y down). y up: top row r_start → y1.
    y1 = hy - r_starts * pitch_um
    y0 = hy - r_ends * pitch_um
    arr = np.stack([x0, x1, y0, y1], axis=1).astype(float)
    return _merge_adjacent_columns(arr)


def _merge_adjacent_columns(rects: np.ndarray) -> np.ndarray:
    """Merge horizontally-adjacent rects that share [y0,y1] into wider rects
    (colinear-run merge across columns) — the fewest-polygon representation of a
    vertical-slot field. Fully vectorised.

    Sort by (y0, y1, x0); a merge run is a maximal block of rows with the SAME
    (y0,y1) band whose x0 touches the previous x1. A new run starts wherever the
    band changes OR there is an x-gap; ``np.add.reduceat`` collapses each run to
    (min x0, max x1, band).
    """
    if rects.shape[0] < 2:
        return rects
    yb = np.round(rects[:, 2:4], 6)
    order = np.lexsort((rects[:, 0], yb[:, 1], yb[:, 0]))
    r = rects[order]
    yb = np.round(r[:, 2:4], 6)
    x0 = r[:, 0]
    x1 = r[:, 1]
    same_band = (yb[1:, 0] == yb[:-1, 0]) & (yb[1:, 1] == yb[:-1, 1])
    touch = np.abs(x0[1:] - x1[:-1]) < 1e-6
    new_run = ~(same_band & touch)
    starts = np.concatenate([[0], np.flatnonzero(new_run) + 1])
    # For each run [starts[i], starts[i+1]) collapse to (x0[start], max x1, band).
    ends = np.concatenate([starts[1:], [r.shape[0]]]) - 1
    out = np.empty((starts.size, 4), dtype=float)
    out[:, 0] = x0[starts]
    # max x1 within each run: since sorted by x0 and touching, x1 is monotone,
    # so x1 at the run END is the max.
    out[:, 1] = x1[ends]
    out[:, 2] = r[starts, 2]
    out[:, 3] = r[starts, 3]
    return out


def _zone_filled_rect(zone: np.ndarray) -> tuple[int, int, int, int] | None:
    """``(r0, r1, c0, c1)`` inclusive bounds when ``zone`` is a FILLED axis-aligned
    rectangle, else None (also None for an empty zone).

    Two 1-D ``any`` reductions for the bounds plus one ``sum`` against the bbox
    area — no per-cell index materialisation, so the check is cheap enough to run
    unconditionally before every angled clip.
    """
    rows = zone.any(axis=1)
    if not rows.any():
        return None
    cols = zone.any(axis=0)
    r0 = int(np.argmax(rows))
    r1 = int(rows.size - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(cols.size - 1 - np.argmax(cols[::-1]))
    if int(zone.sum()) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    return (r0, r1, c0, c1)


def _zone_bbox(
    zone: np.ndarray, pitch_um: float, extent_um: tuple[float, float]
) -> tuple[float, float, float, float]:
    """(x0,y0,x1,y1) µm bounding box of a zone mask (origin center, y up)."""
    ys, xs = np.nonzero(zone)
    h_px, w_px = zone.shape
    hx = w_px * pitch_um / 2.0
    hy = h_px * pitch_um / 2.0
    x0 = xs.min() * pitch_um - hx
    x1 = (xs.max() + 1) * pitch_um - hx
    y1 = hy - ys.min() * pitch_um
    y0 = hy - (ys.max() + 1) * pitch_um
    return (x0, y0, x1, y1)


def _concat_rects(parts: list[np.ndarray]) -> np.ndarray:
    parts = [p for p in parts if p is not None and p.shape[0] > 0]
    if not parts:
        return np.empty((0, 4), dtype=float)
    return np.concatenate(parts, axis=0)


def _drc_rects(rects: np.ndarray) -> np.ndarray:
    if rects.shape[0] == 0:
        return rects
    cleaned = drc_clean_rects(rects, min_width_um=LITHO_FLOOR_UM, min_gap_um=LITHO_FLOOR_UM)
    return np.asarray(cleaned, dtype=float).reshape(-1, 4)


# NOTE: drc_clean_polys (the slow morphological open) is intentionally NOT used
# by the fine writer — all emitted geometry is axis-aligned rects in its own
# frame, so _drc_rects handles it exactly and fast. The poly gate is imported
# for the integrator in case a future traced-silhouette layer needs it.


# =========================================================================== #
#  BSA fiducials (F4)                                                           #
# =========================================================================== #

def _cross_rects(cx: float, cy: float, arm_um: float, width_um: float) -> np.ndarray:
    """Solid plus-sign as two rects (H-arm + V-arm), (2,4) [x0,x1,y0,y1] µm."""
    ha = arm_um / 2.0
    hw = width_um / 2.0
    horiz = (cx - ha, cx + ha, cy - hw, cy + hw)
    vert = (cx - hw, cx + hw, cy - ha, cy + ha)
    return np.asarray([horiz, vert], dtype=float)


def _slot_pad_rects(cx: float, cy: float, pad_um: float, slot_um: float, arm_um: float) -> np.ndarray:
    """Complementary WINDOW target: a solid pad MINUS an open cross slot, as four
    rectangles (the pad split into quadrant bars around the cross slot). The
    front solid cross seats into the open slot when the layers overlay.
    """
    hp = pad_um / 2.0
    hs = slot_um / 2.0
    # Four solid bars: top, bottom, left-middle, right-middle around the +slot.
    # Vertical slot runs full height (width slot_um); horizontal slot full width.
    # The pad minus a plus-shaped slot = 4 corner rectangles.
    left_x = (cx - hp, cx - hs, cy - hp, cy + hp)
    right_x = (cx + hs, cx + hp, cy - hp, cy + hp)
    top_y = (cx - hs, cx + hs, cy + hs, cy + hp)
    bot_y = (cx - hs, cx + hs, cy - hp, cy - hs)
    return np.asarray([left_x, right_x, top_y, bot_y], dtype=float)


def _vernier_rects(cx: float, cy: float, pitch_um: float, n: int, line_um: float, length_um: float) -> np.ndarray:
    """A comb of ``n`` vertical rulings at ``pitch_um`` centered on (cx,cy)."""
    out = []
    total = (n - 1) * pitch_um
    x0 = cx - total / 2.0
    hy = length_um / 2.0
    for i in range(n):
        x = x0 + i * pitch_um
        out.append((x - line_um / 2.0, x + line_um / 2.0, cy - hy, cy + hy))
    return np.asarray(out, dtype=float)


def _group_congruent_rings(
    polys: list[np.ndarray], dbu_um: float
) -> list[tuple[np.ndarray, list[list[int]]]]:
    """Group rings by their DBU-quantized vertex list relative to their own bbox
    origin → ``[(rel_dbu, [[ox_dbu, oy_dbu], ...]), ...]``.

    Groups come out in FIRST-APPEARANCE order and each origin list in input order
    — the exact order the per-polygon dict build produced, so the GDS cell names
    and shape insertion order are unchanged.

    Bucketed by vertex COUNT first: rings with different vertex counts can never be
    congruent (their quantized vertex lists differ in length, which is what the old
    ``rel.tobytes()`` key encoded), so stacking each bucket into one ``(N, K, 2)``
    array is exact — and it turns the per-ring min/round/astype/tobytes round trip
    (~5 numpy calls × ~360k rings per box) into a handful of whole-array calls per
    bucket. Every value is produced by the same expression on the same operands, so
    the emitted DBU integers are identical.
    """
    by_count: dict[int, list[int]] = {}
    valid: list[np.ndarray] = []
    for verts in polys:
        v = np.asarray(verts, dtype=float)
        if v.ndim != 2 or v.shape[0] < 3 or v.shape[1] != 2:
            continue
        by_count.setdefault(v.shape[0], []).append(len(valid))
        valid.append(v)
    if not valid:
        return []
    # (first-appearance index, rel, origins) so buckets can be merged back into one
    # first-appearance-ordered sequence.
    found: list[tuple[int, np.ndarray, list[list[int]]]] = []
    for members in by_count.values():
        member_idx = np.asarray(members)
        stack = np.stack([valid[i] for i in members])            # (N, K, 2)
        origin = stack.min(axis=1)                               # (N, 2)
        rel = np.round((stack - origin[:, None, :]) / dbu_um).astype(np.int64)
        org = np.round(origin / dbu_um).astype(np.int64)          # (N, 2)
        flat = rel.reshape(rel.shape[0], -1)
        _, first, inv = np.unique(flat, axis=0, return_index=True, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        # Stable argsort blocks the members of each group together, ascending within
        # the block — so a block IS the group's origin list in input order.
        order = np.argsort(inv, kind="stable")
        lab = inv[order]
        b0 = np.flatnonzero(np.concatenate([[True], lab[1:] != lab[:-1]]))
        b1 = np.concatenate([b0[1:], [inv.size]])
        for s, e in zip(b0.tolist(), b1.tolist()):
            g = int(lab[s])
            block = order[s:e]
            found.append(
                (int(member_idx[first[g]]), rel[first[g]], org[block].tolist())
            )
    found.sort(key=lambda t: t[0])
    return [(rel, origins) for _, rel, origins in found]


def insert_polys_deduped(
    top,
    layer: int,
    polys: list[np.ndarray],
    *,
    dbu_um: float = DBU_UM,
    min_refs: int = 4,
    cell_prefix: str = "u",
) -> dict[str, int]:
    """Insert healed rings with GDS hierarchy instead of flat replication.

    A plate layer is mostly PERIODIC gold — a carrier grating is ~10^3 copies of
    ONE line, an interlace comb ~10^2 copies of one bar. Writing each copy as
    its own flat polygon replicates identical vertex lists thousands of times;
    GDS's native answer is a cell per unique shape referenced (SREF) at each
    placement, which fab tools expect for periodic masks and which shrinks both
    write time and file size by the repetition factor.

    Shapes are grouped by their vertex list relative to their bbox origin,
    quantized to the DBU grid; groups smaller than ``min_refs`` stay flat so
    one-off silhouette pieces don't pollute the cell table. Every emitted
    coordinate is ``origin_dbu + rel_dbu`` for BOTH flat and referenced forms,
    so instances of one shape are exactly congruent (placement rounding ≤ 1
    DBU = 1 nm, far below the 2 µm litho floor).

    Returns ``{"cells": ..., "refs": ..., "flat": ...}`` for stats/logging.
    """
    import klayout.db as kdb

    ly = top.layout()
    stats = {"cells": 0, "refs": 0, "flat": 0}
    for rel, origins in _group_congruent_rings(polys, dbu_um):
        # Python ints once per GROUP: the flat branch rebuilds the point list per
        # placement, and an np.int64 → int cast per vertex there is pure overhead.
        rel_pts = rel.tolist()
        if len(origins) < min_refs:
            for ox, oy in origins:
                pts = [kdb.Point(x + ox, y + oy) for x, y in rel_pts]
                top.shapes(layer).insert(kdb.Polygon(pts))
            stats["flat"] += len(origins)
            continue
        cell = ly.create_cell(f"{cell_prefix}{stats['cells']}")
        stats["cells"] += 1
        cell.shapes(layer).insert(
            kdb.Polygon([kdb.Point(x, y) for x, y in rel_pts])
        )
        idx = cell.cell_index()
        for ox, oy in origins:
            top.insert(kdb.CellInstArray(idx, kdb.Trans(kdb.Vector(ox, oy))))
        stats["refs"] += len(origins)
    return stats


def build_fiducials() -> dict[str, np.ndarray]:
    """Complementary BSA marks + verniers for both centers.

    Returns per-layer rect arrays keyed 'front', 'back', 'fid_front',
    'fid_back' (the dedicated fiducial datalayer copies). Coordinates are
    WAFER-centered µm (the fiducial centers are fixed at ∓30 mm on the x axis).
    """
    front: list[np.ndarray] = []
    back: list[np.ndarray] = []
    for (cx, cy) in FIDUCIAL_CENTERS_UM:
        # Front: solid cross.
        front.append(_cross_rects(cx, cy, FID_CROSS_ARM_UM, FID_CROSS_WIDTH_UM))
        # Back: complementary window target (pad minus cross slot).
        back.append(_slot_pad_rects(cx, cy, FID_PAD_UM, FID_SLOT_WIDTH_UM, FID_CROSS_ARM_UM))
        # Vernier pair beside each mark (front pitch A, back pitch B → 5 µm scale)
        vy = cy + FID_VERNIER_OFFSET_UM
        front.append(
            _vernier_rects(cx, vy, FID_VERNIER_PITCH_A_UM, FID_VERNIER_N, FID_VERNIER_LINE_UM, FID_VERNIER_LEN_UM)
        )
        back.append(
            _vernier_rects(cx, vy, FID_VERNIER_PITCH_B_UM, FID_VERNIER_N, FID_VERNIER_LINE_UM, FID_VERNIER_LEN_UM)
        )
    front_arr = np.concatenate(front, axis=0)
    back_arr = np.concatenate(back, axis=0)
    return {
        "front": front_arr,
        "back": back_arr,
        "fid_front": front_arr.copy(),
        "fid_back": back_arr.copy(),
    }


# =========================================================================== #
#  wafer assembly + GDS write                                                  #
# =========================================================================== #

def _plate_hits_fiducial(pl) -> bool:
    """True if plate footprint ``pl`` intersects EITHER r=2.5 mm keep-out disc."""
    for (fx, fy) in FIDUCIAL_CENTERS_UM:
        nx = min(max(fx, pl.x0), pl.x0 + pl.width_um)
        ny = min(max(fy, pl.y0), pl.y0 + pl.height_um)
        if (nx - fx) ** 2 + (ny - fy) ** 2 < FIDUCIAL_KEEPOUT_R_UM ** 2:
            return True
    return False


def repack_with_keepout(result: Any) -> tuple[Any, list[Any]]:
    """Repack so NO plate footprint overlaps either fiducial keep-out disc.

    The two BSA marks are pinned at (∓30 mm, 0) by the user constraint, and the
    max-scale mini box packs plates right across x=±29 mm at y=0 — so the default
    layout collides. The marks cannot move; instead we SHRINK the box (re-running
    ``solve_max_scale`` with a descending scale cap) until the tightest packing
    clears both discs, then re-derive the spec at that size. Keeps the 3 mm edge
    exclusion + 300 µm streets (both baked into ``solve_max_scale``'s packer).

    Returns ``(mini_result, placements)`` — the (possibly shrunk) MiniBoxResult
    and its collision-free placements.
    """
    from .export_wafer import solve_max_scale

    if not any(_plate_hits_fiducial(p) for p in result.placements):
        return result, result.placements

    # Bracket down from the current fitting width. solve_max_scale binary-searches
    # up to hi_um; capping hi_um below the colliding size forces a smaller box.
    base_w = result.width_um
    for frac in (0.94, 0.90, 0.86, 0.82, 0.78, 0.74, 0.70):
        hi = base_w * frac
        shrunk = solve_max_scale(hi_um=hi)
        if shrunk is None:
            continue
        if not any(_plate_hits_fiducial(p) for p in shrunk.placements):
            _log.info(
                "repacked: shrank mini box %.1f→%.1f mm to clear fiducial keep-outs",
                base_w / 1000.0, shrunk.width_um / 1000.0,
            )
            return shrunk, shrunk.placements
    _log.warning(
        "could not clear fiducial keep-outs by shrinking to 0.70×; "
        "keeping original packing (plates WILL overlap the marks)"
    )
    return result, result.placements


def _rects_to_gds_polys(rects: np.ndarray, dx: float, dy: float):
    """(N,4) [x0,x1,y0,y1] µm → list of klayout DPolygon at (dx,dy) offset."""
    import klayout.db as kdb

    polys = []
    for x0, x1, y0, y1 in rects:
        polys.append(
            kdb.DPolygon(
                [
                    kdb.DPoint(x0 + dx, y0 + dy),
                    kdb.DPoint(x1 + dx, y0 + dy),
                    kdb.DPoint(x1 + dx, y1 + dy),
                    kdb.DPoint(x0 + dx, y1 + dy),
                ]
            )
        )
    return polys


def _verts_to_gds_poly(verts: np.ndarray, dx: float, dy: float):
    import klayout.db as kdb

    return kdb.DPolygon([kdb.DPoint(float(x) + dx, float(y) + dy) for x, y in verts])


def mini_spec_real_faces(result: Any) -> Any:
    """Mini-box spec carrying the REAL six-face plan (F1), not one flat slug.

    ``export_wafer.mini_box_spec`` stamps ``DEFAULT_FACE_PATTERN_SLUG`` on ALL
    six faces (fine for the coarse layout preview), but the fab wafer must route
    each face's true pattern: front colibrí-globe, back capybara-scanimation,
    left food-pair-chirp, right gear-quill-switch, top monogram-jp, bottom
    inscription-line. Rather than duplicate that slug table, we take
    ``boxes.default_box_spec()`` — the single source of truth for per-face slug +
    frame profile — and RESIZE it to the mini dims (glass + smallest foil so the
    shrunk plates keep a valid keep-out rim), then re-normalize the cut dims. The
    "optical" period params (native 22/24/60/4.4 µm) come from
    ``_carrier_recipe_data`` regardless, so no per-face pattern_params are needed.
    """
    from .assembly import FoilSpec
    from .boxes import default_box_spec

    spec = default_box_spec()  # real per-face slugs + frame seeds/profiles
    spec.width_um = result.width_um
    spec.depth_um = result.depth_um
    spec.height_um = result.height_um
    # Smallest foil preset so the keep-out rim stays inside the shrunk plates.
    spec.foil = FoilSpec(tape_width_um=result.tape_width_um)
    spec.normalize_face_dims()
    return spec


def build_wafer_fine_gds(
    out_path: Path,
    *,
    rotate_plate_geom: bool = True,
    drc_report_path: Path | None = None,
) -> dict[str, Any]:
    """Full fine-pitch wafer: per-face native geometry + BSA fiducials → GDS.

    Reads the six-face plan from ``boxes.default_box_spec`` (the single source of
    truth for face→slug routing) resized to the mini box — see
    :func:`mini_spec_real_faces`. Streams one plate-layer at a time into the
    klayout layout to keep the in-RAM polygon set bounded.

    If ``drc_report_path`` is given, the per-face-layer DRC before/after reports
    (captured inside :func:`build_plate_fine`) are written there as JSON.
    """
    import klayout.db as kdb

    from .assembly import FACE_IDS
    from .export_wafer import (
        USABLE_RADIUS_UM,
        solve_max_scale,
        _circle_pts,
    )

    result = solve_max_scale()
    if result is None:
        raise SystemExit("no box fits the wafer")
    # Repack (possibly shrink) so no plate overlaps a fiducial keep-out disc,
    # THEN derive the face spec at the final size so cut dims match placements.
    result, placements = repack_with_keepout(result)
    spec = mini_spec_real_faces(result)  # REAL per-face plan (F1), native periods
    by_face = {p.face: p for p in placements}

    ly = kdb.Layout()
    ly.dbu = DBU_UM
    top = ly.create_cell("wafer_fine")
    L_front = ly.layer(*LAYER_FRONT)
    L_back = ly.layer(*LAYER_BACK)
    L_outline = ly.layer(*LAYER_OUTLINE)
    L_wafer = ly.layer(*LAYER_WAFER)
    L_label = ly.layer(*LAYER_LABEL)
    L_fid_f = ly.layer(*LAYER_FIDUCIAL_FRONT)
    L_fid_b = ly.layer(*LAYER_FIDUCIAL_BACK)

    # Wafer outline.
    top.shapes(L_wafer).insert(
        kdb.DPolygon([kdb.DPoint(x, y) for x, y in _circle_pts(USABLE_RADIUS_UM)])
    )

    report: list[dict[str, Any]] = []
    total_polys = 0

    for fid in FACE_IDS:
        p = by_face.get(fid)
        plate_spec = spec.faces.get(fid)
        if p is None or plate_spec is None:
            continue
        # CLI wafer path keeps the full before/after audit blocks.
        fine = build_plate_fine(plate_spec, fid, drc_before_report=True)
        # Placement rotation (packer may 90°-rotate a plate to fit).
        rot = p.rotated and rotate_plate_geom
        dx, dy = p.cx, p.cy

        def _emit_polys(polys: list[np.ndarray], layer: int) -> int:
            """Emit healed plate-frame polygon rings (blocker-3 merged-DRC output).
            Applies the packer's 90° rotation, if any, then translates to the
            plate center. This is the FINAL geometry — the merged Region already
            unioned + healed it, so nothing further is booleaned here."""
            nonlocal total_polys
            n = 0
            ca, sa = (0.0, 1.0) if rot else (1.0, 0.0)  # +90° CCW: (x,y)->(-y,x)
            for verts in polys:
                if verts.shape[0] < 3:
                    continue
                if rot:
                    rx = -verts[:, 1]
                    ry = verts[:, 0]
                    v = np.stack([rx, ry], axis=1)
                else:
                    v = verts
                top.shapes(layer).insert(_verts_to_gds_poly(v, dx, dy))
                n += 1
            total_polys += n
            return n

        nf = _emit_polys(fine.front_polys, L_front)
        nb = _emit_polys(fine.back_polys, L_back)

        # Plate outline.
        hw, hh = p.width_um / 2.0, p.height_um / 2.0
        top.shapes(L_outline).insert(
            kdb.DBox(p.cx - hw, p.cy - hh, p.cx + hw, p.cy + hh)
        )
        top.shapes(L_label).insert(kdb.DText(fid, kdb.DTrans(kdb.DVector(p.cx, p.cy))))
        report.append({"face": fid, "slug": fine.slug, "front": nf, "back": nb, **fine.stats})

    # --- BSA fiducials ------------------------------------------------------
    fids = build_fiducials()
    for poly in _rects_to_gds_polys(fids["front"], 0.0, 0.0):
        top.shapes(L_front).insert(poly)
    for poly in _rects_to_gds_polys(fids["back"], 0.0, 0.0):
        top.shapes(L_back).insert(poly)
    for poly in _rects_to_gds_polys(fids["fid_front"], 0.0, 0.0):
        top.shapes(L_fid_f).insert(poly)
    for poly in _rects_to_gds_polys(fids["fid_back"], 0.0, 0.0):
        top.shapes(L_fid_b).insert(poly)
    total_polys += (
        fids["front"].shape[0] + fids["back"].shape[0]
        + fids["fid_front"].shape[0] + fids["fid_back"].shape[0]
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(out_path))

    # --- per-face-layer DRC report → JSON ----------------------------------
    # Aggregate over the MERGED (printed) reports — the honest per-blocker-3 tally
    # of what the emitted geometry actually contains — for a headline; keep the
    # full per-face before/after blocks (per-group AND merged) for auditing.
    drc_faces: dict[str, Any] = {}
    global_min_width_after = float("inf")
    global_min_space_after = float("inf")
    n_width_viol_after = 0
    n_space_viol_after = 0
    for r in report:
        d = r.get("drc")
        if not d:
            continue
        drc_faces[r["face"]] = d
        for key in ("front_merged_after", "back_merged_after"):
            blk = d.get(key, {})
            global_min_width_after = min(
                global_min_width_after, float(blk.get("min_width_um", float("inf")))
            )
            global_min_space_after = min(
                global_min_space_after, float(blk.get("min_space_um", float("inf")))
            )
            n_width_viol_after += int(blk.get("n_width_viol", 0))
            n_space_viol_after += int(blk.get("n_space_viol", 0))
    drc_summary = {
        "drc_available": _DRC_AVAILABLE,
        "litho_floor_um": LITHO_FLOOR_UM,
        "merged_geometry": True,
        "global_min_width_um_after": (
            None if math.isinf(global_min_width_after) else global_min_width_after
        ),
        "global_min_space_um_after": (
            None if math.isinf(global_min_space_after) else global_min_space_after
        ),
        "total_width_viol_after": n_width_viol_after,
        "total_space_viol_after": n_space_viol_after,
        "total_subfloor_after": n_width_viol_after + n_space_viol_after,
        "per_face": drc_faces,
    }
    if drc_report_path is not None:
        import json

        drc_report_path = Path(drc_report_path)
        drc_report_path.parent.mkdir(parents=True, exist_ok=True)
        drc_report_path.write_text(json.dumps(drc_summary, indent=2, default=float), encoding="utf-8")
        _log.info("wrote DRC report → %s", drc_report_path)

    return {
        "gds_path": str(out_path),
        "total_polygons": total_polys,
        "drc_available": _DRC_AVAILABLE,
        "drc_report_path": (str(drc_report_path) if drc_report_path is not None else None),
        "drc_summary": drc_summary,
        "fiducial_centers_mm": [(c[0] / 1000.0, c[1] / 1000.0) for c in FIDUCIAL_CENTERS_UM],
        "layers": {
            "front": LAYER_FRONT, "back": LAYER_BACK, "outline": LAYER_OUTLINE,
            "wafer": LAYER_WAFER, "fiducial_front": LAYER_FIDUCIAL_FRONT,
            "fiducial_back": LAYER_FIDUCIAL_BACK,
        },
        "plates": report,
    }


# =========================================================================== #
#  TEST CELL — one grating of each type in a 5×5 mm cell, measured             #
# =========================================================================== #

def build_test_cell(out_path: Path) -> dict[str, Any]:
    """Write a 5×5 mm cell with one instance of each grating type at its NATIVE
    period, then read it back and measure the realized periods. Cheap validation
    (one short run) that the writer produces exact geometry — NOT the full wafer.
    """
    import klayout.db as kdb

    W = H = 5000.0  # 5 mm
    extent = (W, H)
    pitch = 1.0     # 1 µm boundary raster → resolves 4.4 µm accent (F2 native)

    ly = kdb.Layout()
    ly.dbu = DBU_UM
    top = ly.create_cell("test_cell")
    L_front = ly.layer(*LAYER_FRONT)
    L_back = ly.layer(*LAYER_BACK)

    # Four quadrant zones, each a full-True rectangle mask so the grating fills it.
    def _rect_zone(x0, y0, x1, y1):
        fw = int(round(W / pitch)); fh = int(round(H / pitch))
        m = np.zeros((fh, fw), dtype=bool)
        hx = fw * pitch / 2.0; hy = fh * pitch / 2.0
        c0 = int((x0 + hx) / pitch); c1 = int((x1 + hx) / pitch)
        r0 = int((hy - y1) / pitch); r1 = int((hy - y0) / pitch)
        m[r0:r1, c0:c1] = True
        return m

    measured: dict[str, Any] = {}

    # Q1 (top-left): back carrier 22 µm vertical → back layer.
    z = _rect_zone(-2400, 200, -200, 2400)
    rects = _clip_axis_grating(z, pitch, extent, 22.0, 0.5, 0.0)
    for poly in _rects_to_gds_polys(rects, 0, 0):
        top.shapes(L_back).insert(poly)
    measured["back_carrier_22"] = {"target_um": 22.0, "n_lines": rects.shape[0]}

    # Q2 (top-right): front leaf 23.98 µm at +3° → front layer (angled, rotated
    # from grating-local rects at placement).
    z = _rect_zone(200, 200, 2400, 2400)
    local = _angled_grating_local_rects(z, pitch, extent, 23.98, 0.5, 3.0)
    verts = _rotate_rects_to_polys(local, 3.0)
    for i in range(verts.shape[0]):
        top.shapes(L_front).insert(_verts_to_gds_poly(verts[i], 0, 0))
    measured["front_leaf_23p98_off3"] = {"target_um": 23.98, "angle_deg": 3.0, "n_polys": int(verts.shape[0])}

    # Q3 (bottom-left): centerpiece switch 60 µm, front phase 0 + back phase 0.5.
    z = _rect_zone(-2400, -2400, -200, -200)
    rf = _clip_axis_grating(z, pitch, extent, 60.0, 0.5, 0.0, phase=0.0)
    rb = _clip_axis_grating(z, pitch, extent, 60.0, 0.5, 0.0, phase=0.5)
    for poly in _rects_to_gds_polys(rf, 0, 0):
        top.shapes(L_front).insert(poly)
    for poly in _rects_to_gds_polys(rb, 0, 0):
        top.shapes(L_back).insert(poly)
    measured["center_switch_60"] = {
        "target_um": 60.0, "front_lines": rf.shape[0], "back_lines": rb.shape[0],
        "half_period_offset_um": 30.0,
    }

    # Q4 (bottom-right): rainbow accent 4.4 µm at 45° → front layer.
    z = _rect_zone(200, -2400, 2400, -200)
    local = _angled_grating_local_rects(z, pitch, extent, 4.4, 0.5, 45.0)
    verts = _rotate_rects_to_polys(local, 45.0)
    for i in range(verts.shape[0]):
        top.shapes(L_front).insert(_verts_to_gds_poly(verts[i], 0, 0))
    measured["rainbow_accent_4p4_45"] = {"target_um": 4.4, "angle_deg": 45.0, "n_polys": int(verts.shape[0])}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(out_path))

    # --- read back + measure -----------------------------------------------
    ly2 = kdb.Layout()
    ly2.read(str(out_path))
    c = ly2.top_cell()
    dbu = ly2.dbu

    def _measure_period_axis(layer_ld, x_band, y_band, axis="x"):
        """Measure the dominant line pitch on a layer within a band by collecting
        distinct edge coordinates along ``axis``."""
        lyr = ly2.layer(*layer_ld)
        coords = []
        widths = []
        for s in c.shapes(lyr).each():
            # Shapes are stored as polygons; get their integer bbox → µm.
            ib = s.bbox()  # integer Box in dbu
            b = ib.to_dtype(dbu)  # DBox in µm
            xc = (b.left + b.right) / 2.0
            yc = (b.bottom + b.top) / 2.0
            if x_band[0] <= xc <= x_band[1] and y_band[0] <= yc <= y_band[1]:
                if axis == "x":
                    coords.append(b.left)
                    widths.append(b.width())
                else:
                    coords.append(b.bottom)
                    widths.append(b.height())
        coords = np.sort(np.unique(np.round(coords, 3)))
        if coords.size < 2:
            return None
        diffs = np.diff(coords)
        # The period is the median spacing between adjacent line starts.
        return float(np.median(diffs)), float(np.median(widths)) if widths else None

    # Back carrier 22 µm (Q1, vertical lines → x pitch).
    m = _measure_period_axis(LAYER_BACK, (-2400, -200), (200, 2400), "x")
    if m:
        measured["back_carrier_22"]["measured_period_um"] = round(m[0], 3)
        measured["back_carrier_22"]["measured_line_um"] = round(m[1], 3)
    # Center switch front phase 0 (Q3, x pitch).
    m = _measure_period_axis(LAYER_FRONT, (-2400, -200), (-2400, -200), "x")
    if m:
        measured["center_switch_60"]["measured_period_um"] = round(m[0], 3)
    # Center switch back phase 0.5 — measure offset vs front.
    mb = _measure_period_axis(LAYER_BACK, (-2400, -200), (-2400, -200), "x")
    if mb and m:
        measured["center_switch_60"]["measured_back_period_um"] = round(mb[0], 3)

    return {
        "gds_path": str(out_path),
        "cell_um": (W, H),
        "boundary_pitch_um": pitch,
        "drc_available": _DRC_AVAILABLE,
        "measured": measured,
    }


# =========================================================================== #
#  CLI                                                                          #
# =========================================================================== #

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Fine-pitch wafer GDS writer (native periods).")
    ap.add_argument("--out", default="data/wafer/wafer_fine.gds", help="output GDS path")
    ap.add_argument("--drc-report", default=None, help="path to write the per-face DRC report JSON")
    ap.add_argument("--test-cell", action="store_true", help="write + measure a 5x5 mm test cell only")
    args = ap.parse_args(argv)

    if args.test_cell:
        summary = build_test_cell(Path(args.out))
        print("TEST CELL:", summary["gds_path"])
        print("boundary pitch:", summary["boundary_pitch_um"], "µm  drc:", summary["drc_available"])
        for k, v in summary["measured"].items():
            print(f"  {k}: {v}")
        return 0

    drc_path = Path(args.drc_report) if args.drc_report else None
    summary = build_wafer_fine_gds(Path(args.out), drc_report_path=drc_path)
    print("WAFER FINE:", summary["gds_path"])
    print("total polygons:", summary["total_polygons"], " drc:", summary["drc_available"])
    print("fiducials (mm):", summary["fiducial_centers_mm"])
    ds = summary["drc_summary"]
    print(f"DRC (merged): min_width_after={ds['global_min_width_um_after']} "
          f"min_space_after={ds['global_min_space_um_after']} "
          f"width_viol={ds['total_width_viol_after']} space_viol={ds['total_space_viol_after']} "
          f"report={summary['drc_report_path']}")
    for r in summary["plates"]:
        print(f"  {r['face']:7s} {r['slug']:22s} front={r['front']:6d} back={r['back']:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
