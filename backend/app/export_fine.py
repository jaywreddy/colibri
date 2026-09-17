"""Fine-pitch mask geometry — TRUE optical periods, per-face dispatch.

The raster path (``plates.materialize_plate`` + ``ensure_plate_svg``) rasters
every layer at a *budget pitch* and, when the ideal 4-samples/period pitch would
blow the 400k-cell cap, RESCALES the grating periods to fit. That is fine for a
screen preview and fatal for a mask: a 4.4 µm colour sub-grating aliases into
noise long before the cap is reached.

This module is the fab-grade alternative, and the only geometry that reaches the
plate. It writes a face's gold as TRUE vector geometry at the design periods, by
generating each grating as RECTANGLES in its own grating-local frame (where lines
are axis-aligned and one line is ONE full-span rectangle, so the polygon count
scales with LINE count, not pixel count), clipping each line to its zone by a
numpy row-span pass against a freshly-regenerated SOURCE mask, and rotating the
whole set to placement as polygons only when the grating is angled.

Only the zone *boundaries* (which pixels are frame / centrepiece / art) are
quantized to the source-mask raster (~20-30 µm silhouette-edge steps, which are
unavoidable and visually irrelevant); the PERIODS inside every zone are exact
vector geometry.

Per-face dispatch reads the six-face plan from ``boxes.default_box_spec`` — one
source of truth, no duplicated slug table.

:func:`build_plate_fine` is the entry point. Its caller is
``witness_dies.build_face_die``, which inverts the result to the clear-field
polarity the plate is written in; the deliverable itself is written by
``app.export_witness`` (``uv run python -m app.export_witness``).

All lengths µm unless a ``_mm`` suffix says otherwise.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .patterns.effects.gratings import band_select

_log = logging.getLogger("optics.export_fine")

# The process constants live in ``app.production`` — one definition each, read
# by the generators, both mask writers and the die inverter.
from .production import DBU_UM, FINISH_RADIUS_UM, LITHO_FLOOR_UM  # noqa: E402


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
    # Full centerpiece art box (the CENTERPIECE_FILL square, weld-rim clipped).
    # The barrier-interlace front comb spans this WHOLE square — never the
    # silhouette union (a union-gated comb is itself a static front image).
    art_box: np.ndarray | None = None
    # SINGLE-LAYER DIFFRACTION centrepiece (region_art): int32 label raster on
    # the plate grid (0 = glass) and the Region each label is written as. When
    # set, ``front_art`` / ``back_art`` are EMPTY — the regions ARE the
    # centrepiece and no carrier, comb or switch is emitted for it.
    art_regions: np.ndarray | None = None
    region_specs: dict[int, Any] | None = None


def _build_zone_masks(spec: Any, pitch_um: float) -> ZoneMasks:
    """Regenerate a plate's zone masks at ``pitch_um`` from the SOURCE motifs.

    Reuses plates.py's own helpers (`frame_scene_for_plate` →
    `render_scene_to_image` for the foliage band graylevels — the composed
    plate's own scene, not a regrown one — and `region_art.centerpiece_regions` /
    `_centerpiece_masks` for the centrepiece). Boundary quantization only —
    periods are added later as vector geometry.
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
    art_box = np.zeros((fh, fw), dtype=bool)
    art_regions: np.ndarray | None = None
    region_specs: dict[int, Any] | None = None

    if spec.kind in (P.FaceKind.BLANK, P.FaceKind.SOLID):
        # BARE GLASS (and SOLID GOLD, whose one rectangle build_plate_fine adds
        # itself): every zone stays empty, so every ``.any()`` gate in
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

    # --- CENTERPIECE silhouettes -------------------------------------------
    aperture = P._aperture(spec)
    side_px = max(8, int(round(P.CENTERPIECE_FILL * aperture / pitch_um))) if aperture > 0 else 0
    if side_px > 0:
        # Full art-box square (the CENTERPIECE_FILL footprint) — the zone the
        # barrier-interlace front comb spans, independent of the silhouettes.
        ax0 = max(0, fw // 2 - side_px // 2)
        ay0 = max(0, fh // 2 - side_px // 2)
        art_box[ay0 : min(fh, ay0 + side_px), ax0 : min(fw, ax0 + side_px)] = True
        _mask_rim(art_box, spec.weld_margin_um, pitch_um)
        from . import region_art as RA

        ra = (RA.centerpiece_regions(spec.pattern_slug, side_px, spec.pattern_params)
              if spec.kind is P.FaceKind.REGION else None)
        if ra is not None:
            # SINGLE-LAYER DIFFRACTION centrepiece: the label map goes on the
            # grid at the art box; the silhouette masks stay empty so nothing
            # below fills them with a carrier.
            lab = np.zeros((fh, fw), dtype=np.int32)
            x0 = fw // 2 - side_px // 2
            y0 = fh // 2 - side_px // 2
            xa, ya = max(0, x0), max(0, y0)
            xb, yb = min(fw, x0 + side_px), min(fh, y0 + side_px)
            lab[ya:yb, xa:xb] = ra.labels[ya - y0 : yb - y0, xa - x0 : xb - x0]
            rim = np.ones((fh, fw), dtype=bool)
            _mask_rim(rim, spec.weld_margin_um, pitch_um)
            lab[~rim] = 0
            art_regions, region_specs = lab, dict(ra.regions)
            masks = None
        else:
            art_regions, region_specs = None, None
            masks = P._centerpiece_masks(spec, side_px)
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

            front_art, _front_side = _place(masks[0])
            back_art, _ = _place(masks[1])
            _mask_rim(front_art, spec.weld_margin_um, pitch_um)

    return ZoneMasks(
        pitch_um=pitch_um,
        extent_um=(W, H),
        frame=frame,
        frame_level=frame_level,
        back_window=back_window,
        front_art=front_art,
        back_art=back_art,
        art_box=art_box,
        art_regions=art_regions,
        region_specs=region_specs,
    )


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


def _dilate_zone(mask: np.ndarray, cells: int = 1) -> np.ndarray:
    """Grow a bool zone by ``cells`` grid cells (4-neighbour) — the complement
    of :func:`_erode_zone`, used to cut a gutter into a NEIGHBOUR."""
    if cells <= 0 or not mask.any():
        return mask
    m = mask
    for _ in range(cells):
        d = m.copy()
        d[1:, :] |= m[:-1, :]
        d[:-1, :] |= m[1:, :]
        d[:, 1:] |= m[:, :-1]
        d[:, :-1] |= m[:, 1:]
        m = d
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
    rectangles in the PLATE frame (vertical gratings, photo bands, region
    fills) — cleaned by the fast ``drc_clean_rects``.

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


def zone_pitch_um(spec: Any, back_period_um: float | None = None) -> float:
    """The zone-BOUNDARY raster pitch of a plate's fine bake: fine enough to
    resolve the finest zone edge (the carrier), capped by the lattice budget so
    a big plate keeps a small source raster. One function, so
    ``build_plate_fine`` and :func:`single_layer_region_rects` (the fab SVG's
    copy of the centrepiece) quantise the same boundaries.

    The ideal term used to take the MINIMUM of the carrier and a scanimation
    frame slot (``water_scan_fab_pitch_um / N``), which went with the two-ply
    optics on 2026-09-16. On every face of the production box the budget term
    binds anyway — 48-53 um against a ~16 um ideal — so dropping the slot leaves
    the pitch, and the mask, unchanged."""
    from . import plates as P
    from .patterns._helpers import MAX_LATTICE_CELLS

    if back_period_um is None:
        back_period_um = float(P._carrier_recipe_data(spec)["fab_back_period_um"])
    W, H = spec.width_um, spec.height_um
    ideal = back_period_um / 4.0
    budget_pitch = math.sqrt(W * H / (0.9 * MAX_LATTICE_CELLS))
    return max(ideal, budget_pitch)


def _zone_solid_rects(zone: np.ndarray, pitch_um: float, extent_um: tuple[float, float]) -> np.ndarray:
    """SOLID gold over a zone: one plate-frame rect per contiguous run of set
    cells in each row (rows merge later in the metal heal). Plate-centred µm,
    row 0 at the top, y up — the ``ZoneMasks`` convention."""
    if not zone.any():
        return np.empty((0, 4), dtype=np.float64)
    h_px, w_px = zone.shape
    W, H = extent_um
    hx, hy = W / 2.0, H / 2.0
    out: list[np.ndarray] = []
    for r in np.flatnonzero(zone.any(axis=1)):
        row = zone[r]
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        y1 = hy - r * pitch_um
        y0 = y1 - pitch_um
        out.append(np.stack([starts * pitch_um - hx, ends * pitch_um - hx,
                             np.full(starts.size, y0), np.full(starts.size, y1)], axis=1))
    return np.concatenate(out, axis=0) if out else np.empty((0, 4), dtype=np.float64)


def check_region_periods(regions: dict[int, Any]) -> None:
    """Every diffractive region must survive the litho floor AND the die
    finish (the guard the single-ply leaf families pass): line = p·duty must
    exceed 2 × the finish radius and both line and gap must clear the floor."""
    finish_min_line = 2.0 * FINISH_RADIUS_UM
    for rid, r in regions.items():
        if r.period_um <= 0:
            continue
        line = r.period_um * r.duty
        gap = r.period_um - line
        if line <= finish_min_line + 1e-9:
            raise ValueError(
                f"region {rid} {r.name!r}: {r.period_um} um at duty {r.duty} writes {line:.3f} um "
                f"lines, which the {finish_min_line / 2:.2f} um die finish (open) erases")
        if min(line, gap) < LITHO_FLOOR_UM - 1e-9:
            raise ValueError(
                f"region {rid} {r.name!r}: {r.period_um} um at duty {r.duty} breaks the "
                f"{LITHO_FLOOR_UM} um litho floor (line {line:.3f}, gap {gap:.3f})")


def _emit_region_art(spec: Any, rects_out: list[np.ndarray]) -> dict[str, Any]:
    """Write a single-layer diffraction centrepiece: every region of the
    motif's map as its own VERTICAL grating at its period (solid gold where the
    period is 0), each inset one cell so no two gratings meet.

    Rastered on its OWN grid over the art box at ``region_art.REGION_ZONE_PITCH_UM``,
    not on the plate's zone grid: the plate grid is budget-capped at ~50 µm on
    a 32 mm face, and a region edge (a letter's outline, a coastline) quantised
    to 50 µm steps, with a 50 µm glass gutter between neighbouring regions, is
    at the eye's limit at 300 mm. At 20 µm both are a quarter of it. The art
    box is centred on the plate, so the rects need no shift.

    Appends plate-frame rects to ``rects_out`` and returns the stats block."""
    from . import plates as P
    from . import region_art as RA

    side = P.CENTERPIECE_FILL * P._aperture(spec)
    if side <= 0:
        return {"n_regions": 0, "cells": 0, "regions": {}, "pitch_um": 0.0}
    n = int(round(side / RA.REGION_ZONE_PITCH_UM))
    n = max(16, min(n, RA.REGION_ZONE_MAX_PX))
    pitch = side / n
    ra = RA.centerpiece_regions(spec.pattern_slug, n, spec.pattern_params)
    if ra is None:
        return {"n_regions": 0, "cells": 0, "regions": {}, "pitch_um": pitch}
    check_region_periods(ra.regions)
    extent = (side, side)
    n_cells = 0
    per_region: dict[str, Any] = {}
    metal_all = ra.labels > 0
    for rid, reg in sorted(ra.regions.items()):
        zone = ra.labels == rid
        if not zone.any():
            continue
        # Seam gutter between regions of different periods (and between a
        # grating and solid gold): one cell of glass wherever this region
        # touches ANOTHER metal region — never at its edge against bare glass,
        # so a thin solid feature (a graticule line, a star) keeps its drawn
        # width. A region that vanishes under the gutter was thinner than the
        # zone pitch where it met its neighbour.
        others = metal_all & ~zone
        zone_in = zone & ~_dilate_zone(others, 1)
        if not zone_in.any():
            per_region[reg.name] = {"period_um": reg.period_um, "cells": 0,
                                    "note": "thinner than one zone cell after the gutter; nothing written"}
            continue
        n_before = len(rects_out)
        if reg.solid:
            rects_out.append(_zone_solid_rects(zone_in, pitch, extent))
        else:
            _emit_grating(zone_in, pitch, extent, reg.period_um, reg.duty, 0.0, 0.0,
                          rects_out, [])
        n_new = sum(int(r.shape[0]) for r in rects_out[n_before:])
        per_region[reg.name] = {"period_um": float(reg.period_um), "duty": float(reg.duty),
                                "cells": int(zone_in.sum()), "rects": n_new}
        n_cells += int(zone_in.sum())
    return {"n_regions": len(per_region), "cells": n_cells, "regions": per_region,
            "pitch_um": float(pitch), "art_box_um": float(side)}


def single_layer_region_rects(spec: Any) -> np.ndarray:
    """The EXACT front-layer rects of a single-layer diffraction centrepiece,
    ``(N, 4)`` ``[x0, x1, y0, y1]`` µm plate-centred — the same geometry
    ``build_plate_fine`` writes, for the fab SVG bake to concatenate (the
    2-6 µm gratings are far below the SVG's raster pitch). Empty when the face
    has no region centrepiece."""
    from . import plates as P

    if spec.kind is not P.FaceKind.REGION:
        return np.empty((0, 4), dtype=np.float64)
    parts: list[np.ndarray] = []
    _emit_region_art(spec, parts)
    return _drc_rects(_concat_rects(parts))


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
    kind = spec.kind
    # WHICH exemplar, inside FaceKind.TWO_PLY — see plates._carrier_recipe_data.
    is_interlace = spec.pattern_slug in P.SWITCH_INTERLACE_SLUGS
    # PHOTOGRAPH: the centerpiece is a line screen, emitted as exact rectangles
    # from ``plates.photo_band_rects`` (front layer only) rather than a
    # silhouette filled with the switch carrier.
    is_photo = kind is P.FaceKind.PHOTO
    # SINGLE PLY: no inner ply exists, so the uniform carrier joins the leaves on
    # the FRONT layer and the back layer stays empty.
    single_ply = bool(getattr(spec, "single_ply", False))

    W, H = spec.width_um, spec.height_um

    # Zone-boundary raster pitch. Fine ENOUGH to resolve the finest zone edge
    # feature, but this is BOUNDARY only — the periods are exact vector geometry
    # regardless. Cap the grid so the source masks stay small (well under the
    # lattice budget), floored so a huge plate does not explode the source
    # raster. See :func:`zone_pitch_um`.
    pitch = zone_pitch_um(spec, back_period)

    zm = _build_zone_masks(spec, pitch)
    extent = (W, H)

    front_rects_parts: list[np.ndarray] = []
    back_rects_parts: list[np.ndarray] = []
    front_angled: list[tuple[np.ndarray, float]] = []
    back_angled: list[tuple[np.ndarray, float]] = []
    stats: dict[str, Any] = {"pitch_um": pitch}

    if kind is P.FaceKind.SOLID:
        # SOLID GOLD (the base plate): the whole outer ply, one rectangle, no
        # rim — the gold runs under the foil. Every zone mask is empty, so
        # nothing else below emits.
        front_rects_parts.append(np.array([[-W / 2.0, W / 2.0, -H / 2.0, H / 2.0]], dtype=np.float64))
        stats["solid"] = True

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
        # every family's finest LINE must survive the die finish: an open of
        # radius r (production.FINISH_RADIUS_UM) deletes lines under 2r, and the
        # written line is p * duty. Probe ALL buckets, not the first.
        probe = min([p for b in range(frame_count)
                     for p, _ in LF.bucket_layers(fill, b, frame_count, 0.0, leaf_period, hue_periods)]
                    or [leaf_period])
        finish_min_line = 2.0 * FINISH_RADIUS_UM
        if probe * duty <= finish_min_line + 1e-9:
            raise ValueError(
                f"single-ply leaf fill {fill!r}: a {probe} um period at duty {duty} writes "
                f"{probe * duty:.3f} um lines, which the {FINISH_RADIUS_UM} um "
                f"die finish (open) erases (needs > {finish_min_line:.3f} um)")
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
    if kind is P.FaceKind.REGION and zm.art_regions is not None:
        # SINGLE-LAYER DIFFRACTION centrepiece (region_art): each region of the
        # motif is its own vertical grating at its own period — colour by
        # region, one ply, no carrier — or solid gold where the period is 0.
        # Same guard the leaf families pass (finish + litho floor).
        stats["single_layer_regions"] = _emit_region_art(spec, front_rects_parts)
    elif is_photo:
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
        comb_zone = _erode_zone(comb_box, 1)
        if comb_zone.any():
            _emit_grating(
                comb_zone, pitch, extent, center_period, 0.5, center_axis,
                -0.25, front_rects_parts, front_angled,
            )
    elif zm.front_art.any():
        # Inset 1 cell so the 0° switch carrier does not abut the frame band's
        # angled leaf gratings at the silhouette boundary (a different-angle seam
        # would cross into sub-floor wedges).
        art_zone = _erode_zone(zm.front_art, 1)
        if art_zone.any():
            _emit_grating(
                art_zone, pitch, extent, center_period, duty, center_axis,
                0.0, front_rects_parts, front_angled,
            )

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
        # phase-π switch carrier).
        if is_interlace:
            # Both silhouettes live on the back as interleaved lanes → clear the
            # plain carrier across their whole union so the lanes read.
            carrier_zone = zm.back_window & ~(zm.front_art | zm.back_art)
        else:
            carrier_zone = zm.back_window & ~zm.back_art
        # Seam gutter: the carrier (base_angle) abuts the 0° switch art at a
        # different-angle boundary. Inset it 2 cells, not 1: the carrier is an
        # ANGLED grating and its rotated-rectangle lines end in a SLANTED tip. A
        # 1-cell gutter leaves that tip descending to ~2 µm past the zone edge and
        # raking toward the abutting field, closing the seam to a sub-floor gap
        # (measured: 2 gaps of 1.085 µm where a 165° carrier tip neared the 0°
        # switch carrier). A 2-cell inset pulls the whole slanted tip clear,
        # opening a ≥2 µm gutter so no carrier line approaches the neighbour field
        # within the floor. The extra cell is one ~39 µm boundary-raster step off a
        # rim already inset by the weld margin — optically invisible, and it only
        # shrinks gold so it cannot add defects.
        carrier_zone = _erode_zone(carrier_zone, 2)
        # base_angle is a per-face carrier rotation (seed-keyed); non-zero →
        # grating-local rects + angle, else exact axis rects.
        if carrier_zone.any():
            _emit_grating(
                carrier_zone, pitch, extent, back_period, duty, base_angle,
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
    elif zm.back_art.any():
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
    into the plate's cached fine geometry, so that is a version-bumping change,
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


