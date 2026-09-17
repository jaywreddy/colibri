"""The FAB SVG: the same aperture-scaled geometry the preview PNG composes.

``ensure_plate_svg`` builds the pair lazily on the first export request — the
polygon/SVG path is ~40x slower than the raster one and the interactive UI
never needs it. It MUST compose what ``compose._raster_compose_plate`` does;
``PLATE_SVG_FINGERPRINT`` (below) is what invalidates a stale cached SVG when this
module's geometry changes.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon

from ..cache_fingerprint import PHOTO_ASSETS, PLATE_SVG_CLOSURE, fingerprint
from ..export_svg import to_svg
from ..patterns.frames import RectFrame, render_scene_to_image
from ..service import (
    cache_lock,
    read_json_cache,
    write_json_atomic,
    write_text_atomic,
)
from .compose import (
    _centerpiece_masks,
    _concat_polygons,
    _mask_rim,
    frame_scene_for_plate,
)
from .photo import _photo_multipolygon
from .recipe import (
    CENTERPIECE_FILL,
    FRAME_BUCKET0,
    FRAME_BUCKET_STEP,
    SWITCH_INTERLACE_SLUGS,
    _carrier_recipe_data,
    _frame_level_for,
)
from .spec import FaceKind, PlateSpec, _aperture
# Read through the module — see compose.py's note on PLATES_ROOT.
from . import spec as _spec

_log = logging.getLogger("optics.plates")

def _back_window_grid(
    spec: PlateSpec, fw: int, fh: int, pitch_um: float
) -> "np.ndarray":
    """The uniform-carrier window as a bool grid, rimmed at the BACK keep-out.

    The window spans the whole EXPOSED face (``back_dims`` — foil overlap only,
    wider than the front weld margin). Shared by the two-ply BACK bake and the
    single-ply FRONT bake so the carrier occupies the identical rectangle
    whichever ply it ends up on.
    """
    import numpy as np

    win = np.zeros((fh, fw), dtype=bool)
    back_w, back_h = spec.back_dims()
    if back_w <= 0 or back_h <= 0:
        return win
    bw = max(1, int(round(back_w / pitch_um)))
    bh = max(1, int(round(back_h / pitch_um)))
    bx = (fw - bw) // 2
    by = (fh - bh) // 2
    win[by : by + bh, bx : bx + bw] = True
    back_margin = spec.weld_margin_um if spec.back_margin_um is None else spec.back_margin_um
    _mask_rim(win, back_margin, pitch_um)
    return win



def _strip_svg_body(full_svg: str) -> str:
    """Return the inner body of a drawsvg <svg>…</svg> document (no wrapper)."""
    open_idx = full_svg.find("<svg")
    if open_idx == -1:
        return full_svg
    open_end = full_svg.find(">", open_idx)
    close_idx = full_svg.rfind("</svg>")
    if open_end == -1 or close_idx == -1:
        return full_svg
    return full_svg[open_end + 1 : close_idx]


def _grating_grid(
    w_px: int,
    h_px: int,
    pitch_um: float,
    period_um: float,
    duty: float,
    angle_deg: float,
    phase: float = 0.0,
) -> "np.ndarray":
    """Boolean grid (1 = gold line) of a rotated linear grating.

    Pure numpy — a projected coordinate compared against the duty window.
    No GEOS anywhere. ``(h_px, w_px)`` row-major to match ``raster_to_polygons``.
    ``phase`` shifts the grating by that fraction of a period (0.5 = half-period
    offset, the phase-π interlace used by the centerpiece globe vs colibrí).
    """
    import numpy as np

    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Physical μm coords per pixel, origin at grid center (matches raster_to_polygons).
    xs = (np.arange(w_px) - (w_px - 1) / 2.0) * pitch_um
    ys = ((h_px - 1) / 2.0 - np.arange(h_px)) * pitch_um
    X, Y = np.meshgrid(xs, ys)
    coord = (X * ca + Y * sa) / period_um + phase
    frac = coord - np.floor(coord)
    return frac < duty


# Cells per barrier period, floor and quantum: the interlace channel is p/2 and
# each half of the open slit is p/4, so only a MULTIPLE OF FOUR cells puts the
# A|B boundary and both slit edges on cell boundaries. Four is the coarsest
# lattice that still has a switch (2-cell channel, 2-cell slit).
BARRIER_COLS_QUANTUM = 4
# Relative period deviation the raster snap is allowed to introduce silently.
# Above it the baked switch crosses at a visibly different tilt angle than the
# stamped one, so it is logged at WARNING next to the coarse-bake record.
BARRIER_SNAP_TOL = 0.02


def _barrier_plate_lattice(period_um: float, pitch_um: float) -> tuple[int, float]:
    """Snap a barrier period onto the plate raster: ``(cols_per_period, period_um)``.

    The plate compositor cannot call ``_helpers.barrier_lattice`` verbatim — that
    solver derives the back RASTER from the slit lattice, whereas here the raster
    pitch is already fixed by the plate's lattice budget — but the constraint that
    matters is the same one: the front comb and the back A|B interlace must live on
    ONE lattice, so their periods are equal by construction instead of by two float
    phases agreeing. See its docstring for the physics, and for why the straddle
    class (channel A on the +x side of every slit) is the class both paths owe the
    manifest's ``switch_half_angle_deg``.

    On a raster that means a whole number of cells per period, and a multiple of
    ``BARRIER_COLS_QUANTUM`` of them. A fractional cell count is what the two
    analytic phases used to produce: the printed slit centre lands up to half a
    cell off the printed channel boundary and the printed slit width alternates
    between floor and ceil cells — at the ~11 cells per period the default box
    bakes at, ~18 % of the ±p/4 shift that only has ~2.7 cells of margin.

    When the requested period is under four cells the raster simply cannot carry
    it; the returned period is then COARSER than asked (the caller reports the
    deviation, which scales the switch tilt angle by the same factor).
    """
    q = BARRIER_COLS_QUANTUM
    cols = max(q, q * int(round(period_um / (q * pitch_um))))
    return cols, cols * pitch_um


def _barrier_masks(
    w_px: int, cols_per_period: int, anchor_col: int
) -> tuple["np.ndarray", "np.ndarray"]:
    """1-D column masks of an exactly registered barrier: ``(comb_bar, lane_a)``.

    ``comb_bar`` is the opaque FRONT bar (its complement is the open slit),
    ``lane_a`` the BACK channel-A lane. Both are pure column-index arithmetic on
    one shared lattice, which is what makes the registration exact: with ``c``
    cells per period the A|B boundary sits on the cell edge at ``anchor_col`` and
    the open slit is the ``c/2`` cells centred on that SAME edge, so every
    open-slit centre is a B→A boundary with channel A on its +x side. That is the
    straddle class and the "+tilt reveals B" parity the six generators solve for.

    Vertical bars only (``CENTER_SWITCH_AXIS_DEG`` = 0, the same assumption
    ``_check_front_comb_pure`` makes); a rotated barrier would have to go back
    through ``_grating_grid`` and give up cell-exact registration.
    """
    import numpy as np

    c = cols_per_period
    half = c // 2
    quarter = c // 4
    rel = (np.arange(w_px) - anchor_col) % c
    lane_a = rel < half
    comb_bar = ((rel + quarter) % c) >= half
    return comb_bar, lane_a


def _check_front_comb_pure(
    comb: "np.ndarray",
    box: tuple[int, int, int, int],
    plate_id: str,
    slug: str,
) -> None:
    """Contract check: a barrier-interlace front comb carries NO image feature.

    On SWITCH_INTERLACE_SLUGS the front (outer) layer must be a pure period-p
    comb across the art box. The front mask does not move with tilt, so any
    row-to-row variation in its column support is a static silhouette residual
    that no tilt angle can gate out — the banned front-image construction. The
    comb axis is CENTER_SWITCH_AXIS_DEG = 0 (vertical bars), so "pure" is
    exactly "every non-empty art-box row has identical column support"; rows the
    keep-out rim cleared entirely are skipped. Cheap: one row-broadcast compare
    over the art-box slice, no GEOS.

    Logged, not raised — a residual is an upstream design defect to fix, not a
    reason to refuse to serve an otherwise valid fab pair.
    """
    import numpy as np

    y0, y1, x0, x1 = box
    sub = comb[y0:y1, x0:x1]
    if sub.size == 0:
        return
    rows = sub[sub.any(axis=1)]
    if rows.shape[0] == 0:
        return
    differs = (rows != rows[0]).any(axis=1)
    if bool(differs.any()):
        _log.error(
            "barrier-interlace FRONT comb is not a pure period-p comb: %d of %d "
            "art-box rows differ in column support, so the front plane carries an "
            "image-shaped feature that cannot vanish under tilt (plate=%s slug=%s)",
            int(differs.sum()),
            int(rows.shape[0]),
            plate_id,
            slug,
        )


def ensure_plate_svg(plate_id: str) -> tuple[Path, Path] | None:
    """Lazily build the SVG fab pair for an existing plate (box-first moiré).

    FRONT = the foliage silhouette FILLED with a true fine grating (front
            period @ front angle), clipped to the silhouette by a row-span /
            scanline pass (silhouette raster AND grating raster → row-run
            rectangles via ``raster_to_polygons``). No GEOS booleans.
    BACK  = a uniform fine grating (back period @ back angle) across the whole
            back-carrier window rectangle.

    The two gratings differ by the small period ratio + angle offset stamped in
    ``recipe_data`` — the same moiré the shader renders analytically, now as
    real gold lines the fab can write. Respects the 400k lattice budget by
    rasterizing at a bounded pitch, so line count stays in the thousands.

    On a real (tens-of-mm) plate that budget pitch is far coarser than the
    design periods, so the baked geometry is COARSENED (see the bake_scale block
    below) — a preview-grade mask, not a production one. When that happens the
    EFFECTIVE baked periods are stamped into the manifest's ``recipe_data``
    (``svg_bake_*`` + ``svg_bake_coarsened``) and logged at WARNING, so nothing
    downstream can mistake these files for true-pitch masks; true-pitch geometry
    comes from ``export_fine.build_plate_fine`` / a tiled GDS export.

    Returns (front, back) paths or None if the plate is unknown (or its manifest
    is unreadable — the spec to bake from lives in it).
    """
    plate_dir = _spec.PLATES_ROOT / plate_id
    manifest_path = plate_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    front_svg = plate_dir / "front.svg"
    back_svg = plate_dir / "back.svg"
    if front_svg.exists() and back_svg.exists() and _svg_is_current(front_svg):
        return front_svg, back_svg
    # Same lock key as materialize_plate: the bake is heavy AND rewrites the
    # shared manifest (svg_bake_* keys), so it must not interleave with a
    # recompose of the same slot or with a second export of the same plate.
    with cache_lock(f"plate:{plate_id}"):
        if front_svg.exists() and back_svg.exists() and _svg_is_current(front_svg):
            return front_svg, back_svg
        return _bake_plate_svg(plate_id)


def _bake_plate_svg(plate_id: str) -> tuple[Path, Path] | None:
    """Raster + write the fab SVG pair. Caller holds the plate cache lock."""
    import numpy as np

    from ..patterns._helpers import MAX_LATTICE_CELLS, check_lattice_budget, raster_to_polygons

    plate_dir = _spec.PLATES_ROOT / plate_id
    manifest_path = plate_dir / "manifest.json"
    front_svg = plate_dir / "front.svg"
    back_svg = plate_dir / "back.svg"

    manifest = read_json_cache(manifest_path)
    if manifest is None or "spec" not in manifest:
        return None
    spec = PlateSpec.from_dict(manifest["spec"])
    if spec.kind is FaceKind.BLANK:
        # BARE GLASS: two valid but empty documents. Written (rather than
        # skipped) so the export bundle carries a file per layer per face and a
        # fab reader sees "this face is blank", not "this face is missing".
        _publish_svg_pair(
            manifest, manifest_path, plate_id, front_svg, back_svg,
            spec.width_um, spec.height_um, "", "", {},
        )
        return front_svg, back_svg
    if spec.kind is FaceKind.SOLID:
        # SOLID GOLD: the whole outer ply as one rectangle, the inner empty —
        # the same geometry export_fine writes for this face.
        from shapely.geometry import box as _box

        hw, hh = spec.width_um / 2.0, spec.height_um / 2.0
        front_group = _group(
            "solid", _strip_svg_body(to_svg(MultiPolygon([_box(-hw, -hh, hw, hh)]),
                                            (spec.width_um, spec.height_um), background=None)))
        _publish_svg_pair(
            manifest, manifest_path, plate_id, front_svg, back_svg,
            spec.width_um, spec.height_um, front_group, "", {},
        )
        return front_svg, back_svg
    rd = _carrier_recipe_data(spec)
    back_period = float(rd["fab_back_period_um"])
    front_period = float(rd["fab_front_period_um"])
    angle_off = float(rd["fab_angle_offset_deg"])
    base_angle = float(rd["carrier_angle_deg"])
    duty = float(rd["grating_duty"])
    frame_span = float(rd["frame_angle_span_deg"])
    frame_count = int(round(float(rd["frame_bucket_count"])))

    W, H = spec.width_um, spec.height_um

    # Fab raster pitch: fine enough to resolve the grating lines (>= 4 samples
    # per period) but the grid MUST fit the 400k lattice budget — a run-length
    # merge still allocates one row-span rectangle per line per row, and the
    # export writes real polygons. On a large plate a true 22 µm grating over
    # the whole face would blow past the cap, so if the ideal 4-samples/period
    # pitch would overflow, we COARSEN the grating period (preserving the 1.06
    # ratio + angle offset) to the finest the budget allows. The moiré geometry
    # is scale-free, so a coarser-but-legal grating carries the identical beat;
    # a genuinely sub-30 µm production mask would come from a tiled/streamed
    # GDS export, not this single-shot lazy SVG path.
    #
    # The coarsening is a real substitution (order-of-magnitude on a 50 mm face),
    # so it is NOT silent: ``bake_scale`` is logged at WARNING and the effective
    # periods land in the manifest as ``svg_bake_*`` below. The design periods in
    # ``fab_*`` stay untouched — the pair of numbers is what tells a reader these
    # SVGs are preview-grade.
    ideal_pitch = min(back_period, front_period) / 4.0
    budget_pitch = math.sqrt(W * H / (0.92 * MAX_LATTICE_CELLS))
    pitch = max(ideal_pitch, budget_pitch)
    bake_scale = (pitch / ideal_pitch) if ideal_pitch > 0 else 1.0
    if bake_scale > 1.0:
        back_period *= bake_scale
        front_period *= bake_scale
    fw = max(1, int(round(W / pitch)))
    fh = max(1, int(round(H / pitch)))
    check_lattice_budget(fw * fh, "plate grating raster", pitch_um=pitch, w=fw, h=fh)

    # Centerpiece stripe carrier (vertical): fab period + the half-period phase
    # offset baked into the back globe. Coarsen with the frame grating if the
    # budget forced a coarser pitch (same scale factor keeps the beat).
    center_period = float(rd.get("fab_center_period_um", rd["center_period_um"]))
    if bake_scale > 1.0:
        center_period *= bake_scale
    center_axis = float(rd["switch_axis_deg"])

    # Effective-bake record. Written into the manifest's recipe_data next to the
    # (unchanged) design ``fab_*`` fields so the export bundle is self-consistent:
    # a consumer reading fab_back_period_um==22 alongside svg_bake_back_period_um
    # ==330 knows exactly what these SVGs are. Empty dict when the bake is
    # period-exact AND the face is not a barrier interlace (whose lattice snap is
    # recorded even at a period-exact pitch); the keys are STRIPPED from a stale
    # manifest below either way (a finer plate must not inherit a coarse record).
    svg_bake: dict[str, Any] = {}
    if bake_scale > 1.0:
        svg_bake = {
            "svg_bake_coarsened": True,
            "svg_bake_scale": bake_scale,
            "svg_bake_pitch_um": pitch,
            "svg_bake_back_period_um": back_period,
            "svg_bake_front_period_um": front_period,
            "svg_bake_center_period_um": center_period,
        }
        _log.warning(
            "plate SVG bake COARSENED %.2f× to fit the lattice budget "
            "(raster pitch %.3g µm): back %.4g→%.4g µm, front %.4g→%.4g µm, "
            "center %.4g→%.4g µm. front.svg/back.svg are PREVIEW-grade, not fab "
            "masks — use export_fine/tiled GDS for true-pitch geometry "
            "(plate=%s slug=%s)",
            bake_scale,
            pitch,
            back_period / bake_scale,
            back_period,
            front_period / bake_scale,
            front_period,
            center_period / bake_scale,
            center_period,
            plate_id,
            spec.pattern_slug,
        )
    # WHICH exemplar, inside FaceKind.TWO_PLY — see _carrier_recipe_data.
    is_interlace = spec.pattern_slug in SWITCH_INTERLACE_SLUGS
    is_photo = spec.kind is FaceKind.PHOTO
    single_ply = bool(getattr(spec, "single_ply", False))
    is_region = spec.kind is FaceKind.REGION
    aperture = _aperture(spec)
    side_px = max(8, int(round(CENTERPIECE_FILL * aperture / pitch))) if aperture > 0 else 0
    cxg = fw // 2
    cyg = fh // 2

    # Barrier REGISTRATION on the plate raster. The front comb and the back A|B
    # lanes used to be two independent analytic gratings agreeing only through a
    # pair of float phase constants (-0.25 and 0.0) written 60 lines apart in two
    # different blocks; at a raster pitch that does not divide the period a whole
    # number of times that agreement does not survive sampling. Solve ONE lattice
    # here — cell-exact period, cell-exact phase, both layers built from it below.
    #
    # Anchored on ``cxg`` — the grid centre column, which is also the centerpiece
    # art-box centre column — so the registration is independent of the aperture:
    # the extent-dependent class flip ``_helpers.barrier_lattice`` has to solve for
    # (its raster starts at the tile EDGE) cannot arise here, and the published
    # ``switch_barrier_phase_um`` = 0 stays true at every plate size.
    barrier_cols = 0
    barrier_anchor = cxg
    barrier_comb: "np.ndarray | None" = None
    barrier_lane_a: "np.ndarray | None" = None
    if is_interlace:
        if abs(center_axis % 180.0) > 1e-9:
            # Cell-exact registration is column arithmetic, so it only exists for
            # vertical bars. A rotated switch axis would have to go back through
            # _grating_grid and give the registration up — and _check_front_comb_pure
            # would stop meaning anything either. Loud, but keep baking.
            _log.error(
                "barrier-interlace face has a non-vertical switch axis (%.3f°); the "
                "cell-exact barrier lattice below assumes vertical bars, so the "
                "baked comb ignores the rotation (plate=%s slug=%s)",
                center_axis,
                plate_id,
                spec.pattern_slug,
            )
        want_period = center_period
        barrier_cols, center_period = _barrier_plate_lattice(want_period, pitch)
        barrier_comb, barrier_lane_a = _barrier_masks(fw, barrier_cols, barrier_anchor)
        # The A|B boundary is the LEFT EDGE of column ``barrier_anchor``; x = 0 is
        # the centre of column (fw-1)/2 (see ``_grating_grid``). Zero on an even
        # grid, half a cell on an odd one — reported, not hidden, because it is the
        # number a shader would need to draw the same lattice.
        barrier_phase_um = (barrier_anchor - (fw - 1) / 2.0 - 0.5) * pitch
        period_err = center_period - want_period
        svg_bake.update(
            {
                # Achieved comb == interlace period, and the witnesses that say so
                # (the generator side publishes the same three in ``extra``).
                "svg_bake_barrier_period_um": center_period,
                "svg_bake_barrier_cell_um": pitch,
                "svg_bake_barrier_cols_per_period": barrier_cols,
                "svg_bake_barrier_phase_um": barrier_phase_um,
                "svg_bake_barrier_period_err_um": period_err,
            }
        )
        if svg_bake.get("svg_bake_coarsened"):
            # The coarse-bake record above was stamped from the pre-snap period.
            svg_bake["svg_bake_center_period_um"] = center_period
        if abs(period_err) > BARRIER_SNAP_TOL * want_period:
            _log.warning(
                "barrier lattice SNAPPED to the plate raster: %d cells × %.4g µm "
                "= %.4g µm comb+interlace period against the requested %.4g µm "
                "(%+.1f %%, so the baked switch crosses at %.2f× the tilt angle "
                "the manifest stamps). Registration itself is EXACT at the snapped "
                "period — every open-slit centre sits on an A|B channel boundary — "
                "but the raster pitch is set by the lattice budget and only a whole "
                "multiple-of-%d cell count can hold that. front.svg/back.svg are "
                "preview-grade here; true-pitch barrier geometry comes from "
                "export_fine (plate=%s slug=%s)",
                barrier_cols,
                pitch,
                center_period,
                want_period,
                100.0 * period_err / want_period,
                center_period / want_period,
                BARRIER_COLS_QUANTUM,
                plate_id,
                spec.pattern_slug,
            )

    def _center_masks() -> "tuple[np.ndarray, np.ndarray]":
        """Full-grid ``(front_art, back_art)`` bools for the aperture.

        Data-directed by the pattern slug (whatever ``_centerpiece_masks``
        returns), so the fab pair matches the preview centerpiece. Empty for a
        production face: those are blank, solid, photo or region_art, none of
        which fills a silhouette with a carrier.
        """
        empty = np.zeros((fh, fw), dtype=bool)
        if side_px <= 0 or is_region:
            # a region centrepiece has no silhouette to fill with a carrier:
            # its exact rects are concatenated below (single_layer_region_rects)
            return empty, empty.copy()
        masks = _centerpiece_masks(spec, side_px)
        if masks is None:
            return empty, empty.copy()
        from PIL import Image as _Image

        def _place(art: "np.ndarray") -> "np.ndarray":
            if art.shape[0] != side_px or art.shape[1] != side_px:
                im = _Image.fromarray((art.astype(np.uint8) * 255), "L").resize(
                    (side_px, side_px), _Image.NEAREST
                )
                art = np.asarray(im) > 127
            m = np.zeros((fh, fw), dtype=bool)
            x0 = cxg - side_px // 2
            y0 = cyg - side_px // 2
            xa = max(0, x0); ya = max(0, y0)
            xb = min(fw, x0 + side_px); yb = min(fh, y0 + side_px)
            m[ya:yb, xa:xb] = art[ya - y0 : yb - y0, xa - x0 : xb - x0]
            return m, art

        front_full, _front_side = _place(masks[0])
        back_full, _ = _place(masks[1])
        return front_full, back_full

    # Centerpiece art (front + back), placed on the full fab grid once so both
    # layers share it.
    front_art, back_art = _center_masks()

    # --- FRONT: perimeter foliage frame (grating) + centerpiece -------------
    active_w, active_h = spec.active_dims()
    front_group = ""
    if active_w > 0 and active_h > 0:
        rect = RectFrame(width_um=active_w, height_um=active_h)
        frame_params = spec.frame.to_frame_params()
        frame_params.fill_interior = False  # perimeter band, center open for art
        scene = frame_scene_for_plate(plate_dir, manifest, rect, frame_params)
        # Per-motif graylevel silhouette — the same angle-bucket encoding the
        # preview uses, so the fab lines match the shimmer the user sees.
        sil_img = render_scene_to_image(
            scene, rect, frame_params, pitch, level_fn=_frame_level_for
        )
        # Paste the active-rect graylevel band into a full-plate grid.
        sil_lvl = np.zeros((fh, fw), dtype=np.uint8)
        aw = min(fw, sil_img.size[0])
        ah = min(fh, sil_img.size[1])
        ox = (fw - aw) // 2
        oy = (fh - ah) // 2
        sil_lvl[oy : oy + ah, ox : ox + aw] = np.asarray(sil_img)[:ah, :aw]
        sil = sil_lvl > 0
        # Enforce the front keep-out rim.
        _mask_rim(sil, spec.weld_margin_um, pitch)
        # Fill each angle bucket with a grating rotated to that bucket, so the
        # baked gold lines carry the SAME per-motif fringe directions as the
        # shader (frameAngle = base + (b - (N-1)/2)·span). Decode the graylevel
        # back to a bucket index and OR the per-bucket grating in.
        base_front = base_angle + angle_off
        frame_grating = np.zeros((fh, fw), dtype=bool)
        for b in range(frame_count):
            lvl_lo = FRAME_BUCKET0 + b * FRAME_BUCKET_STEP - FRAME_BUCKET_STEP // 2
            lvl_hi = FRAME_BUCKET0 + b * FRAME_BUCKET_STEP + FRAME_BUCKET_STEP // 2
            in_bucket = (sil_lvl > max(0, lvl_lo)) & (sil_lvl <= lvl_hi)
            if not in_bucket.any():
                continue
            ang = base_front + (b - 0.5 * (frame_count - 1)) * frame_span
            g = _grating_grid(fw, fh, pitch, front_period, duty, ang)
            frame_grating |= in_bucket & g
        # Centerpiece front layer. Three constructions:
        #   * barrier-interlace (SWITCH_INTERLACE_SLUGS — the hidden switch
        #     exemplar): a NEUTRAL slit comb (open duty 0.5) over the FULL
        #     centerpiece art box — the CENTERPIECE_FILL square, NOT the union
        #     of the silhouettes. A union-clipped comb is itself a static front
        #     image (its envelope is the union, which never moves with tilt —
        #     the measured ~0.31 front residual), and physically the comb must
        #     cover every column any back lane can slide under within the first
        #     zone (>= p/2 beyond the union), so the full art-box square is the
        #     clean choice. Comb AND lanes come from the one lattice solved
        #     above (``_barrier_plate_lattice`` / ``_barrier_masks``), so every
        #     open slit straddles an A|B lane boundary head-on to the cell and
        #     +/- tilt reveals A or B cleanly; the solved phase is published as
        #     ``switch_barrier_phase_um`` for the preview shader to draw the
        #     same lattice instead of its own constant.
        #   * photo / region_art (every written production face): NOTHING from
        #     the raster path. Their geometry is EXACT vector — the line screen
        #     (``photo_band_rects``) and the region gratings
        #     (``single_layer_region_rects``) — concatenated after the raster
        #     below. Filling a silhouette with the centerpiece carrier as well
        #     would superimpose a second 50 %-duty grating on geometry that is
        #     already carrying the effect.
        #   * shading moire (SHIMMER_MOIRE_SLUGS — the other exemplar): the
        #     FRONT silhouette filled with the vertical carrier (phase 0).
        if is_interlace:
            art_box = np.zeros((fh, fw), dtype=bool)
            bx0 = by0 = bx1 = by1 = 0
            if side_px > 0:
                bx0 = max(0, cxg - side_px // 2)
                by0 = max(0, cyg - side_px // 2)
                bx1 = min(fw, bx0 + side_px)
                by1 = min(fh, by0 + side_px)
                art_box[by0:by1, bx0:bx1] = True
                _mask_rim(art_box, spec.weld_margin_um, pitch)
            # Opaque bars of the solved lattice (``barrier_comb`` is a column
            # mask; its complement is the open slit, centred on an A|B boundary).
            art_carrier = art_box & barrier_comb[None, :]
        elif is_photo or is_region:
            art_carrier = np.zeros((fh, fw), dtype=bool)
        else:
            center_grating = _grating_grid(fw, fh, pitch, center_period, duty, center_axis)
            art_carrier = front_art & center_grating
        if is_interlace:
            # Front plane of a barrier switch: image-free comb or nothing, so
            # any silhouette leaking into the comb is caught here.
            _check_front_comb_pure(
                art_carrier, (by0, by1, bx0, bx1), plate_id, spec.pattern_slug
            )
        front_grid = (sil & frame_grating) | art_carrier
        # SINGLE PLY: nothing extra on this layer. One sheet has no second plane
        # for a carrier to beat against, so (2026-09-10) the face writes the
        # photograph and the leaf gratings on bare glass and NOTHING else — the
        # same geometry ``_raster_compose_plate`` composes and the single-ply
        # block in ``export_fine.build_plate_fine`` writes ("fab SVG = preview
        # PNG", CLAUDE.md). The carrier this branch used to bake over the back
        # window was the earlier design; on one ply its beat with the leaves was
        # static bars, not a moiré, so it went — and a bake that still emitted it
        # would have put ~500 mm² of gold on the mask that no other path has.
        front_polys = raster_to_polygons(front_grid, pitch, (W, H))
        if is_photo:
            # The line screen is EXACT vector geometry — bands plus, inside each
            # coloured band, its diffraction sub-grating — so it bypasses the
            # coarse budget raster entirely and is concatenated (never unioned;
            # see _concat_polygons) onto the frame's polygons.
            front_polys = _concat_polygons(
                front_polys, _photo_multipolygon(spec)
            )
        if is_region:
            # Same for a single-layer diffraction centrepiece: its 4-6 um
            # region gratings are exact rects from the fine bake's own emitter.
            from ..export_fine import single_layer_region_rects
            from ..patterns.bitmap.photo import rects_to_multipolygon

            rr = single_layer_region_rects(spec)
            if rr.shape[0]:
                front_polys = _concat_polygons(front_polys, rects_to_multipolygon(rr))
        front_group = _group("frame+centerpiece", _strip_svg_body(to_svg(front_polys, (W, H), background=None)))

    # --- BACK: uniform carrier grating + centerpiece (phase pi) -------------
    # Empty on a single-ply face: there is no inner ply, and the carrier that
    # would live here is already on the front layer above.
    back_group = ""
    back_w, back_h = spec.back_dims()
    if not single_ply and back_w > 0 and back_h > 0:
        win = _back_window_grid(spec, fw, fh, pitch)
        carrier_grating = _grating_grid(fw, fh, pitch, back_period, duty, base_angle)
        if is_interlace:
            # Barrier-interlace back layer: BOTH silhouettes interleaved in
            # alternating lanes (lane pitch = half the barrier pitch). A (front
            # silhouette) fills the channel-A lanes, B (back silhouette) the
            # channel-B lanes. ``barrier_lane_a`` is the SAME solved lattice the
            # front comb above is cut from — that shared lattice IS the
            # registration, so the two periods cannot drift and each open slit
            # sits on a lane boundary to the cell. This is the whole moving image
            # — the front barrier reveals one lane class per tilt. The plain
            # carrier is cleared across the union so the lanes read.
            union_art = front_art | back_art
            even_lane = barrier_lane_a[None, :]
            interleave = (front_art & even_lane) | (back_art & ~even_lane)
            back_grid = (win & carrier_grating & ~union_art) | (win & interleave)
        else:
            # Back centerpiece art filled with the vertical carrier, shifted by
            # half a period (phase pi). Empty on the shading-moire exemplar, in
            # which case the back is a clean uniform carrier window.
            center_grating_pi = _grating_grid(
                fw, fh, pitch, center_period, duty, center_axis, phase=0.5
            )
            # The back art overrides the carrier inside the aperture.
            back_grid = (win & carrier_grating & ~back_art) | (back_art & center_grating_pi)
        back_polys = raster_to_polygons(back_grid, pitch, (W, H))
        back_group = _group("carrier+centerpiece", _strip_svg_body(to_svg(back_polys, (W, H), background=None)))

    _publish_svg_pair(
        manifest, manifest_path, plate_id, front_svg, back_svg,
        W, H, front_group, back_group, svg_bake,
    )
    return front_svg, back_svg


def _publish_svg_pair(
    manifest: dict[str, Any],
    manifest_path: Path,
    plate_id: str,
    front_svg: Path,
    back_svg: Path,
    width_um: float,
    height_um: float,
    front_group: str,
    back_group: str,
    svg_bake: dict[str, Any],
) -> None:
    """Write the fab SVG pair and stamp the manifest that describes it.

    SVGs first, manifest last — the manifest's ``svg_bake_*`` record must never
    describe geometry that is not on disk yet. Shared with the BLANK early
    return so a blank face publishes through exactly the same steps (empty
    groups, cleared bake record) rather than a second, nearly-identical tail.
    """
    write_text_atomic(front_svg, _wrap_svg(width_um, height_um, [front_group]))
    write_text_atomic(back_svg, _wrap_svg(width_um, height_um, [back_group]))

    files = manifest.setdefault("files", {})
    files["front_svg"] = f"/data/plates/{plate_id}/front.svg"
    files["back_svg"] = f"/data/plates/{plate_id}/back.svg"
    # Stamp (or clear) the effective-bake record so the manifest that ships next
    # to these SVGs describes the geometry they actually contain. The design
    # ``fab_*`` fields are left alone — the export bundle needs BOTH numbers.
    rd_out = manifest.setdefault("recipe_data", {})
    for key in _SVG_BAKE_KEYS:
        rd_out.pop(key, None)
    rd_out.update(svg_bake)
    write_json_atomic(manifest_path, manifest)


# recipe_data keys ensure_plate_svg writes to describe the EFFECTIVE baked
# geometry when the lattice budget forced a coarser period (see the bake_scale
# block) or a coarser barrier lattice (the barrier snap). Listed here so a re-bake
# at a finer pitch can strip a stale record instead of leaving a manifest that
# claims a coarsening that no longer applies.
_SVG_BAKE_KEYS = (
    "svg_bake_coarsened",
    "svg_bake_scale",
    "svg_bake_pitch_um",
    "svg_bake_back_period_um",
    "svg_bake_front_period_um",
    "svg_bake_center_period_um",
    "svg_bake_barrier_period_um",
    "svg_bake_barrier_cell_um",
    "svg_bake_barrier_cols_per_period",
    "svg_bake_barrier_phase_um",
    "svg_bake_barrier_period_err_um",
)


# Bump when the SVG compose geometry changes: cached plate SVGs are only
# reused if they carry the current marker, so a formula fix (e.g. the
# aperture-scaling fix) invalidates stale files under unchanged spec hashes.
#
# It used to be a hand-bumped string (last value: "plate-svg-v20"; its v4..v20
# history is in docs/plates-changelog.md) and is COMPUTED now — see
# ``cache_fingerprint``. The closure OVERLAPS the compose one on purpose: the
# rule is that the fab SVG and the preview PNG draw the same geometry
# (CLAUDE.md), and ``_bake_plate_svg`` reuses compose's masks, so a change there
# moves both markers.
#
# The marker is written into the SVG itself (``_wrap_svg``) rather than into a
# manifest, because a cached SVG pair is a pair of FILES with no manifest of
# their own; ``_svg_is_current`` reads it back out of the first 256 bytes.
PLATE_SVG_FINGERPRINT = f"plate-svg-{fingerprint(PLATE_SVG_CLOSURE, PHOTO_ASSETS)}"


def _svg_is_current(svg_path: Path) -> bool:
    try:
        head = svg_path.read_text(encoding="utf-8", errors="ignore")[:256]
    except OSError:
        return False
    return PLATE_SVG_FINGERPRINT in head


def _wrap_svg(width_um: float, height_um: float, groups: list[str]) -> str:
    # width/height carry an explicit physical unit (mm) so importers that
    # honor them place the plate at true scale; the viewBox keeps user units
    # = μm, matching every nested path coordinate.
    body = "".join(groups)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<!--{PLATE_SVG_FINGERPRINT}-->'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_um / 1000.0:.4f}mm" height="{height_um / 1000.0:.4f}mm" '
        f'viewBox="{-width_um/2:.2f} {-height_um/2:.2f} {width_um:.2f} {height_um:.2f}">'
        f'{body}</svg>'
    )


def _group(layer_id: str, inner: str) -> str:
    return f'<g id="{layer_id}">{inner}</g>'


def _inner_svg_paths(full_svg: str, scale: float = 1.0) -> str:
    """Strip the outer <svg>...</svg> wrapper off a drawsvg output so we can
    nest it inside our wrapper, optionally scaling the group about the shared
    centered origin (used to blow the central pattern up to the aperture).
    Drawsvg emits a fixed prelude that we don't want twice in one document.
    """
    # Drawsvg emits something like:
    #   <?xml ...?><svg ...><defs>...</defs><rect ...>...<path .../></svg>
    # Find the first '>' after '<svg' and the closing '</svg>'.
    open_idx = full_svg.find("<svg")
    if open_idx == -1:
        return full_svg
    open_end = full_svg.find(">", open_idx)
    close_idx = full_svg.rfind("</svg>")
    if open_end == -1 or close_idx == -1:
        return full_svg
    inner = full_svg[open_end + 1 : close_idx]
    if abs(scale - 1.0) > 1e-9:
        return f'<g id="central" transform="scale({scale:.8g})">{inner}</g>'
    return _group("central", inner)
