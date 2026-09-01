"""Single-blank bonded-pair panelization — all 6 faces as 12 sub-plates on one
square chrome photomask blank, written in ONE litho pass (no backside alignment).

Physical architecture (the BONDED build — ``BoxSpec.bonded``): each face is TWO
single-side chrome plates from thick sheet stock (default 1.5 mm soda lime),
glued face-to-face with UV optical adhesive —

    viewer
      │   OUTER ply glass (full cut dims)
      │   F chrome  ── at the bond interface (protected inside the glue line)
      │   NOA optical adhesive (index-matched, negligible thickness)
      │   INNER ply glass — inset one ply per edge (the nested-shell bevel step)
      │   B chrome  ── faces the box interior
    interior

The ply thickness IS the optical parallax gap; the wall is the 2-ply stack, and
the inner ply's inset makes the box corners interleave as a two-step staircase
approximating a 45° miter (see ``assembly.bonded_cut_list``). Both chrome
layers face DOWN relative to how the maskless aligner writes them (chrome up),
so BOTH sub-plates are written MIRRORED (``mirror_for_stack=True``, x → −x in
the plate frame); after the physical flip at assembly the viewer sees the
design frame. Barrier/scanimation periods need NO special handling here: the
plate compositor derives them from the spec's glass (``plates.fab_center_
period_um`` / ``water_scan_fab_pitch_um``), so the ~5° switch crossing holds
on this stock automatically while the fine 22/24/4.4 µm families stay fine.

Registration between F and B happens at the bench, not in the tool: matching
moiré vernier combs (front pitch != back pitch → beat fringe amplifying
relative slide ~11×) sit at IDENTICAL stack coordinates on both plies, inside
the interior foil-fold band — the only rim zone where BOTH plies exist (the
outer ply protrudes one ply past the inner everywhere). Plus tick-code IDs so
the 12 diced plates stay identifiable, and corner dicing ticks in the streets.

What this module produces:

  1. :func:`pack_blank` — deterministic shelf packer for rectangles inside the
     blank's usable SQUARE (edge margin excluded), ≥ one dicing street apart.
  2. :func:`solve_blank_max_scale` — binary-search the largest UPRIGHT box
     (H/W held, default 1.1) whose 12 sub-plates pack onto one blank AND pass
     ``validate_bonded_assembly``.
  3. :func:`build_blank_gds` — the deliverable: every sub-plate's fine
     geometry (via ``export_fine.build_plate_fine`` — glass-derived periods,
     DRC healing) mirrored/rotated/placed into a SINGLE chrome layer, plus
     verniers, IDs and dicing ticks. One write session on the MLA.
  4. :func:`write_blank_layout_svg` — fast layout preview (rects only).

All lengths are micrometers unless a ``_mm`` suffix says otherwise.

CLI (full generation materializes 6 fine plates — one heavy compute; do not
run alongside other heavy processes):

    uv run python -m app.export_blank --out data/blank/blank.gds
    uv run python -m app.export_blank --pack-only     # pure math + SVG
    uv run python -m app.export_blank --plate-thickness 700 --blank-side 101600
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .export_wafer import Placement, PlateRect

# --- blank + dicing constants (um) -------------------------------------------

BLANK_SIDE_UM = 127_000.0        # 5" square mask blank (4" = 101_600 via CLI)
BLANK_EDGE_MARGIN_UM = 4_000.0   # resist edge bead + tweezer handling rim
BLANK_STREET_UM = 1_000.0        # hand-scribe dicing lane between sub-plates

# Default stock: standard 1.5 mm soda-lime sheet (the cheap, everywhere-stocked
# option). Thickness is a first-class parameter — thinner coated stock (0.7 mm
# D263, 0.5 mm quartz) drops straight in via --plate-thickness and every
# derived number (wall, optical gap, fab periods, foil margins) follows.
PLATE_THICKNESS_UM = 1_500.0
BLANK_GLASS_N = 1.52             # soda lime
BLANK_GLASS_MATERIAL = "soda lime"

# 1/4" copper foil: the bonded stepped edge consumes 3 plies (4.5 mm at the
# 1.5 mm default) of tape width; 1/4" (6350) leaves a 925 µm fold per face.
BLANK_TAPE_WIDTH_UM = 6_350.0

# Upright ring box: taller than wide so the ring can stand.
DEFAULT_ASPECT_H_OVER_W = 1.1

# Solver hinge (validation only): 3 segments at 0.7 coverage stays cuttable on
# small boxes where the default 5-segment hinge would reject.
_SOLVER_HINGE = {"segments": 3, "coverage": 0.7}

# Default spare bonded pairs: the two hero faces (front switch, capybara back)
# — simultaneously the most load-bearing designs and the largest hand-cleave
# cuts. Costs ~1.8 mm of box (30.5 -> 28.7 mm on the 5" blank); the interior
# still stands the ring. Pass --spares "" to reclaim the size.
DEFAULT_SPARE_FACES: tuple[str, ...] = ("front", "back")

# --- assembly-registration verniers (bench-readable, NOT the BSA fiducials) --
# export_fine's 20/25 µm verniers are sized for a mask-aligner microscope; a
# bonded pair is aligned by hand under a loupe, so these are coarser and
# longer. Stacking an 80 µm comb over an 88 µm comb beats at
# p_f·p_b/(p_b−p_f) = 880 µm with motion amplification p_b/(p_b−p_f) = 11× —
# a 5 µm relative slide moves the fringe 55 µm, clearly visible at 10×.
# Line duty 0.5 keeps the minimum feature at 40 µm, far above the litho floor.
#
# PLACEMENT: the combs sit in the INTERIOR foil-fold band — the ring between
# the inner ply's edge and the interior fold-over — because that is the only
# rim zone where BOTH plies have glass (the outer ply protrudes one ply past
# the inner everywhere, so the outer keep-out rim has nothing beneath it).
# The interior fold hides them from inside the finished box; from outside they
# sit in the decorative frame band and read as border texture.
VERNIER_PITCH_F_UM = 80.0
VERNIER_PITCH_B_UM = 88.0
VERNIER_LINES = 24               # ≈2 beat fringes over the comb
VERNIER_BAR_LEN_UM = 600.0       # bar length across the comb
VERNIER_CORNER_CLEAR_UM = 800.0  # comb end clearance from the INNER ply corner
VERNIER_MIN_LINES = 8

# ID tick code: face index (1..6 bars) + a B-layer underline, near the bottom
# edge in the same both-plies band as the verniers.
ID_TICK_W_UM = 150.0
ID_TICK_LEN_UM = 400.0
ID_TICK_PITCH_UM = 350.0

# Dicing ticks just outside each sub-plate corner, inside the street.
DICE_TICK_LEN_UM = 400.0
DICE_TICK_W_UM = 60.0
DICE_TICK_GAP_UM = 150.0

# GDS layer map — ONE chrome layer: the whole panel is a single-side write.
LAYER_CHROME = (10, 0)
LAYER_OUTLINE = (1, 0)     # sub-plate outlines (annotation, not chrome)
LAYER_BLANK = (99, 0)      # blank + usable-region outlines
LAYER_LABEL = (3, 0)       # text annotation

SUBPLATE_SUFFIXES = ("F", "B")  # F = outer ply (front layer), B = inner ply


def subplate_id(face: str, suffix: str) -> str:
    return f"{face}:{suffix}"


# --- rect building ------------------------------------------------------------

def pair_rects(
    width_um: float,
    depth_um: float,
    height_um: float,
    ply_um: float,
    spare_faces: tuple[str, ...] = (),
) -> list[PlateRect]:
    """12 sub-plate rects from the bonded nested-shell cut list, plus one full
    spare bonded PAIR per entry in ``spare_faces``.

    ``face:F`` is the OUTER ply (full cut dims), ``face:B`` the INNER ply
    (inset one ply per edge — genuinely smaller, which packs tighter). Spares
    are hand-cleave insurance: a face may appear more than once in
    ``spare_faces`` for multiple spare pairs; each spare's id gains a
    ``:spareN`` suffix but its GEOMETRY is identical to the original (same
    masks, same verniers, same ID ticks — interchangeable at the bench). The
    solver maximizes the box that fits 12 + 2·len(spare_faces) sub-plates, so
    every spare pair costs box size: on the 5-inch default, 0 spares → ~30.5 mm,
    +2 pairs → ~28.7 mm (the sweet spot), a full duplicate set → ~23 mm.
    """
    from .assembly import bonded_cut_list

    entries = bonded_cut_list(width_um, depth_um, height_um, ply_um)
    by_id: dict[str, tuple[float, float]] = {}
    rects = []
    for e in entries:
        sid = subplate_id(e["face"], "F" if e["ply"] == "outer" else "B")
        by_id[sid] = (e["width_um"], e["height_um"])
        rects.append(PlateRect(sid, e["width_um"], e["height_um"]))
    for i, fid in enumerate(spare_faces):
        for suf in SUBPLATE_SUFFIXES:
            sid = subplate_id(fid, suf)
            if sid not in by_id:
                raise ValueError(f"spare face {fid!r} is not a box face")
            w, h = by_id[sid]
            rects.append(PlateRect(f"{sid}:spare{i + 1}", w, h))
    rects.sort(key=lambda r: (max(r.width_um, r.height_um), r.area_um2()), reverse=True)
    return rects


# --- square shelf packer --------------------------------------------------------

def pack_blank(
    rects: list[PlateRect],
    *,
    usable_side_um: float,
    street_um: float = BLANK_STREET_UM,
) -> list[Placement] | None:
    """Deterministic shelf packer for rectangles inside a SQUARE.

    Same reservation scheme as ``export_wafer.pack_plates`` (each plate
    inflated by one full street on its right and top; the blank edge margin
    absorbs the outer trailing street), but containment is the usable square,
    not a circle — so the sweep keeps the packing with the smallest bounding
    SQUARE side. Returns blank-centered placements or ``None`` if no fit.
    """
    if not rects:
        return []

    def _try_pack(order: list[PlateRect], strip_width: float) -> list[Placement] | None:
        placements: list[Placement] = []
        cursor_x = 0.0
        shelf_y = 0.0
        shelf_h = 0.0
        for r in order:
            cands = [(r.width_um, r.height_um, False)]
            if abs(r.width_um - r.height_um) > 1e-6:
                cands.append((r.height_um, r.width_um, True))
            cands.sort(key=lambda c: (c[1], c[2]))

            placed = False
            for w, h, rot in cands:
                if cursor_x + w + street_um <= strip_width + 1e-6:
                    placements.append(Placement(r.face, cursor_x, shelf_y, w, h, rot))
                    cursor_x += w + street_um
                    shelf_h = max(shelf_h, h + street_um)
                    placed = True
                    break
            if placed:
                continue
            new_shelf_y = shelf_y + shelf_h
            for w, h, rot in cands:
                if w + street_um <= strip_width + 1e-6:
                    placements.append(Placement(r.face, 0.0, new_shelf_y, w, h, rot))
                    cursor_x = w + street_um
                    shelf_y = new_shelf_y
                    shelf_h = h + street_um
                    placed = True
                    break
            if not placed:
                return None
        return placements

    def _center_and_check(placements: list[Placement]) -> list[Placement] | None:
        min_x = min(p.x0 for p in placements)
        min_y = min(p.y0 for p in placements)
        max_x = max(p.x0 + p.width_um for p in placements)
        max_y = max(p.y0 + p.height_um for p in placements)
        if max_x - min_x > usable_side_um + 1e-6 or max_y - min_y > usable_side_um + 1e-6:
            return None
        dx = -(min_x + max_x) / 2.0
        dy = -(min_y + max_y) / 2.0
        return [replace(p, x0=p.x0 + dx, y0=p.y0 + dy) for p in placements]

    widest = max(max(r.width_um, r.height_um) + street_um for r in rects)
    raw_widths = [widest * k for k in range(1, len(rects) + 1)] + [usable_side_um]
    candidate_widths = sorted({w for w in raw_widths if w <= usable_side_um + 1e-6})

    best: list[Placement] | None = None
    best_side = float("inf")
    for sw in candidate_widths:
        packed = _try_pack(list(rects), sw)
        if packed is None:
            continue
        centered = _center_and_check(packed)
        if centered is None:
            continue
        bw = max(p.x0 + p.width_um for p in centered) - min(p.x0 for p in centered)
        bh = max(p.y0 + p.height_um for p in centered) - min(p.y0 for p in centered)
        side = max(bw, bh)
        if side < best_side:
            best_side = side
            best = centered
    return best


# --- max-scale solver -----------------------------------------------------------

@dataclass
class BlankPlanResult:
    """The largest bonded box that packs (and validates), plus its packing."""

    width_um: float
    depth_um: float
    height_um: float
    plate_thickness_um: float   # one ply (also the optical parallax gap glass)
    wall_thickness_um: float    # bonded stack = 2 x ply
    tape_width_um: float
    blank_side_um: float
    usable_side_um: float
    placements: list[Placement]

    def dims_mm(self) -> dict[str, float]:
        return {
            "width_mm": round(self.width_um / 1000.0, 3),
            "depth_mm": round(self.depth_um / 1000.0, 3),
            "height_mm": round(self.height_um / 1000.0, 3),
            "ply_mm": round(self.plate_thickness_um / 1000.0, 3),
            "wall_mm": round(self.wall_thickness_um / 1000.0, 3),
            # Ring-fit readout: clear interior span between opposite walls.
            "interior_width_mm": round((self.width_um - 2 * self.wall_thickness_um) / 1000.0, 3),
            "interior_depth_mm": round((self.depth_um - 2 * self.wall_thickness_um) / 1000.0, 3),
            "interior_height_mm": round((self.height_um - 2 * self.wall_thickness_um) / 1000.0, 3),
        }


def solve_blank_max_scale(
    *,
    aspect_h_over_w: float = DEFAULT_ASPECT_H_OVER_W,
    plate_thickness_um: float = PLATE_THICKNESS_UM,
    tape_width_um: float = BLANK_TAPE_WIDTH_UM,
    blank_side_um: float = BLANK_SIDE_UM,
    edge_margin_um: float = BLANK_EDGE_MARGIN_UM,
    street_um: float = BLANK_STREET_UM,
    spare_faces: tuple[str, ...] = DEFAULT_SPARE_FACES,
    # lo must itself be a VALID bonded box (the search brackets from it): at
    # 1.5 mm plies + 1/4" tape the aperture floor rejects anything much under
    # ~12 mm, so start comfortably above that.
    lo_um: float = 14_000.0,
    hi_um: float = 60_000.0,
    iters: int = 40,
) -> BlankPlanResult | None:
    """Binary-search the largest box (W = D, H/W held) whose 12 bonded-pair
    sub-plates pack onto ONE blank and whose bonded assembly validates
    (nested-shell existence, per-ply apertures, hinge). Width rounded down to
    100 µm. Same bracketing scheme as ``export_wafer.solve_max_scale``."""
    from .assembly import FoilSpec, HingeSpec, validate_bonded_assembly

    p = plate_thickness_um
    usable = blank_side_um - 2.0 * edge_margin_um
    foil = FoilSpec(tape_width_um=tape_width_um)
    hinge = HingeSpec(**_SOLVER_HINGE)

    def fits(w: float) -> list[Placement] | None:
        h = w * aspect_h_over_w
        try:
            validate_bonded_assembly(w, w, h, p, foil, hinge)
        except ValueError:
            return None
        rects = pair_rects(w, w, h, p, spare_faces)
        return pack_blank(rects, usable_side_um=usable, street_um=street_um)

    lo_fit = fits(lo_um)
    if lo_fit is None:
        return None
    best_w, best_pl = lo_um, lo_fit
    hi_fit = fits(hi_um)
    if hi_fit is not None:
        best_w, best_pl = hi_um, hi_fit
    else:
        lo, hi = lo_um, hi_um
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            pl = fits(mid)
            if pl is not None:
                best_w, best_pl = mid, pl
                lo = mid
            else:
                hi = mid
    w = math.floor(best_w / 100.0) * 100.0
    pl = fits(w)
    if pl is None:
        w, pl = best_w, best_pl
    return BlankPlanResult(
        width_um=w,
        depth_um=w,
        height_um=w * aspect_h_over_w,
        plate_thickness_um=p,
        wall_thickness_um=2.0 * p,
        tape_width_um=tape_width_um,
        blank_side_um=blank_side_um,
        usable_side_um=usable,
        placements=pl,
    )


# --- box spec at the blank dims ---------------------------------------------------

def blank_box_spec(result: BlankPlanResult, *, glass_n: float = BLANK_GLASS_N,
                   glass_material: str = BLANK_GLASS_MATERIAL) -> Any:
    """The real six-face plan resized to the blank's bonded mini box.

    ``bonded=True`` + the ply glass drive everything downstream: cut dims and
    foil margins via the nested-shell math in ``normalize_face_dims``, and the
    barrier/scanimation fab periods via the glass-derived
    ``plates.fab_center_period_um`` — no per-face pattern_params needed.
    """
    from .assembly import FoilSpec
    from .boxes import default_box_spec
    from .plates import GlassSpec

    spec = default_box_spec()
    spec.width_um = result.width_um
    spec.depth_um = result.depth_um
    spec.height_um = result.height_um
    spec.bonded = True
    spec.glass = GlassSpec(
        thickness_um=result.plate_thickness_um, material=glass_material, n=glass_n
    )
    spec.foil = FoilSpec(tape_width_um=result.tape_width_um)
    spec.normalize_face_dims()
    return spec


# --- bench marks: verniers, IDs, dicing ticks ---------------------------------------

def _comb_rects(
    cx: float, cy: float, pitch_um: float, n: int, *, vertical_bars: bool
) -> np.ndarray:
    """A comb of ``n`` bars at ``pitch_um``, duty 0.5, centered on (cx, cy).

    ``vertical_bars=True`` → bars run in y, pitch along x (senses x-slide);
    False → transposed (senses y-slide). (N,4) [x0,x1,y0,y1] µm rects, phase
    symmetric about the comb center so the layout survives the stack mirror.
    """
    line = pitch_um / 2.0
    total = (n - 1) * pitch_um
    hy = VERNIER_BAR_LEN_UM / 2.0
    out = []
    for i in range(n):
        c = -total / 2.0 + i * pitch_um
        if vertical_bars:
            out.append((cx + c - line / 2.0, cx + c + line / 2.0, cy - hy, cy + hy))
        else:
            out.append((cx - hy, cx + hy, cy + c - line / 2.0, cy + c + line / 2.0))
    return np.asarray(out, dtype=float)


def vernier_band_offset_um(ply_um: float, fold_um: float) -> float:
    """Comb-center distance from the STACK (outer ply) edge.

    Centered in the interior foil-fold band [ply, ply + fold] — the ring where
    BOTH plies have glass and the interior fold hides the marks from inside.
    """
    return ply_um + fold_um / 2.0


def vernier_blocks(
    stack_w_um: float,
    stack_h_um: float,
    ply_um: float,
    fold_um: float,
    pitch_um: float,
    *,
    n_lines: int = VERNIER_LINES,
) -> np.ndarray:
    """Assembly-vernier combs for ONE sub-plate, in STACK-centered coords.

    Both plies of a pair receive combs at IDENTICAL stack coordinates (their
    frames share the center — the inner ply is concentric), differing only in
    pitch, so the stacked pair beats. Layout is mirror-symmetric in x so the
    panel's stack mirror maps it onto itself:

      * two X-sense combs (vertical bars) along the TOP edge, near each corner
        — differential fringe reads ROTATION over the plate width;
      * two Y-sense combs (horizontal bars) on the LEFT/RIGHT edges near the
        bottom corners — differential reads rotation about the other axis.

    Combs live in the interior fold band (see ``vernier_band_offset_um``),
    end at least ``VERNIER_CORNER_CLEAR_UM`` from the INNER ply corner, shrink
    (fewer lines) on tiny plates, and drop entirely below ``VERNIER_MIN_LINES``
    or when the fold is too narrow for the bar length.
    """
    if fold_um < VERNIER_BAR_LEN_UM * 0.8:
        return np.zeros((0, 4), dtype=float)
    off = vernier_band_offset_um(ply_um, fold_um)
    # Along-edge span must stay inside the INNER ply with corner clearance.
    inner_hw = stack_w_um / 2.0 - ply_um
    inner_hh = stack_h_um / 2.0 - ply_um
    blocks: list[np.ndarray] = []

    def _fit_lines(run_um: float) -> int:
        return min(n_lines, int(run_um // pitch_um))

    # X-sense combs, top band.
    cy_top = stack_h_um / 2.0 - off
    half_run = inner_hw - VERNIER_CORNER_CLEAR_UM - off
    n_x = _fit_lines(half_run)
    if n_x >= VERNIER_MIN_LINES:
        comb_w = (n_x - 1) * pitch_um
        cx_off = inner_hw - VERNIER_CORNER_CLEAR_UM - comb_w / 2.0
        for sx in (-1.0, 1.0):
            blocks.append(_comb_rects(sx * cx_off, cy_top, pitch_um, n_x, vertical_bars=True))

    # Y-sense combs, left/right bands, near the bottom corners.
    cx_side = stack_w_um / 2.0 - off
    half_run_y = inner_hh - VERNIER_CORNER_CLEAR_UM - off
    n_y = _fit_lines(half_run_y)
    if n_y >= VERNIER_MIN_LINES:
        comb_h = (n_y - 1) * pitch_um
        cy_off = -(inner_hh - VERNIER_CORNER_CLEAR_UM - comb_h / 2.0)
        for sx in (-1.0, 1.0):
            blocks.append(_comb_rects(sx * cx_side, cy_off, pitch_um, n_y, vertical_bars=False))

    if not blocks:
        return np.zeros((0, 4), dtype=float)
    return np.concatenate(blocks, axis=0)


def id_tick_rects(
    face_index: int,
    is_back: bool,
    stack_w_um: float,
    stack_h_um: float,
    ply_um: float,
    fold_um: float,
) -> np.ndarray:
    """Tick-code plate ID near the bottom edge, in the same both-plies band as
    the verniers: ``face_index + 1`` bars, plus a long underline bar for the B
    (inner) sub-plate. Diced plates all look alike under a loupe — this keeps
    the bench build sane. Stack-centered coords, identical on both plies."""
    n = face_index + 1
    cy = -(stack_h_um / 2.0 - vernier_band_offset_um(ply_um, fold_um))
    total = (n - 1) * ID_TICK_PITCH_UM
    out = []
    for i in range(n):
        cx = -total / 2.0 + i * ID_TICK_PITCH_UM
        out.append(
            (cx - ID_TICK_W_UM / 2.0, cx + ID_TICK_W_UM / 2.0,
             cy - ID_TICK_LEN_UM / 2.0, cy + ID_TICK_LEN_UM / 2.0)
        )
    if is_back:
        half = (total + ID_TICK_PITCH_UM) / 2.0
        y0 = cy + ID_TICK_LEN_UM / 2.0 + ID_TICK_W_UM
        out.append((-half, half, y0, y0 + ID_TICK_W_UM))
    return np.asarray(out, dtype=float)


def dice_tick_rects(p: Placement) -> np.ndarray:
    """L-shaped scribe ticks just OUTSIDE each corner of a placed sub-plate,
    inside the dicing street — blank-frame coords. Doubles as a post-dice
    orientation reference (the L opens toward the plate)."""
    out = []
    for sx, cx in ((-1.0, p.x0), (1.0, p.x0 + p.width_um)):
        for sy, cy in ((-1.0, p.y0), (1.0, p.y0 + p.height_um)):
            gx = cx + sx * DICE_TICK_GAP_UM
            gy = cy + sy * DICE_TICK_GAP_UM
            out.append((
                min(gx, gx + sx * DICE_TICK_W_UM), max(gx, gx + sx * DICE_TICK_W_UM),
                min(gy, gy + sy * DICE_TICK_LEN_UM), max(gy, gy + sy * DICE_TICK_LEN_UM),
            ))
            out.append((
                min(gx, gx + sx * DICE_TICK_LEN_UM), max(gx, gx + sx * DICE_TICK_LEN_UM),
                min(gy, gy + sy * DICE_TICK_W_UM), max(gy, gy + sy * DICE_TICK_W_UM),
            ))
    return np.asarray(out, dtype=float)


# --- geometry transforms -----------------------------------------------------------

def mirror_rects(rects: np.ndarray) -> np.ndarray:
    """(N,4) [x0,x1,y0,y1] mirrored about the plate's vertical axis (x → −x)."""
    if rects.size == 0:
        return rects
    out = rects.copy()
    out[:, 0] = -rects[:, 1]
    out[:, 1] = -rects[:, 0]
    return out


def _transform_rects(rects: np.ndarray, *, mirror: bool, rotated: bool) -> np.ndarray:
    if mirror:
        rects = mirror_rects(rects)
    if rotated and rects.size:
        # +90° CCW: (x, y) → (−y, x); rect [x0,x1,y0,y1] → [−y1,−y0,x0,x1].
        rects = np.stack(
            [-rects[:, 3], -rects[:, 2], rects[:, 0], rects[:, 1]], axis=1
        )
    return rects


def _transform_verts(verts: np.ndarray, *, mirror: bool, rotated: bool) -> np.ndarray:
    v = verts
    if mirror:
        v = np.stack([-v[:, 0], v[:, 1]], axis=1)
    if rotated:
        v = np.stack([-v[:, 1], v[:, 0]], axis=1)
    return v


# --- GDS build -------------------------------------------------------------------

def build_blank_gds(
    out_path: Path,
    *,
    plate_thickness_um: float = PLATE_THICKNESS_UM,
    blank_side_um: float = BLANK_SIDE_UM,
    aspect_h_over_w: float = DEFAULT_ASPECT_H_OVER_W,
    glass_n: float = BLANK_GLASS_N,
    glass_material: str = BLANK_GLASS_MATERIAL,
    spare_faces: tuple[str, ...] = DEFAULT_SPARE_FACES,
    mirror_for_stack: bool = True,
    drc_report_path: Path | None = None,
    _build_plate_fine: Any = None,   # test seam: fake fine-geometry builder
) -> dict[str, Any]:
    """The one-blank deliverable: 12 sub-plates, single chrome layer, one write.

    Per face, ``export_fine.build_plate_fine`` (glass-derived periods, merged
    DRC heal) runs ONCE; its front layer lands on the ``face:F`` (outer-ply)
    sub-plate and its back layer on ``face:B`` (inner-ply, smaller cut), each
    mirrored for the chrome-down bonded stack (see module docstring), rotated
    if the packer rotated that footprint, and translated into place. Assembly
    verniers (pitch split F/B at identical stack coordinates), tick-code IDs
    and corner dicing ticks are added to every sub-plate. Heavy compute — run
    alone.
    """
    import klayout.db as kdb

    from .assembly import FACE_IDS, bonded_overlap_um
    from .export_fine import DBU_UM, _rects_to_gds_polys, _verts_to_gds_poly

    if _build_plate_fine is None:
        from .export_fine import build_plate_fine as _build_plate_fine

    result = solve_blank_max_scale(
        plate_thickness_um=plate_thickness_um,
        blank_side_um=blank_side_um,
        aspect_h_over_w=aspect_h_over_w,
        spare_faces=spare_faces,
    )
    if result is None:
        raise SystemExit("no box fits the blank — check constants")
    spec = blank_box_spec(result, glass_n=glass_n, glass_material=glass_material)
    by_id = {p.face: p for p in result.placements}
    ply = result.plate_thickness_um
    fold = bonded_overlap_um(spec.foil, ply)

    ly = kdb.Layout()
    ly.dbu = DBU_UM
    top = ly.create_cell("blank_bonded_pairs")
    L_chrome = ly.layer(*LAYER_CHROME)
    L_outline = ly.layer(*LAYER_OUTLINE)
    L_blank = ly.layer(*LAYER_BLANK)
    L_label = ly.layer(*LAYER_LABEL)

    half = blank_side_um / 2.0
    uhalf = result.usable_side_um / 2.0
    top.shapes(L_blank).insert(kdb.DBox(-half, -half, half, half))
    top.shapes(L_blank).insert(kdb.DBox(-uhalf, -uhalf, uhalf, uhalf))

    report: list[dict[str, Any]] = []
    drc_faces: dict[str, Any] = {}
    total_polys = 0

    for face_index, fid in enumerate(FACE_IDS):
        plate_spec = spec.faces.get(fid)
        if plate_spec is None:
            continue
        fine = _build_plate_fine(plate_spec, fid)
        if getattr(fine, "stats", None) and fine.stats.get("drc"):
            drc_faces[fid] = fine.stats["drc"]

        # Stack dims = the OUTER ply cut dims (shared mark coordinates).
        stack_w = plate_spec.width_um
        stack_h = plate_spec.height_um

        # Original placement plus any :spareN duplicates — identical geometry,
        # marks and all (spares are bench-interchangeable with the original).
        for suf, polys in (("F", fine.front_polys), ("B", fine.back_polys)):
          sid = subplate_id(fid, suf)
          for p in [by_id.get(sid)] + [
              q for q in result.placements if q.face.startswith(sid + ":spare")
          ]:
            if p is None:
                continue
            dx, dy = p.cx, p.cy
            n = 0
            for verts in polys:
                if verts.shape[0] < 3:
                    continue
                v = _transform_verts(
                    np.asarray(verts, dtype=float),
                    mirror=mirror_for_stack, rotated=p.rotated,
                )
                top.shapes(L_chrome).insert(_verts_to_gds_poly(v, dx, dy))
                n += 1

            pitch = VERNIER_PITCH_F_UM if suf == "F" else VERNIER_PITCH_B_UM
            marks = [
                vernier_blocks(stack_w, stack_h, ply, fold, pitch),
                id_tick_rects(face_index, suf == "B", stack_w, stack_h, ply, fold),
            ]
            for rects in marks:
                rects = _transform_rects(rects, mirror=mirror_for_stack, rotated=p.rotated)
                for poly in _rects_to_gds_polys(rects, dx, dy):
                    top.shapes(L_chrome).insert(poly)
                n += rects.shape[0]

            # Dicing ticks live in the street (blank frame — no transform).
            ticks = dice_tick_rects(p)
            for poly in _rects_to_gds_polys(ticks, 0.0, 0.0):
                top.shapes(L_chrome).insert(poly)
            n += ticks.shape[0]

            hw, hh = p.width_um / 2.0, p.height_um / 2.0
            top.shapes(L_outline).insert(kdb.DBox(p.cx - hw, p.cy - hh, p.cx + hw, p.cy + hh))
            top.shapes(L_label).insert(
                kdb.DText(p.face, kdb.DTrans(kdb.DVector(p.cx, p.cy)))
            )
            total_polys += n
            report.append({
                "plate": p.face, "slug": fine.slug,
                "polys": n, "rotated": p.rotated,
            })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(out_path))

    if drc_report_path is not None:
        import json

        drc_report_path = Path(drc_report_path)
        drc_report_path.parent.mkdir(parents=True, exist_ok=True)
        drc_report_path.write_text(
            json.dumps(drc_faces, indent=2, default=float), encoding="utf-8"
        )

    from .plates import (
        _carrier_recipe_data,
        fab_center_period_um,
        parallax_gap_um,
        water_scan_fab_pitch_um,
    )

    any_face = next(iter(spec.faces.values()))
    amplification = VERNIER_PITCH_B_UM / (VERNIER_PITCH_B_UM - VERNIER_PITCH_F_UM)
    beat_um = VERNIER_PITCH_F_UM * VERNIER_PITCH_B_UM / (VERNIER_PITCH_B_UM - VERNIER_PITCH_F_UM)
    return {
        "gds_path": str(out_path),
        "total_polygons": total_polys,
        "blank": {
            "side_um": blank_side_um,
            "usable_side_um": result.usable_side_um,
            "street_um": BLANK_STREET_UM,
            "single_chrome_layer": LAYER_CHROME,
        },
        "box": result.dims_mm(),
        "spare_pairs": list(spare_faces),
        "glass": {
            "material": glass_material,
            "n": glass_n,
            "ply_um": ply,
            "parallax_gap_um": round(parallax_gap_um(any_face), 1),
        },
        "optics": {
            # Glass-derived fab periods actually baked for THIS stock.
            "switch_barrier_period_um": fab_center_period_um(any_face),
            "scanimation_frame_pitch_um": water_scan_fab_pitch_um(any_face),
            # Effective carrier per face (PlateSpec.carrier_scale_mode: 'gap'
            # faces scale with t/n for controlled reveals; 'fixed' faces keep
            # the fine design pitch for refraction shimmer).
            "carrier_period_um": {
                fid: _carrier_recipe_data(f)["fab_back_period_um"]
                for fid, f in spec.faces.items()
            },
            "fixed_accents_um": [24.0, 4.4],  # body shimmer, rainbow grating
        },
        "stack": {
            "mirrored_for_stack": mirror_for_stack,
            "wall_thickness_um": result.wall_thickness_um,
            "inner_ply_inset_um": ply,
            "assembly": (
                "Both sub-plates assemble chrome-DOWN (F chrome at the bond "
                "interface, B chrome toward the box interior); artwork is "
                "pre-mirrored so the viewer sees the design frame. The inner "
                "(B) ply is one ply smaller per edge — bond it CENTERED on the "
                "outer ply; the nested-shell corners then interleave as the "
                "45-deg-approximating staircase."
            ),
        },
        "verniers": {
            "pitch_front_um": VERNIER_PITCH_F_UM,
            "pitch_back_um": VERNIER_PITCH_B_UM,
            "amplification": round(amplification, 2),
            "beat_um": round(beat_um, 1),
            "band_offset_um": vernier_band_offset_um(ply, fold),
        },
        "plates": report,
    }


# --- layout preview SVG ---------------------------------------------------------

def write_blank_layout_svg(
    placements: list[Placement],
    out_path: Path,
    *,
    blank_side_um: float = BLANK_SIDE_UM,
    usable_side_um: float | None = None,
) -> Path:
    """Blank-layout preview: outlines + labels only (the square sibling of
    ``export_wafer.write_layout_svg``)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if usable_side_um is None:
        usable_side_um = blank_side_um - 2.0 * BLANK_EDGE_MARGIN_UM

    H = blank_side_um / 2.0
    U = usable_side_um / 2.0
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" '
        f'viewBox="{-H:.1f} {-H:.1f} {blank_side_um:.1f} {blank_side_um:.1f}">',
        f'<rect x="{-H:.1f}" y="{-H:.1f}" width="{blank_side_um:.1f}" '
        f'height="{blank_side_um:.1f}" fill="#1b1d24" stroke="#4a4f5a" stroke-width="300"/>',
        f'<rect x="{-U:.1f}" y="{-U:.1f}" width="{usable_side_um:.1f}" '
        f'height="{usable_side_um:.1f}" fill="none" stroke="#6b7280" '
        f'stroke-width="200" stroke-dasharray="1500 900"/>',
    ]
    label_px = max(1200.0, usable_side_um / 24.0)
    for p in placements:
        x, y = p.x0, -(p.y0 + p.height_um)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{p.width_um:.1f}" '
            f'height="{p.height_um:.1f}" fill="#e3b53b" fill-opacity="0.18" '
            f'stroke="#e3b53b" stroke-width="150"/>'
        )
        parts.append(
            f'<text x="{p.cx:.1f}" y="{-p.cy + label_px / 3:.1f}" '
            f'font-family="sans-serif" font-size="{label_px:.0f}" '
            f'fill="#f4f4f5" text-anchor="middle">{p.face}'
            f'{"↻" if p.rotated else ""}</text>'
        )
    parts.append("</svg>")
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


# --- CLI --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Panelize all 6 faces as bonded-pair sub-plates on ONE square mask blank."
    )
    ap.add_argument("--out", default="data/blank/blank.gds", help="output GDS path")
    ap.add_argument("--plate-thickness", type=float, default=PLATE_THICKNESS_UM,
                    help="single PLY thickness in um (bonded wall is 2x; also the optical gap)")
    ap.add_argument("--blank-side", type=float, default=BLANK_SIDE_UM,
                    help="square blank side in um (5in=127000, 4in=101600)")
    ap.add_argument("--aspect", type=float, default=DEFAULT_ASPECT_H_OVER_W,
                    help="box H/W ratio (default 1.1 — upright ring box)")
    ap.add_argument("--glass-n", type=float, default=BLANK_GLASS_N,
                    help="refractive index of the stock (soda lime 1.52, fused silica 1.46)")
    ap.add_argument("--glass-material", default=BLANK_GLASS_MATERIAL)
    ap.add_argument("--spares", default=",".join(DEFAULT_SPARE_FACES),
                    help="comma-separated faces to duplicate as spare bonded "
                    "pairs (default 'front,back'); each pair shrinks the max "
                    "box (~30.5 mm at 0, ~28.7 mm at 2 on the 5in default). "
                    "Pass an empty string for no spares.")
    ap.add_argument("--no-mirror", action="store_true",
                    help="write artwork UN-mirrored (chrome-up assembly stack)")
    ap.add_argument("--pack-only", action="store_true",
                    help="pure-math packer + layout SVG, no pattern generation")
    ap.add_argument("--drc-report", default=None, help="per-face DRC report JSON path")
    args = ap.parse_args(argv)

    out = Path(args.out)
    svg_out = out.with_suffix(".svg")

    spare_faces = tuple(f.strip() for f in args.spares.split(",") if f.strip())
    result = solve_blank_max_scale(
        plate_thickness_um=args.plate_thickness,
        blank_side_um=args.blank_side,
        aspect_h_over_w=args.aspect,
        spare_faces=spare_faces,
    )
    if result is None:
        print("no fit")
        return 1
    write_blank_layout_svg(
        result.placements, svg_out,
        blank_side_um=args.blank_side, usable_side_um=result.usable_side_um,
    )
    print(f"box (mm): {result.dims_mm()}")
    if args.pack_only:
        for p in result.placements:
            print(
                f"  {p.face:9s} x0={p.x0:9.1f} y0={p.y0:9.1f} "
                f"{p.width_um:8.1f}x{p.height_um:8.1f} rot={p.rotated}"
            )
        print(f"svg -> {svg_out}")
        return 0

    summary = build_blank_gds(
        out,
        plate_thickness_um=args.plate_thickness,
        blank_side_um=args.blank_side,
        aspect_h_over_w=args.aspect,
        glass_n=args.glass_n,
        glass_material=args.glass_material,
        spare_faces=spare_faces,
        mirror_for_stack=not args.no_mirror,
        drc_report_path=Path(args.drc_report) if args.drc_report else None,
    )
    print(f"gds -> {summary['gds_path']} ({summary['total_polygons']} polygons)")
    print(f"svg -> {svg_out}")
    print(f"optics: {summary['optics']}")
    print(f"verniers: {summary['verniers']}")
    for r in summary["plates"]:
        print(f"  {r['plate']:9s} {r['slug']:22s} polys={r['polys']:6d} rot={r['rotated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
