"""Production dies on the witness plate: four box faces that come off as plies.

The 5″ plate is the stock the bonded box is built from (``witness_geom.PLY_UM``
of ``GLASS_MATERIAL``), so a rectangle of it written with a face's fine
geometry IS that face's ply once it is diced. Four faces ride along with the experiments:

    top    F + B   monogram-jp centerpiece + foliage garland, bonded pair
    front  F + B   globe-duo-phase barrier switch + garland, bonded pair
    left   F       colour-zoned halftone portrait (ZONES) + garland (both gratings on the one ply)
    right  F       the same portrait, HUE-mapped + garland

The two sides are single plies: their optics are all single-layer (a line
screen and period-ratio diffraction colour, §1.1 / §1.4 of the plan), so the
backing ply is bare glass and costs no plate area.

Polarity. The plate is a darkfield write with positive resist: the file holds
the openings, chrome stays wherever the file is empty. A face's fine geometry is
authored as METAL (gold where the rectangles are), so each die is inverted —
``die box − metal`` — and that inversion runs as a klayout Region boolean on the
one die, never a GEOS union (the same C++ edge set the merged-DRC heal already
builds for every face). The result is decomposed to trapezoids so no polygon
carries holes or more vertices than a mask shop will take.

Mirroring. Both plies of a face are written MIRRORED (x → −x about the die
centre), exactly as ``export_blank`` does: the chrome faces the bond, the
viewer looks through the glass, and the flip at assembly restores the design
frame. The assembly verniers (80 µm on F, 88 µm on B) and the tick-code ID sit
at identical stack coordinates on both plies of a pair, inside the interior
foil-fold band, so the pair beats when stacked; the F-only sides carry the F
comb so a bare backing ply can still be squared to them.

Everything is in micrometres, plate-centred, y up — the ``CellArt`` contract
of ``export_witness``.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from . import export_blank as eb
from . import witness_cells as wc
from .patterns.bitmap import colourplan as cp
from .witness_geom import (CLEAR, GLASS_MATERIAL, GLASS_N, METAL, PLY_UM, CellArt,
                           Cell, _cat)

# The plate IS the box stock (witness_geom.PLY_UM / GLASS_N). Every derived
# number on the witness -- parallax rate, near-field boundary, comb pitch -- is
# therefore the box's own.

SIDE_MODES: dict[str, str] = {"left": "zones", "right": "hue"}
"""Which colour treatment each side carries. Both are the Paula portrait; the
subjective choice between them is made on the box, not on a test cell."""

# Garland colour: a diffraction-colour sub-grating per motif family, so every
# leaf and flower of the frame carries its own hue the way the portrait's zones
# do. Hues in degrees on the same circle ``colourplan.hue_to_scale`` reads
# (red = long period = 6.0 µm at spread 1.45, blue = short = 4.15 µm; the
# magenta end folds to red at HUE_WRAP_DEG). Orchids sit past the fold and come
# out at the violet end, which is what an orchid is.
MOTIF_HUE_DEG: dict[str, float] = {
    "vine": 120.0,
    "anthurium": 0.0,
    "coffee": 20.0,
    "heliconia": 40.0,
    "orchid": 290.0,
    "wax_palm": 100.0,
    "plantain": 120.0,
    "fern": 140.0,
    "philodendron": 160.0,
}
GARLAND_LEVEL0 = 10
"""Graylevel of the first garland colour rung in the rendered frame scene."""

GARLAND_MODE = "moire"
"""How the single-ply sides carry their garland.

``"moire"``: the leaf gratings AND the carrier on the one ply. By the union
identity (plan §2.1) the superposition of the two gratings on one plane is the
zero-gap two-ply pattern, so each leaf shows the same beat fringes the lid's
two-ply garland shows head-on — they simply do not travel with tilt. The
carrier covers the window outside the art box as it does on the bonded faces.
``"colour"``: a diffraction sub-grating per motif family instead (flat leaves
whose hue moves with tilt)."""

TRAPEZOID_DECOMP = True

MOTIF_SCALE = 0.75
"""Garland leaf/flower size on the production faces, as a fraction of the
design's. The band and vine keep their width; the grower packs motifs
proportionally denser. Chosen on a 24.6 mm side, where the design's 13% band
with 1.6 mm leaves read too heavy against a 15 mm portrait."""


# --- the box this plate feeds -------------------------------------------------


@lru_cache(maxsize=1)
def blank_plan() -> tuple[Any, Any]:
    """``(BlankPlanResult, BoxSpec)`` of the largest bonded box whose twelve
    sub-plates fit one 5″ blank at this ply — the SAME solve ``export_blank``
    makes, so a die here is interchangeable with one from the full panel."""
    result = eb.solve_blank_max_scale(plate_thickness_um=PLY_UM)
    if result is None:
        raise RuntimeError("no bonded box fits the blank at this ply")
    spec = eb.blank_box_spec(result, glass_n=GLASS_N, glass_material=GLASS_MATERIAL)
    import dataclasses
    for fid, ps in spec.faces.items():
        ps.frame = dataclasses.replace(ps.frame, motif_scale=MOTIF_SCALE)
    return result, spec


def die_dims(face: str) -> dict[str, float]:
    """Cut dimensions (µm) of a face's outer (F) and inner (B) plies."""
    result, _ = blank_plan()
    dims = {r.face: (r.width_um, r.height_um)
            for r in eb.pair_rects(result.width_um, result.depth_um,
                                   result.height_um, result.plate_thickness_um)}
    fw, fh = dims[eb.subplate_id(face, "F")]
    bw, bh = dims[eb.subplate_id(face, "B")]
    return {"f_w": fw, "f_h": fh, "b_w": bw, "b_h": bh}


def _fold_um() -> float:
    from .assembly import bonded_overlap_um

    _, spec = blank_plan()
    return bonded_overlap_um(spec.foil, PLY_UM)


def bench_marks(face: str, ply: str, stack_w: float, stack_h: float) -> np.ndarray:
    """Vernier combs + tick-code ID for one ply, stack-centred, METAL sense."""
    from .assembly import FACE_IDS

    pitch = eb.VERNIER_PITCH_F_UM if ply == "F" else eb.VERNIER_PITCH_B_UM
    fold = _fold_um()
    return _cat(
        eb.vernier_blocks(stack_w, stack_h, PLY_UM, fold, pitch),
        eb.id_tick_rects(list(FACE_IDS).index(face), ply == "B",
                         stack_w, stack_h, PLY_UM, fold),
    )


def dice_ticks(w: float, h: float) -> np.ndarray:
    """Corner L-ticks in the street around a ``w × h`` die centred at the
    origin. Emitted as DATA in either polarity: in the darkfield write the
    street is chrome and a clear L is what a scribe can see."""
    return eb.dice_tick_rects(eb.Placement("die", -w / 2.0, -h / 2.0, w, h, False))


# --- inversion ----------------------------------------------------------------


def clear_field(w: float, h: float, metal: list[np.ndarray], *,
                dbu_um: float = 0.001) -> list[np.ndarray]:
    """``die box − metal`` as hole-free polygons, origin-centred.

    ``metal`` mixes ``(N,4)`` rect arrays and ``(K,2)`` vertex rings. One
    klayout Region boolean over the die, then a trapezoid decomposition so the
    writer never sees a polygon with holes or a 100k-vertex ring.
    """
    from .patterns.effects.drc import _region_from_polys

    reg, kdb = _region_from_polys(metal, dbu_um)
    s = 1.0 / dbu_um
    box = kdb.Region(kdb.Box(int(round(-w / 2 * s)), int(round(-h / 2 * s)),
                             int(round(w / 2 * s)), int(round(h / 2 * s))))
    clear = box - reg
    clear.merge()
    if TRAPEZOID_DECOMP:
        clear = clear.decompose_trapezoids_to_region()
    out: list[np.ndarray] = []
    for poly in clear.each():
        pts = [(pt.x * dbu_um, pt.y * dbu_um) for pt in poly.each_point_hull()]
        if len(pts) >= 3:
            out.append(np.asarray(pts, dtype=np.float64))
    return out


def _mirror_polys(polys: list[np.ndarray]) -> list[np.ndarray]:
    return [eb._transform_verts(np.asarray(p, dtype=np.float64), mirror=True, rotated=False)
            for p in polys]


def _shift_polys(polys: list[np.ndarray], dx: float, dy: float) -> list[np.ndarray]:
    return [np.asarray(p, dtype=np.float64) + np.array([dx, dy]) for p in polys]


def _shift_rects(r: np.ndarray, dx: float, dy: float) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    if r.size == 0:
        return np.empty((0, 4))
    out = r.copy()
    out[:, 0] += dx
    out[:, 1] += dx
    out[:, 2] += dy
    out[:, 3] += dy
    return out


def _ply_art(w: float, h: float, metal: list[np.ndarray], polarity: str,
             ) -> tuple[np.ndarray, list[np.ndarray]]:
    """One ply's written geometry, origin-centred and mirrored for the stack:
    ``(rects, polys)`` — the metal itself, or its clear-field complement."""
    if polarity == METAL:
        rects = [np.asarray(m) for m in metal if np.asarray(m).ndim == 2 and np.asarray(m).shape[1] == 4]
        polys = [np.asarray(m) for m in metal if np.asarray(m).ndim == 2 and np.asarray(m).shape[1] == 2]
        return eb.mirror_rects(_cat(*rects)) if rects else np.empty((0, 4)), _mirror_polys(polys)
    return np.empty((0, 4)), _mirror_polys(clear_field(w, h, metal))


# --- bonded faces: monogram lid, globe front ----------------------------------


def build_face_die(face: str, cx: float, cy: float, w: float, h: float,
                   polarity: str = METAL) -> CellArt:
    """A bonded face as two dies: F (outer ply, ``w × h``) at ``(cx, cy)`` and
    B (inner ply) at the SAME centre — ``export_witness.build_plate`` moves the
    back die to its pair position. Fine geometry from
    ``export_fine.build_plate_fine`` at the box's glass-derived periods, merged
    DRC heal included."""
    from .export_fine import build_plate_fine

    _, spec = blank_plan()
    pspec = spec.faces[face]
    d = die_dims(face)
    if abs(d["f_w"] - w) > 1.0 or abs(d["f_h"] - h) > 1.0:
        raise ValueError(f"{face}: cell is {w:.0f}x{h:.0f} but the F ply cuts "
                         f"{d['f_w']:.0f}x{d['f_h']:.0f}")
    fine = build_plate_fine(pspec, face)

    metal_f: list[np.ndarray] = list(fine.front_polys) + [bench_marks(face, "F", w, h)]
    metal_b: list[np.ndarray] = list(fine.back_polys) + [bench_marks(face, "B", w, h)]
    fr, fp = _ply_art(w, h, metal_f, polarity)
    br, bp = _ply_art(d["b_w"], d["b_h"], metal_b, polarity)

    art = CellArt()
    art.front = _shift_rects(_cat(fr, dice_ticks(w, h)), cx, cy)
    art.polys = _shift_polys(fp, cx, cy)
    art.back = _shift_rects(_cat(br, dice_ticks(d["b_w"], d["b_h"])), cx, cy)
    art.back_polys = _shift_polys(bp, cx, cy)
    drc = fine.stats.get("drc", {})
    art.stats = {
        "polarity": polarity, "face": face, "slug": pspec.pattern_slug,
        "ply_um": PLY_UM, "glass_n": GLASS_N, "mirrored": True,
        "f_um": [w, h], "b_um": [d["b_w"], d["b_h"]],
        "periods": fine.stats.get("periods", {}),
        "metal_polys_front": len(fine.front_polys),
        "metal_polys_back": len(fine.back_polys),
        "written_polys_front": len(art.polys),
        "written_polys_back": len(art.back_polys),
        "drc_front": drc.get("front_merged_after"),
        "drc_back": drc.get("back_merged_after"),
        "single_layer": False,
    }
    return art


# --- colour sides: portrait + colour garland, one ply -------------------------


def _garland_levels(pspec: Any, pitch_um: float) -> tuple[np.ndarray, list[float]]:
    """Render the face's frame scene with a COLOUR rung per motif family.

    Returns the ``(h_px, w_px)`` level grid (0 = no frame) and the list of
    sub-grating periods, indexed by ``level − GARLAND_LEVEL0``.
    """
    from . import plates as P
    from .export_fine import _mask_rim
    from .patterns.frames import RectFrame, render_scene_to_image

    plan = cp.PAULA_ZONES
    keys = sorted(MOTIF_HUE_DEG)
    periods = [round(float(plan.base_period_um
                           * cp.hue_to_scale(np.array(MOTIF_HUE_DEG[k]), plan.spread)), 2)
               for k in keys]
    rung = {k: i for i, k in enumerate(keys)}

    def level_fn(key: str) -> int:
        return GARLAND_LEVEL0 + rung.get(key, rung["vine"])

    W, H = pspec.width_um, pspec.height_um
    fw, fh = max(1, int(round(W / pitch_um))), max(1, int(round(H / pitch_um)))
    levels = np.zeros((fh, fw), dtype=np.uint8)
    aw_um, ah_um = pspec.active_dims()
    if aw_um > 0 and ah_um > 0:
        rect = RectFrame(width_um=aw_um, height_um=ah_um)
        fp = pspec.frame.to_frame_params()
        fp.fill_interior = False
        pid = P.plate_hash(pspec)
        scene = P.frame_scene_for_plate(P.PLATES_ROOT / pid, P.get_plate(pid), rect, fp)
        img = render_scene_to_image(scene, rect, fp, pitch_um, level_fn=level_fn)
        aw, ah = min(fw, img.size[0]), min(fh, img.size[1])
        ox, oy = (fw - aw) // 2, (fh - ah) // 2
        levels[oy:oy + ah, ox:ox + aw] = np.asarray(img)[:ah, :aw]
        keep = levels > 0
        _mask_rim(keep, pspec.weld_margin_um, pitch_um)
        levels[~keep] = 0
    return levels, periods


def _static_garland_metal(pspec: Any, w: float, h: float, pitch: float) -> list[np.ndarray]:
    """The sides' garland as METAL polygons on one ply: every leaf filled with
    the front grating at its angle bucket, the carrier over the window outside
    the art box, unioned and healed to the litho floor exactly as the fine
    builder heals a two-ply face's front layer. Plate-centred, unmirrored."""
    from . import plates as P
    from .export_fine import (LITHO_FLOOR_UM, _build_zone_masks, _compose_layer_polys,
                              _concat_rects, _drc_rects, _emit_grating, _erode_zone)
    from .patterns.effects.drc import drc_clean_region

    rd = P._carrier_recipe_data(pspec)
    zm = _build_zone_masks(pspec, pitch)
    rects: list[np.ndarray] = []
    angled: list[tuple[np.ndarray, float]] = []
    n_b = int(round(float(rd["frame_bucket_count"])))
    span = float(rd["frame_angle_span_deg"])
    duty = float(rd["grating_duty"])
    for b in range(n_b):
        lo = P.FRAME_BUCKET0 + b * P.FRAME_BUCKET_STEP - P.FRAME_BUCKET_STEP // 2
        hi = P.FRAME_BUCKET0 + b * P.FRAME_BUCKET_STEP + P.FRAME_BUCKET_STEP // 2
        in_b = _erode_zone((zm.frame_level > max(0, lo)) & (zm.frame_level <= hi), 1)
        if in_b.any():
            ang = float(rd["slit_axis_deg"]) + (b - 0.5 * (n_b - 1)) * span
            _emit_grating(in_b, pitch, (w, h), float(rd["slit_period_um"]), duty, ang, 0.0, rects, angled)
    carrier_zone = zm.back_window.copy()
    if zm.art_box is not None:
        carrier_zone &= ~zm.art_box
    carrier_zone = _erode_zone(carrier_zone, 2)
    if carrier_zone.any():
        _emit_grating(carrier_zone, pitch, (w, h), float(rd["carrier_period_um"]), duty,
                      float(rd["carrier_angle_deg"]), 0.0, rects, angled)
    polys = _compose_layer_polys(_drc_rects(_concat_rects(rects)) if rects else np.empty((0, 4)),
                                 [(_drc_rects(r), a) for r, a in angled])
    return drc_clean_region(polys, min_width_um=LITHO_FLOOR_UM, min_gap_um=LITHO_FLOOR_UM)


def build_colour_side(face: str, cx: float, cy: float, w: float, h: float,
                      polarity: str = METAL, *, mode: str | None = None,
                      garland: str | None = None) -> CellArt:
    """A single-ply side: the portrait at the face's art box, a colour garland
    in the frame band, the F vernier and ID, all mirrored for the stack.

    The portrait is emitted by ``witness_cells.build_halftone`` in the requested
    polarity directly (its inverse is analytic and its sub-gratings stay array
    references); only the garland and the bare remainder of the die go through
    the Region inversion.
    """
    from . import plates as P
    from .export_fine import _clip_axis_grating, _erode_zone
    from .patterns._helpers import MAX_LATTICE_CELLS

    _, spec = blank_plan()
    pspec = spec.faces[face]
    d = die_dims(face)
    if abs(d["f_w"] - w) > 1.0 or abs(d["f_h"] - h) > 1.0:
        raise ValueError(f"{face}: cell is {w:.0f}x{h:.0f} but the F ply cuts "
                         f"{d['f_w']:.0f}x{d['f_h']:.0f}")
    mode = mode or SIDE_MODES[face]
    plan = {"plain": cp.PAULA_PLAIN, "hue": cp.PAULA_HUE, "zones": cp.PAULA_ZONES}[mode]

    garland_mode = garland or GARLAND_MODE
    # Boundary raster for the garland silhouettes — same rule as the fine
    # builder: fine enough for the leaf edges, capped by the lattice budget.
    pitch = max(10.0, math.sqrt(w * h / (0.9 * MAX_LATTICE_CELLS)))
    garland_rects: list[np.ndarray] = []
    garland_polys: list[np.ndarray] = []
    per_rung_rects: dict[float, int] = {}
    periods: list[float] = []
    if garland_mode == "moire":
        garland_polys = _static_garland_metal(pspec, w, h, pitch)
    else:
        levels, periods = _garland_levels(pspec, pitch)
        for i, per in enumerate(periods):
            zone = _erode_zone(levels == GARLAND_LEVEL0 + i, 1)
            if not zone.any():
                continue
            r = _clip_axis_grating(zone, pitch, (w, h), per, plan.duty, 0.0, phase=0.0)
            if r.shape[0]:
                garland_rects.append(r)
                per_rung_rects[per] = per_rung_rects.get(per, 0) + int(r.shape[0])

    # Portrait: the CENTERPIECE_FILL square of the aperture, at the die centre.
    side = P.CENTERPIECE_FILL * P._aperture(pspec)
    port = wc.build_halftone(0.0, 0.0, side, side, plan=plan, polarity=polarity)
    art_box = np.array([[-side / 2, side / 2, -side / 2, side / 2]])

    # Everything else on the ply: garland metal, bench marks, and the art box
    # (already fully written by the portrait, so it is excluded from the
    # remainder in either polarity).
    marks = bench_marks(face, "F", w, h)
    if polarity == METAL:
        rects = eb.mirror_rects(_cat(*garland_rects, marks)) if garland_rects else eb.mirror_rects(marks)
        polys: list[np.ndarray] = _mirror_polys(garland_polys)
    else:
        rects = np.empty((0, 4))
        polys = _mirror_polys(clear_field(w, h, garland_rects + garland_polys + [marks, art_box]))

    # Mirror the portrait too: rect edges swap about x = 0, and each array
    # band's stripe phase reflects — stripe k spanned [phase + k·d, +line], so
    # its image spans [−phase − k·d − line, +line], i.e. phase' = −phase − line.
    prect = eb.mirror_rects(port.front)
    arrays = []
    for a in port.arrays:
        rr = eb.mirror_rects(np.asarray(a["rects"], dtype=np.float64))
        line = np.asarray(a["line_um"], dtype=np.float64)
        ph = np.asarray(a.get("phase_um", 0.0), dtype=np.float64)
        arrays.append({**a, "rects": _shift_rects(rr, cx, cy),
                       "phase_um": -ph - line + cx})

    art = CellArt()
    art.front = _shift_rects(_cat(rects, prect, dice_ticks(w, h)), cx, cy)
    art.polys = _shift_polys(polys, cx, cy)
    art.arrays = arrays
    art.stats = {
        "polarity": polarity, "face": face, "mode": mode, "slug": "portrait-" + mode,
        "ply_um": PLY_UM, "glass_n": GLASS_N, "mirrored": True,
        "f_um": [w, h], "portrait_um": round(side, 1),
        "garland_pitch_um": round(pitch, 2),
        "garland_mode": garland_mode,
        "garland_metal_polys": len(garland_polys),
        "garland_hue_deg": {k: MOTIF_HUE_DEG[k] for k in sorted(MOTIF_HUE_DEG)} if periods else {},
        "garland_periods_um": {k: periods[i] for i, k in enumerate(sorted(MOTIF_HUE_DEG))} if periods else {},
        "garland_rects_by_period": {f"{k:g}": v for k, v in per_rung_rects.items()},
        "portrait": port.stats,
        "written_polys": len(art.polys),
        "single_layer": True,
    }
    return art


# --- the cells ------------------------------------------------------------------


def production_cells() -> list[Cell]:
    """The four faces as plate cells, in layout order. Sized from the blank
    solve so they cannot drift from the panelized box."""
    MM = 1000.0
    _, spec = blank_plan()
    cells: list[Cell] = []
    for face, title in (("top", "lid: monogram + garland"),
                        ("front", "front: globe switch + garland")):
        d = die_dims(face)
        slug = spec.faces[face].pattern_slug
        cells.append(Cell(
            cid=f"DIE-{face.upper()}", title=title, group="X",
            w_um=d["f_w"], h_um=d["f_h"],
            back_w_um=d["b_w"], back_h_um=d["b_h"],
            build=(lambda face=face: (lambda cx, cy, w, h, polarity=METAL:
                   build_face_die(face, cx, cy, w, h, polarity)))(),
            label=f"{face.upper()} F {slug}", block="production",
            two_layer=True, takes_polarity=True,
            axis="production die", level=f"{face} F+B",
            note=(f"bonded pair, {d['f_w']/MM:.1f} mm outer / {d['b_w']/MM:.1f} mm inner ply; "
                  "mirrored for the chrome-down stack; 80/88 um verniers in the fold band")))
    for face in ("left", "right"):
        d = die_dims(face)
        mode = SIDE_MODES[face]
        cells.append(Cell(
            cid=f"DIE-{face.upper()}", title=f"{face}: colour portrait ({mode}) + colour garland",
            group="X", w_um=d["f_w"], h_um=d["f_h"],
            build=(lambda face=face: (lambda cx, cy, w, h, polarity=METAL:
                   build_colour_side(face, cx, cy, w, h, polarity)))(),
            label=f"{face.upper()} F colour {mode.upper()}", block="production",
            takes_polarity=True, axis="production die", level=f"{face} F ({mode})",
            note="single ply; backing ply is bare glass. Portrait + garland are "
                 "single-layer: line screen and period-ratio diffraction colour"))
    return cells
