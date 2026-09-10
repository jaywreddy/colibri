"""Production dies on the witness plate: four box faces that come off as plies.

The 5″ plate is the stock the bonded box is built from (``witness_geom.PLY_UM``
of ``GLASS_MATERIAL``), so a rectangle of it written with a face's fine
geometry IS that face's ply once it is diced. Four faces ride along with the experiments:

    top    F + B   monogram-jp centerpiece + foliage garland, bonded pair
    front  F + B   globe-duo-phase barrier switch + garland, bonded pair
    left   F       the beach photograph as a line screen (faces coloured) + garland
    right  F       the sunset photograph, plain + garland

Every die is the face's own ``PlateSpec`` from ``boxes.default_box_spec`` (via
``blank_plan``) put through ``export_fine.build_plate_fine`` — the same
geometry the box's GDS bake writes — so nothing here is a second authoring
path. The two sides are single plies: a halftone is a single-layer effect, so
their garland carries both gratings (leaves and a lighter carrier) on the one
ply and the backing ply is bare glass that costs no plate area.

Polarity. The plate is a darkfield write with positive resist: the file holds
the openings, chrome stays wherever the file is empty. A face's fine geometry is
authored as METAL (gold where the rectangles are), so each die is inverted —
``die box − metal`` — and that inversion runs as a klayout Region boolean on the
one die, never a GEOS union (the same C++ edge set the merged-DRC heal already
builds for every face). Before and after the inversion the geometry is opened
to the 2 µm floor and checked whole (``clear_field``), so the WRITTEN clear
data — not just the authored metal — passes a shop's incoming width/space
rule; the result is decomposed to convex pieces so no polygon carries holes or
more vertices than a mask shop will take.

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
from .witness_geom import (CLEAR, GLASS_MATERIAL, GLASS_N, METAL, PLY_UM, CellArt,
                           Cell, _cat)

# The plate IS the box stock (witness_geom.PLY_UM / GLASS_N). Every derived
# number on the witness -- parallax rate, near-field boundary, comb pitch -- is
# therefore the box's own.

TRAPEZOID_DECOMP = True

# The garland dials the plate is written at are the box's own
# (boxes.PRODUCTION_MOTIF_SCALE / PRODUCTION_BAND_UM); MOTIF_SCALE is kept as a
# name for the tools and tests that read it here.
from .boxes import PRODUCTION_MOTIF_SCALE as MOTIF_SCALE  # noqa: E402


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
    # motif_scale / band_um / foil already come from boxes.default_box_spec — the
    # production dials — so the die IS the box face, with nothing re-applied here.
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


CLEAR_FLOOR_UM = 2.0
"""The shop's rule, both senses: no written CLEAR feature and no CHROME left
between clear features narrower than this."""

FINISH_ART_UM = CLEAR_FLOOR_UM / 2.0
FINISH_FRAME_UM = 1.2
"""Half-widths of the morphological OPENs that finish a die (see
``clear_field``): the METAL is opened (eroded then dilated) so no chrome
filament or tip thinner than the floor survives, then the CLEAR complement is
opened so no clear slit, pinch or DBU sliver does. Inside the centrepiece ART
BOX the radius is exactly the half-floor: the finest colour sub-grating leaves
2.075 µm of clear and 2.075 µm of chrome, which survive a 1.0 µm erosion at
75 nm and regrow. In the FRAME the radius is 1.2 µm: nothing there is finer
than a 40 µm vernier bar, and where a single-ply garland's two gratings cross
at 2.5° the clear gap tapers to a wedge that the open must cap — klayout
rebuilds the cap as a flat cut whose width is 2r less the wedge taper, so at
r = 1.0 the caps came out 1.91 µm and were flagged; at 1.2 they are 2.3 µm.
What the opens leave (a jagged weld pocket around a colour-stripe end from
the metal-side heal, a decomposition sliver) is chromed over where the width
check flags it, in at most two monotone rounds, and the remainder is counted in
the die's ``drc_written_*`` stats — a few sub-micron pockets that resist will
not resolve anyway, listed rather than hidden."""


def _drc_checks(reg, kdb, floor_dbu: int):
    """(width, space) EdgePairs at the floor: Euclidian, edges at 80° or more
    ignored, zero-distance touches excluded — the incoming check a mask shop
    runs."""
    wv = reg.width_check(floor_dbu, False, kdb.Region.Euclidian, 80, None, None,
                         True, False, kdb.Region.IgnoreProperties,
                         kdb.Region.NeverIncludeZeroDistance)
    sv = reg.space_check(floor_dbu, False, kdb.Region.Euclidian, 80, None, None,
                         True, kdb.Region.NoOppositeFilter, kdb.Region.NoRectFilter,
                         False, kdb.Region.IgnoreProperties,
                         kdb.Region.NeverIncludeZeroDistance)
    return wv, sv


def clear_field(w: float, h: float, metal: list[np.ndarray], *,
                dbu_um: float = 0.001, art_box_um: float | None = None,
                finish_art_um: float = FINISH_ART_UM,
                finish_frame_um: float = FINISH_FRAME_UM,
                ) -> list[np.ndarray]:
    """``die box − metal`` as hole-free polygons, origin-centred.

    ``metal`` mixes ``(N,4)`` rect arrays and ``(K,2)`` vertex rings. The
    metal is merged and opened to the floor, one klayout Region boolean over
    the die takes the complement, the clear field is opened and checked to the
    floor (see ``FINISH_FRAME_UM``; ``art_box_um`` is the side of the centred
    centrepiece square that takes the finer radius), then a convex
    decomposition so the writer never sees a polygon with holes or a
    100k-vertex ring (pieces have at most six vertices). Zero-area pieces of
    the decomposition are dropped. ``finish_frame_um=0`` disables the finish.
    """
    from .patterns.effects.drc import _region_from_polys

    reg, kdb = _region_from_polys(metal, dbu_um)
    s = 1.0 / dbu_um
    reg.merge()
    floor_dbu = int(round(CLEAR_FLOOR_UM * s))
    box = kdb.Region(kdb.Box(int(round(-w / 2 * s)), int(round(-h / 2 * s)),
                             int(round(w / 2 * s)), int(round(h / 2 * s))))
    art = None
    if art_box_um and art_box_um > 0:
        a = int(round(art_box_um / 2 * s))
        art = kdb.Region(kdb.Box(-a, -a, a, a))

    def _open(r, radius_um):
        d = max(1, int(round(radius_um * s)))
        r = r.sized(-d).sized(d)
        r.merge()
        return r

    # The two zones overlap by a 5 µm strip: each half is opened on its own
    # and the union is taken, so no seam runs along the art-box edge (cutting
    # at one line and regrowing both halves to it left nanometre gaps along
    # it — 783 chrome-filament flags on the lid's back ply).
    strip = int(round(5.0 * s))
    art_in = art.sized(strip) if art is not None else None
    art_out = art.sized(-strip) if art is not None else None

    def _zone_open(r):
        if art is None:
            return _open(r, finish_frame_um)
        out = _open(r & art_in, finish_art_um) + _open(r - art_out, finish_frame_um)
        out.merge()
        return out

    def _settle(r):
        """Chrome over what the width check still flags — bounded and monotone.

        Only ever REMOVES clear (a sub-floor clear slit is chromed over, a thin
        chrome filament is thickened); never adds it: opening up a thin chrome
        filament puts new clear edges next to existing ones and the flags multiply (one photo die ran to 212,000 flags
        and 39 minutes that way). Two rounds at most, and a round that does not
        reduce the total count is undone. What remains is reported in the die's
        ``drc_written_*`` stats, not hidden.
        """
        r &= box
        r.merge()
        wv, sv = _drc_checks(r, kdb, floor_dbu)
        best, best_n = r, wv.count() + sv.count()
        for _ in range(2):
            if wv.is_empty() and sv.is_empty():
                break
            # a clear slit: chrome it over; a chrome filament: thicken it to
            # the floor by chroming a half-floor halo around it — both remove
            # clear only.
            patch = wv.polygons(0) + sv.polygons(0).sized(floor_dbu // 2)
            cand = best - patch
            cand.merge()
            wv2, sv2 = _drc_checks(cand, kdb, floor_dbu)
            n2 = wv2.count() + sv2.count()
            if n2 >= best_n:
                break
            best, best_n, wv, sv = cand, n2, wv2, sv2
        return best

    finishing = finish_frame_um > 0
    if finishing:
        reg = _zone_open(reg)
    clear = box - reg
    clear.merge()
    if finishing:
        clear = _settle(_zone_open(clear))
    if TRAPEZOID_DECOMP:
        # Convex pieces, horizontal-trapezoid preference. Every decomposition
        # of a polygon with slanted edges rounds its cut points to the DBU and
        # so is exact only to nanometre slivers along the cuts; measured on the
        # globe's back ply, the plain trapezoid decomposition left 29 sub-floor
        # clear notches at those cuts and this one none. ``written_clear_drc``
        # then measures the pieces merged — what the shop sees.
        mode = kdb.PreferredOrientation.PO_htrapezoids.to_i()
        dec = clear.decompose_convex_to_region(mode)
        if finishing:
            # The cut-point rounding of the decomposition is what a shop's check
            # sees; settle the merged pieces once more and keep the better of
            # the two (the settle is monotone and count-guarded, so this cannot
            # run away).
            wv, sv = _drc_checks(dec.merged(), kdb, floor_dbu)
            n1 = wv.count() + sv.count()
            if n1:
                dec2 = _settle(dec.merged()).decompose_convex_to_region(mode)
                wv2, sv2 = _drc_checks(dec2.merged(), kdb, floor_dbu)
                if wv2.count() + sv2.count() < n1:
                    dec = dec2
        clear = dec
    out: list[np.ndarray] = []
    for poly in clear.each():
        if poly.area() <= 0:
            continue
        pts = [(pt.x * dbu_um, pt.y * dbu_um) for pt in poly.each_point_hull()]
        if len(pts) >= 3:
            out.append(np.asarray(pts, dtype=np.float64))
    return out


def written_clear_drc(polys: list[np.ndarray], *, floor_um: float = CLEAR_FLOOR_UM,
                      dbu_um: float = 0.001) -> dict[str, Any]:
    """Width / space check of WRITTEN clear polygons (what the shop's incoming
    DRC sees), merged, Euclidian, zero-distance touches excluded. Counts only —
    a die is small enough that a whole-die check is seconds, not the
    superlinear pathology the fine export tiles around."""
    from .patterns.effects.drc import _region_from_polys

    reg, kdb = _region_from_polys(polys, dbu_um)
    if reg.is_empty():
        return {"n_polys": 0, "n_clear_width_viol": 0, "n_clear_space_viol": 0,
                "min_clear_width_um": float("inf"), "min_chrome_width_um": float("inf")}
    reg.merge()
    wv, sv = _drc_checks(reg, kdb, int(round(floor_um / dbu_um)))
    return {"n_polys": reg.count(), "n_clear_width_viol": wv.count(),
            "n_clear_space_viol": sv.count(),
            "min_clear_width_um": min((e.distance() * dbu_um for e in wv.each()), default=float("inf")),
            "min_chrome_width_um": min((e.distance() * dbu_um for e in sv.each()), default=float("inf"))}


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
             art_box_um: float | None = None,
             ) -> tuple[np.ndarray, list[np.ndarray]]:
    """One ply's written geometry, origin-centred and mirrored for the stack:
    ``(rects, polys)`` — the metal itself, or its clear-field complement."""
    if polarity == METAL:
        rects = [np.asarray(m) for m in metal if np.asarray(m).ndim == 2 and np.asarray(m).shape[1] == 4]
        polys = [np.asarray(m) for m in metal if np.asarray(m).ndim == 2 and np.asarray(m).shape[1] == 2]
        return eb.mirror_rects(_cat(*rects)) if rects else np.empty((0, 4)), _mirror_polys(polys)
    return np.empty((0, 4)), _mirror_polys(clear_field(w, h, metal, art_box_um=art_box_um))


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
    single = bool(getattr(pspec, "single_ply", False))
    from . import plates as P
    art_box = P.CENTERPIECE_FILL * P._aperture(pspec)   # the centred centrepiece square

    metal_f: list[np.ndarray] = list(fine.front_polys) + [bench_marks(face, "F", w, h)]
    fr, fp = _ply_art(w, h, metal_f, polarity, art_box)
    art = CellArt()
    art.front = _shift_rects(_cat(fr, dice_ticks(w, h)), cx, cy)
    art.polys = _shift_polys(fp, cx, cy)
    if not single:
        metal_b: list[np.ndarray] = list(fine.back_polys) + [bench_marks(face, "B", w, h)]
        br, bp = _ply_art(d["b_w"], d["b_h"], metal_b, polarity, art_box)
        art.back = _shift_rects(_cat(br, dice_ticks(d["b_w"], d["b_h"])), cx, cy)
        art.back_polys = _shift_polys(bp, cx, cy)
    drc = fine.stats.get("drc", {})
    art.stats = {
        "polarity": polarity, "face": face, "slug": pspec.pattern_slug,
        "ply_um": PLY_UM, "glass_n": GLASS_N, "mirrored": True,
        "f_um": [w, h], "b_um": [d["b_w"], d["b_h"]],
        "art_rim_um": float(pspec.weld_margin_um), "art_box_um": float(art_box),
        "band_um": float(P._band_um(pspec)),
        "periods": fine.stats.get("periods", {}),
        "metal_polys_front": len(fine.front_polys),
        "metal_polys_back": len(fine.back_polys),
        "written_polys_front": len(art.polys),
        "written_polys_back": len(art.back_polys),
        # metal-side heal result (export_fine), pre-inversion
        "drc_metal_front": drc.get("front_merged_after"),
        "drc_metal_back": drc.get("back_merged_after"),
        # the WRITTEN clear data, post-inversion
        "drc_written_front": written_clear_drc(fp) if polarity == CLEAR else None,
        "drc_written_back": (written_clear_drc(bp) if (polarity == CLEAR and not single) else None),
        "finish_um": [FINISH_ART_UM, FINISH_FRAME_UM],
        "single_layer": single,
        "pattern_params": dict(getattr(pspec, "pattern_params", {}) or {}),
    }
    return art


# --- the cells ------------------------------------------------------------------


def production_cells() -> list[Cell]:
    """The four written faces of the production box as plate cells, in layout
    order, each built from the SAME PlateSpec the box compositor and the GDS
    bake use (``blank_plan().faces``), so a die here is that face. Bonded faces
    are F + B pairs; single-ply faces (the photo sides) are one die whose inner
    ply is bare glass and takes no plate area."""
    MM = 1000.0
    _, spec = blank_plan()
    cells: list[Cell] = []
    for face, title in (("top", "lid"), ("front", "front"), ("left", "left"), ("right", "right")):
        ps = spec.faces[face]
        d = die_dims(face)
        slug = ps.pattern_slug
        params = dict(ps.pattern_params or {})
        single = bool(getattr(ps, "single_ply", False))
        what = slug + (f" {params.get('image')}" if params.get("image") else "")
        cells.append(Cell(
            cid=f"DIE-{face.upper()}", title=f"{title}: {what} + garland", group="X",
            w_um=d["f_w"], h_um=d["f_h"],
            back_w_um=None if single else d["b_w"], back_h_um=None if single else d["b_h"],
            build=(lambda face=face: (lambda cx, cy, w, h, polarity=METAL:
                   build_face_die(face, cx, cy, w, h, polarity)))(),
            label=f"{face.upper()} F {what}", block="production",
            two_layer=not single, takes_polarity=True,
            axis="production die", level=f"{face} " + ("F" if single else "F+B"),
            note=(("single ply: leaf gratings + carrier on the one ply, inner ply is bare glass; "
                   if single else
                   f"bonded pair, {d['f_w']/MM:.1f} mm outer / {d['b_w']/MM:.1f} mm inner ply; ")
                  + "mirrored for the chrome-down stack; 80/88 um verniers in the fold band")))
    return cells
