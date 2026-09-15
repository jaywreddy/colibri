"""Production dies on the witness plate: the box faces that come off as plies.

The 5″ plate is the stock the box is built from (``witness_geom.PLY_UM`` of
``GLASS_MATERIAL``), so a rectangle of it written with a face's fine geometry
IS that face's outer ply once it is diced. Every face is ONE written ply
(2026-09-15: the bonded moiré pairs of the first plate read badly on glass and
could not be cleaved; the inner plies are now bare quartz, cut from the second
blank, and carry no plate area). The dies, in dicing rows:

    32 mm row     DIE-TOP, DIE-TOP-S (spare)   monogram-jp as a single-layer
                  diffraction mapping (region_art) + garland;
                  DIE-BOTTOM   the solid gold base plate (no openings at all)
    30.5 mm rows  DIE-FRONT (globe-atlantic) and DIE-BACK (the Paris photograph
                  at the back's 32 x 30.5 mm), one per row, each followed by three candidate photographs
                  (``SIDE_PHOTOS``) as sides, one ply each, own garland seed;
                  ``SPARE_SIDES`` duplicates would fill any spot left over

Every die is the face's own ``PlateSpec`` from ``boxes.default_box_spec`` (via
``blank_plan``; the sides through ``side_photo_spec``) put through
``export_fine.build_plate_fine`` — the same geometry the box's GDS bake writes —
so nothing here is a second authoring path.

Polarity. The plate is a darkfield write with positive resist: the file holds
the openings, chrome stays wherever the file is empty. A face's fine geometry is
authored as METAL (gold where the rectangles are), so each die is inverted —
``die box − metal`` — and that inversion runs as a klayout Region boolean on the
one die, never a GEOS union. Before and after the inversion the geometry is
opened to the 2 µm floor (``clear_field``), decomposed to convex pieces so no
polygon carries a hole, and the written pieces are width/space checked tiled
(``written_clear_drc``).

Mirroring. A die is written MIRRORED (x → −x) because the box is assembled
chrome-down: the gold faces the inner ply and the picture is seen through its
own glass. A rotated die is mirrored first, then turned +90°.

Marks. Each die carries a tick-code ID (``export_blank.id_tick_rects``) inside
the interior foil-fold band, where the tape hides it, and L-shaped dicing ticks
just outside its corners in the street. The vernier combs of the bonded design
are gone with the pairs: a bare inner ply has nothing to read them against.

Everything is in micrometres, plate-centred, y up — the ``CellArt`` contract
of ``export_witness``.
"""
from __future__ import annotations

import math
import time
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
    """``(dims, BoxSpec)`` of the PRODUCTION box (``boxes.default_box_spec``):
    the dies on this plate are that box's plies, cut at its dimensions. ``dims``
    is the ``(width, depth, height)`` triple the pair rects are derived from.
    The plate no longer sizes the box (the old solve packed all twelve plies of
    the largest box onto one blank); the ring sizes the box, and the plate
    carries what fits."""
    from .boxes import default_box_spec

    spec = default_box_spec()
    if abs(spec.glass.thickness_um - PLY_UM) > 1e-6 or abs(spec.glass.n - GLASS_N) > 1e-9:
        raise RuntimeError("production box glass differs from witness_geom PLY_UM / GLASS_N")
    dims = (spec.width_um, spec.depth_um, spec.height_um)
    return dims, spec


def die_dims(face: str) -> dict[str, float]:
    """Cut dimensions (µm) of a face's outer (F) and inner (B) plies."""
    (w_um, d_um, h_um), _ = blank_plan()
    dims = {r.face: (r.width_um, r.height_um)
            for r in eb.pair_rects(w_um, d_um, h_um, PLY_UM)}
    fw, fh = dims[eb.subplate_id(face, "F")]
    bw, bh = dims[eb.subplate_id(face, "B")]
    return {"f_w": fw, "f_h": fh, "b_w": bw, "b_h": bh}


def _fold_um() -> float:
    from .assembly import bonded_overlap_um

    _, spec = blank_plan()
    return bonded_overlap_um(spec.foil, PLY_UM)


def bench_marks(face: str, ply: str, stack_w: float, stack_h: float) -> np.ndarray:
    """Tick-code ID for one ply (``face`` index + 1 bars, underlined on a B
    ply), in the interior foil-fold band — stack-centred, METAL sense. The
    vernier combs of the bonded design are not written: the plate carries
    single plies and a bare inner ply has nothing to beat against."""
    from .assembly import FACE_IDS

    fold = _fold_um()
    return eb.id_tick_rects(list(FACE_IDS).index(face), ply == "B",
                            stack_w, stack_h, PLY_UM, fold)


def dice_ticks(w: float, h: float) -> np.ndarray:
    """Corner L-ticks in the street around a ``w × h`` die centred at the
    origin. Emitted as DATA in either polarity: in the darkfield write the
    street is chrome and a clear L is what a scribe can see."""
    return eb.dice_tick_rects(eb.Placement("die", -w / 2.0, -h / 2.0, w, h, False))


# --- inversion ----------------------------------------------------------------


CLEAR_FLOOR_UM = 2.0
"""The shop's rule, both senses: no written CLEAR feature and no CHROME left
between clear features narrower than this."""

SETTLE_ROUNDS = 4
"""Patch rounds the written-data settle may take. Four, because that is the
budget the old two-settle structure had: it settled the merged clear (2 rounds),
decomposed, and settled the decomposed pieces again (2 more). The settle now
runs ONCE, after the decomposition — the tiled check needs hole-free pieces and
the pieces are what the file holds — so the rounds are spent in one place
instead of two. Measured on a synthetic 4 mm line-screen die (8010 metal
shapes), against the old structure's 428 flagged sites: 2 rounds left 622,
3 left 424, 4 leaves 235 — and does it while chroming over LESS clear than the
old structure did (+4847 µm² of clear kept, 0.04% of the field). This is a
ceiling, not a schedule: the settle is monotone and any round that does not
reduce the count is undone, so a clean die still costs one check."""

FINISH_ART_UM = CLEAR_FLOOR_UM / 2.0
FINISH_FRAME_UM = 1.0   # == FINISH_ART_UM: an open of radius r deletes every line under 2r, and the
# single-ply garland writes 4.15 um leaf gratings (2.075 um lines) in the frame band.
# 1.2 um (chosen when the frame held only 36 um leaf lines) erased the two finest
# families outright on the 07:37 plate; 1.0 leaves 75 nm of core on the finest line.
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
check flags it, in at most ``SETTLE_ROUNDS`` monotone rounds, and the remainder is counted in
the die's ``drc_written_*`` stats — a few sub-micron pockets that resist will
not resolve anyway, listed rather than hidden."""


def _patch_boxes(patch, dbu_um: float, halo_um: float) -> list[tuple[float, float, float, float]]:
    """Bounding boxes (µm, grown by ``halo_um``) of a settle patch — the only
    places the next round's check can differ. See ``_settle_pieces``."""
    out = []
    for poly in patch.each():
        bb = poly.bbox()
        out.append((bb.left * dbu_um - halo_um, bb.bottom * dbu_um - halo_um,
                    bb.right * dbu_um + halo_um, bb.top * dbu_um + halo_um))
    return out


def _tiled_checks(pieces, kdb, floor_dbu: int, dbu_um: float, only_near=None):
    """The die's width/space check, run TILED on hole-free convex pieces.

    Returns ``(n_width, min_w_um, n_space, min_s_um, w_markers, s_markers)`` —
    counts are merged violation SITES and the marker Regions are the settle's
    patch. The check options are the tiled checker's
    (``patterns.effects.drc._tiled_check_stats``): Euclidian, ignore_angle 80,
    NeverIncludeZeroDistance, and shielded=FALSE where the whole-region call
    below asks for shielded=True — a shielded violation always co-occurs with
    an unshielded one (drc.py's module note), shielding costs ~7x on dense
    geometry, and on every die-like case measured here the two agreed to the
    pair. ``include_touching`` keeps the ZERO-distance pairs — the acute corner
    where an angled line is clipped by the die edge, whose taper the settle has
    always chromed over — so the patch set stays what it was.

    Why tiled at all: klayout's width/space check is QUADRATIC in contiguous
    edge length, and the clear field of a garland-covered die is exactly the
    pathological shape — clear runs the length of the die between long angled
    lines. Measured on that geometry (angled 22 µm lines on a 44 µm pitch,
    clipped to a square die), one whole-region check pair:

        die   2 mm    4 mm    8 mm     16 mm     24 mm
        whole 0.05 s  0.40 s  3.51 s   34.05 s   121.50 s
        tiled 0.02 s  0.03 s  0.12 s    0.37 s     0.78 s

    A production die is 22-30 mm and the old code ran the whole-region check
    four to six times per die.
    """
    from .patterns.effects.drc import _tiled_check_stats

    return _tiled_check_stats(pieces, kdb, width_dbu=floor_dbu, gap_dbu=floor_dbu,
                              dbu_um=dbu_um, include_touching=True,
                              only_near=only_near)


def _drc_checks(reg, kdb, floor_dbu: int):
    """(width, space) EdgePairs at the floor: Euclidian, edges at 80° or more
    ignored, zero-distance touches excluded — the incoming check a mask shop
    runs.

    Only the ``TRAPEZOID_DECOMP = False`` fallback uses this now: on a merged
    region with holes there is nothing to tile (a hole-free piece list is what
    the tiled checker needs), and that path never ships. Everything that does
    ship goes through :func:`_tiled_checks`."""
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
                timing: dict[str, Any] | None = None,
                drc_out: dict[str, Any] | None = None,
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

    ``timing``, if given, is filled with the per-stage seconds this die spent
    (merge / opens / inversion / decomposition / settle / ring extraction) —
    the manifest carries it, so a slow die says WHERE it was slow.

    ``drc_out``, if given, is filled with the settle's OWN final measurement in
    :func:`written_clear_drc` form. The settle ends by checking the piece set it
    returns, with the identical options and floor that ``written_clear_drc``
    uses, so running that check again on the returned pieces measured the same
    geometry twice — 10 s of the 163 s a coloured photo die cost. It is left
    empty when there is no settle (``finish_frame_um=0``, or the
    ``TRAPEZOID_DECOMP=False`` fallback, whose check is the whole-region one),
    and the caller then measures for itself.

    Order of work. The decomposition runs BEFORE the settle, not after: the
    settle's width/space check is quadratic in contiguous edge length on the
    merged clear polygon (see :func:`_tiled_checks`), and hole-free convex
    pieces are exactly what the tiled checker needs. Same geometry either way —
    the pieces merge back to the region they came from — but the check that
    used to run four to six times per die at whole-die edge length now runs at
    most three times, tiled. The decomposition's own cut-point rounding is
    therefore inside the settle rather than after it, which is where a shop's
    check sees it anyway.
    """
    from .patterns.effects.drc import _region_from_polys

    _T = time.perf_counter
    tm: dict[str, float] = {}
    _t = _T()
    reg, kdb = _region_from_polys(metal, dbu_um)
    s = 1.0 / dbu_um
    reg.merge()
    tm["merge_metal_s"] = round(_T() - _t, 2)
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

    def _settle_pieces(dec, mode):
        """Chrome over what the check still flags — bounded and monotone.

        Only ever REMOVES clear (a sub-floor clear slit is chromed over, a thin
        chrome filament is thickened); never adds it: opening up a thin chrome
        filament puts new clear edges next to existing ones and the flags
        multiply (one photo die ran to 212,000 flags and 39 minutes that way).
        ``SETTLE_ROUNDS`` rounds at most, and a round that does not reduce the
        total count is undone. What remains is reported in the die's
        ``drc_written_*`` stats, not hidden.

        Runs on the DECOMPOSED pieces, so the check is tiled (see
        :func:`_tiled_checks`) and what it measures is what the file holds —
        the cut-point rounding of the decomposition included. Each round
        subtracts the patch, which re-fuses the pieces into a region with
        holes, so the round ends by decomposing again: ``best`` is always a
        piece set.
        """
        nw, mnw, ns, mns, wm, sm = _tiled_checks(dec, kdb, floor_dbu, dbu_um)
        best, best_n = dec, nw + ns
        best_stats = (nw, mnw, ns, mns)
        rounds = [[nw, ns]]
        for _ in range(SETTLE_ROUNDS):
            if wm.is_empty() and sm.is_empty():
                break
            # a clear slit: chrome it over; a chrome filament: thicken it to
            # the floor by chroming a half-floor halo around it — both remove
            # clear only.
            patch = wm + sm.sized(floor_dbu // 2)
            cand = best - patch
            cand.merge()
            cand = cand.decompose_convex_to_region(mode)
            # Re-check only the TILES the patch reaches. Away from the patch the
            # merged clear is bit-identical, so its decomposition is too (the
            # decomposition is per polygon), and the round that just ran found no
            # site there — every site it found is inside this patch. Restricting
            # by tile rather than by polygon is what keeps it exact: the pieces
            # ABUT, and a tile measures the shape they fuse into, so dropping a
            # neighbouring PIECE would report its neighbour's own width as a
            # violation (see drc.patched_neighbourhood). Tiles are independent,
            # so a kept tile answers exactly what it answers in a whole-die pass.
            near = _patch_boxes(patch, dbu_um, 2.0 * CLEAR_FLOOR_UM)
            nw2, mnw2, ns2, mns2, wm2, sm2 = _tiled_checks(
                cand, kdb, floor_dbu, dbu_um, only_near=near)
            n2 = nw2 + ns2
            rounds.append([nw2, ns2])
            if n2 >= best_n:
                break
            best, best_n, wm, sm = cand, n2, wm2, sm2
            best_stats = (nw2, mnw2, ns2, mns2)
        if drc_out is not None:
            # The check that produced ``best_stats`` ran on ``best`` — the piece
            # set this returns — with written_clear_drc's own options. Hand it
            # over rather than repeating it. ``n_polys`` is filled by the caller,
            # which is the one that knows how many pieces survived the zero-area
            # drop below.
            nwf, mnwf, nsf, mnsf = best_stats
            drc_out.update({
                "n_clear_width_viol": nwf, "n_clear_space_viol": nsf,
                "min_clear_width_um": mnwf, "min_chrome_width_um": mnsf,
                "settle_rounds": rounds,
            })
        return best

    def _settle_region(r):
        """The ``TRAPEZOID_DECOMP = False`` fallback: same settle on a merged
        region with holes, which only the whole-region check can measure."""
        r &= box
        r.merge()
        wv, sv = _drc_checks(r, kdb, floor_dbu)
        best, best_n = r, wv.count() + sv.count()
        for _ in range(2):
            if wv.is_empty() and sv.is_empty():
                break
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
    _t = _T()
    if finishing:
        reg = _zone_open(reg)
    tm["open_metal_s"] = round(_T() - _t, 2)
    _t = _T()
    clear = box - reg
    clear.merge()
    tm["invert_s"] = round(_T() - _t, 2)
    _t = _T()
    if finishing:
        clear = _zone_open(clear)
        clear &= box
        clear.merge()
    tm["open_clear_s"] = round(_T() - _t, 2)
    if TRAPEZOID_DECOMP:
        # Convex pieces, horizontal-trapezoid preference. Every decomposition
        # of a polygon with slanted edges rounds its cut points to the DBU and
        # so is exact only to nanometre slivers along the cuts; measured on the
        # globe's back ply, the plain trapezoid decomposition left 29 sub-floor
        # clear notches at those cuts and this one none. The settle below and
        # ``written_clear_drc`` then measure the pieces locally merged, tile by
        # tile — what the shop sees, at a cost that does not explode with the
        # die's size.
        mode = kdb.PreferredOrientation.PO_htrapezoids.to_i()
        _t = _T()
        clear = clear.decompose_convex_to_region(mode)
        tm["decompose_s"] = round(_T() - _t, 2)
        _t = _T()
        if finishing:
            clear = _settle_pieces(clear, mode)
        tm["settle_s"] = round(_T() - _t, 2)
    elif finishing:
        _t = _T()
        clear = _settle_region(clear)
        tm["settle_s"] = round(_T() - _t, 2)
    _t = _T()
    out: list[np.ndarray] = []
    for poly in clear.each():
        if poly.area() <= 0:
            continue
        pts = [(pt.x * dbu_um, pt.y * dbu_um) for pt in poly.each_point_hull()]
        if len(pts) >= 3:
            out.append(np.asarray(pts, dtype=np.float64))
    tm["rings_s"] = round(_T() - _t, 2)
    tm["n_pieces"] = len(out)
    if timing is not None:
        timing.update(tm)
    if drc_out:
        drc_out["n_polys"] = len(out)
    return out


def written_clear_drc(polys: list[np.ndarray], *, floor_um: float = CLEAR_FLOOR_UM,
                      dbu_um: float = 0.001) -> dict[str, Any]:
    """Width / space check of WRITTEN clear polygons (what the shop's incoming
    DRC sees), Euclidian, locally merged, run TILED.

    A die is NOT small enough for a whole-die check: it is exactly the shape
    that check is quadratic on (see :func:`_tiled_checks` for the measured
    ladder — 121 s for one check pair on a 24 mm die of angled lines, 0.78 s
    tiled). The written pieces are hole-free convex polygons, which is what the
    tiled checker wants, so this is the cheap direction AND the honest one.

    Two things about the numbers changed with the tiled checker, and neither is
    a change of what is being measured:

    * ``n_clear_width_viol`` / ``n_clear_space_viol`` count merged violation
      SITES — connected sub-floor regions — not raw klayout edge PAIRS. One
      physical notch used to be reported as however many pairs klayout chose to
      split its edges into; sites are stable under tiling and under geometry
      representation.
    * ``min_clear_width_um`` is the narrowest genuine (non-zero) distance. The
      old whole-region tally included the zero-distance pairs at the acute
      corner where an angled line meets the die edge, so it reported a
      meaningless ``0.0`` on almost every die. Those corners are still COUNTED
      (``include_touching``, same as the settle patches them); they just no
      longer set the headline minimum.

    ``n_polys`` is the number of written pieces handed in, not the merged
    island count: merging the whole die to count islands is the one expensive
    thing this function used to do for a number nobody reads.
    """
    import klayout.db as kdb

    n = sum(1 for p in polys if np.asarray(p).size)
    if not n:
        return {"n_polys": 0, "n_clear_width_viol": 0, "n_clear_space_viol": 0,
                "min_clear_width_um": float("inf"), "min_chrome_width_um": float("inf")}
    nw, mnw, ns, mns, _, _ = _tiled_checks(
        list(polys), kdb, int(round(floor_um / dbu_um)), dbu_um)
    return {"n_polys": n, "n_clear_width_viol": nw, "n_clear_space_viol": ns,
            "min_clear_width_um": mnw, "min_chrome_width_um": mns}


def _written_drc(polys: list[np.ndarray], measured: dict[str, Any]) -> dict[str, Any]:
    """The written-clear DRC block: the settle's own final measurement when it
    made one, otherwise measured here. Same numbers either way — see
    :func:`clear_field`'s ``drc_out`` and :func:`written_clear_drc`."""
    if not measured or "n_polys" not in measured:
        return written_clear_drc(polys)
    out = {"n_polys": int(measured["n_polys"]),
           "n_clear_width_viol": int(measured["n_clear_width_viol"]),
           "n_clear_space_viol": int(measured["n_clear_space_viol"]),
           "min_clear_width_um": measured["min_clear_width_um"],
           "min_chrome_width_um": measured["min_chrome_width_um"]}
    if measured.get("settle_rounds"):
        out["settle_rounds"] = measured["settle_rounds"]
    return out


_MIRROR = np.array([-1.0, 1.0])


def _mirror_polys(polys: list[np.ndarray]) -> list[np.ndarray]:
    # ``eb._transform_verts(..., mirror=True, rotated=False)`` is x -> -x; one
    # multiply per polygon beats its stack() of two column slices, and a die
    # hands this 200k polygons.
    return [np.asarray(p, dtype=np.float64) * _MIRROR for p in polys]


def _shift_polys(polys: list[np.ndarray], dx: float, dy: float) -> list[np.ndarray]:
    d = np.array([dx, dy])          # hoisted: it was rebuilt per polygon
    return [np.asarray(p, dtype=np.float64) + d for p in polys]


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
             timing: dict[str, Any] | None = None,
             drc_out: dict[str, Any] | None = None,
             ) -> tuple[np.ndarray, list[np.ndarray]]:
    """One ply's written geometry, origin-centred and mirrored for the stack:
    ``(rects, polys)`` — the metal itself, or its clear-field complement."""
    if polarity == METAL:
        rects, polys = [], []
        for m in metal:                       # one asarray per item, not three
            a = np.asarray(m)
            if a.ndim == 2 and a.shape[1] == 4:
                rects.append(a)
            elif a.ndim == 2 and a.shape[1] == 2:
                polys.append(a)
        return eb.mirror_rects(_cat(*rects)) if rects else np.empty((0, 4)), _mirror_polys(polys)
    return np.empty((0, 4)), _mirror_polys(
        clear_field(w, h, metal, art_box_um=art_box_um, timing=timing,
                    drc_out=drc_out))


# --- the faces ------------------------------------------------------------------


def _rotate_rects(r: np.ndarray) -> np.ndarray:
    return eb._transform_rects(np.asarray(r, dtype=np.float64), mirror=False, rotated=True)


def _rotate_polys(polys: list[np.ndarray]) -> list[np.ndarray]:
    return [eb._transform_verts(np.asarray(p, dtype=np.float64), mirror=False, rotated=True)
            for p in polys]


def build_face_die(face: str, cx: float, cy: float, w: float, h: float,
                   polarity: str = METAL, pspec: Any | None = None,
                   rotated: bool = False) -> CellArt:
    """One face as ONE die: the outer ply, ``w × h`` at ``(cx, cy)``. Fine
    geometry from ``export_fine.build_plate_fine`` (single-ply branch: the
    centrepiece as region gratings or a line screen, the garland as per-family
    leaf gratings), merged DRC heal included, then inverted to the clear field
    and mirrored for the chrome-down stack.

    ``rotated``: the die is turned +90° on the plate (the cell is then
    ``h × w``), so a 32 × 30.5 mm front can ride a 32 mm dicing row on its side.
    Mirror first, then rotate — the ID ticks turn with the picture and say so.
    """
    from .export_fine import build_plate_fine

    _, spec = blank_plan()
    if pspec is None:
        pspec = spec.faces[face]
    if not bool(getattr(pspec, "single_ply", False)):
        raise ValueError(f"{face}: the plate writes single plies only; {pspec.pattern_slug} "
                         "is not single_ply (every face of the production box is)")
    d = die_dims(face)
    fw, fh = d["f_w"], d["f_h"]
    cw, ch = (fh, fw) if rotated else (fw, fh)
    if abs(cw - w) > 1.0 or abs(ch - h) > 1.0:
        raise ValueError(f"{face}: cell is {w:.0f}x{h:.0f} but the F ply cuts "
                         f"{fw:.0f}x{fh:.0f}{' rotated' if rotated else ''}")
    # Stage timings: a die that costs minutes should say which stage spent
    # them. ``fine`` is the pattern generation + the metal-side merged DRC heal
    # (export_fine), the clear_field entries are the inversion and its
    # written-data finish, ``drc_written`` the report on what is in the file.
    T = time.perf_counter
    timing: dict[str, Any] = {}
    _t = T()
    fine = build_plate_fine(pspec, face)
    timing["fine_s"] = round(T() - _t, 2)
    if len(fine.back_polys):
        raise RuntimeError(f"{face}: a single-ply face emitted {len(fine.back_polys)} back polygons")
    from . import plates as P
    art_box = P.CENTERPIECE_FILL * P._aperture(pspec)   # the centred centrepiece square

    metal_f: list[np.ndarray] = list(fine.front_polys) + [bench_marks(face, "F", fw, fh)]
    cf_f: dict[str, Any] = {}
    drc_f: dict[str, Any] = {}
    _t = T()
    fr, fp = _ply_art(fw, fh, metal_f, polarity, art_box, timing=cf_f, drc_out=drc_f)
    timing["clear_field_front_s"] = round(T() - _t, 2)
    timing["clear_field_front"] = cf_f
    if rotated:
        fr, fp = _rotate_rects(fr), _rotate_polys(fp)
    art = CellArt()
    art.front = _shift_rects(_cat(fr, dice_ticks(cw, ch)), cx, cy)
    art.polys = _shift_polys(fp, cx, cy)
    _t = T()
    # The settle already measured exactly this, on exactly these pieces (see
    # ``clear_field``'s ``drc_out``); ``written_clear_drc`` is the fallback for
    # the no-settle paths. Mirroring and rotation do not enter it: both are
    # isometries, so every width/space distance is the one the settle counted.
    drc_written_front = _written_drc(fp, drc_f) if polarity == CLEAR else None
    timing["drc_written_s"] = round(T() - _t, 2)
    drc = fine.stats.get("drc", {})
    art.stats = {
        "polarity": polarity, "face": face, "slug": pspec.pattern_slug,
        "ply_um": PLY_UM, "glass_n": GLASS_N, "mirrored": True, "rotated": bool(rotated),
        "f_um": [fw, fh], "cell_um": [cw, ch],
        "art_rim_um": float(pspec.weld_margin_um), "art_box_um": float(art_box),
        "band_um": float(P._band_um(pspec)),
        "periods": fine.stats.get("periods", {}),
        "single_layer_regions": fine.stats.get("single_layer_regions"),
        "single_ply_leaves": fine.stats.get("single_ply_leaves"),
        "metal_polys_front": len(fine.front_polys),
        "written_polys_front": len(art.polys),
        # metal-side heal result (export_fine), pre-inversion
        "drc_metal_front": drc.get("front_merged_after"),
        # the WRITTEN clear data, post-inversion
        "drc_written_front": drc_written_front,
        "timing_s": timing,
        "finish_um": [FINISH_ART_UM, FINISH_FRAME_UM],
        "single_layer": True,
        "pattern_params": dict(getattr(pspec, "pattern_params", {}) or {}),
    }
    return art


# --- the cells ------------------------------------------------------------------


# Every prepared photograph, as a side ply. The box has two photo walls; the
# plate carries every candidate so the choice is made on glass, not on a
# screen. (image, colour_mode, cell id, garland seed): the colour mode is
# "authored" for every picture — each carries its own colour plan
# (``assets/photos/<image>.colour.json``, see photo.colour_plan). Every ply
# grows its OWN garland: beach and sunset carry the box's left/right seeds
# (104 / 105, so the die IS that face), the rest continue the sequence.
SIDE_PHOTOS: tuple[tuple[str, str, str, int], ...] = (
    ("beach", "authored", "DIE-LEFT", 104),
    ("sunset", "authored", "DIE-RIGHT", 105),
    ("garden", "authored", "DIE-GARDEN", 106),
    ("paris", "authored", "DIE-PARIS", 107),
    ("night-group", "authored", "DIE-NIGHT", 108),
    ("porch-group", "authored", "DIE-PORCH", 109),
)

# Duplicates that fill any side spot the dicing rows leave after SIDE_PHOTOS:
# (image, cell id, garland seed). With the solid base plate in the 32 mm row
# (2026-09-15) the front's spare moved down beside the sides, and the two
# 30.5 mm rows hold exactly the six photographs — no spot is free. To carry a
# spare photograph, drop a spare face (or a photograph) and list it here.
SPARE_SIDES: tuple[tuple[str, str, int], ...] = ()

# Spare copies of the lid and the front: a diced ply that chips is replaced
# from the same plate. (face, cell id, rotated 90 deg on the plate)
SPARE_FACES: tuple[tuple[str, str, bool], ...] = (
    ("top", "DIE-TOP-S", False),
)
# (the front's spare, DIE-FRONT-S, gave its 32 x 30.5 slot to the BACK
# photograph on 2026-09-15; the two 30.5 mm rows hold the front, the back and
# six sides exactly)


def side_photo_spec(image: str, colour_mode: str, seed: int | None = None) -> Any:
    """The LEFT wall's PlateSpec with another photograph in it — same glass,
    frame dials, single ply and cut dims, so every photo die is a drop-in side —
    and its own garland seed."""
    import dataclasses

    _, spec = blank_plan()
    base = spec.faces["left"]
    params = dict(base.pattern_params or {})
    params.update({"image": image, "colour_mode": colour_mode})
    if seed is None:
        seed = next(sd for im, _, _, sd in SIDE_PHOTOS if im == image)
    frame = dataclasses.replace(base.frame, seed=int(seed))
    return dataclasses.replace(base, pattern_params=params, frame=frame)


def _face_cell(face: str, cid: str, title: str, *, rotated: bool = False,
               pspec: Any | None = None, label: str, level: str, note: str) -> Cell:
    d = die_dims(face)
    w, h = (d["f_h"], d["f_w"]) if rotated else (d["f_w"], d["f_h"])
    return Cell(
        cid=cid, title=title, group="X", w_um=w, h_um=h,
        build=(lambda face=face, pspec=pspec, rotated=rotated:
               (lambda cx, cy, w, h, polarity=METAL:
                build_face_die(face, cx, cy, w, h, polarity, pspec=pspec, rotated=rotated)))(),
        label=label, block="production", two_layer=False, takes_polarity=True,
        axis="production die", level=level, note=note)


def production_cells() -> list[Cell]:
    """The written faces of the production box as plate cells, in LAYOUT
    order, each built from the SAME PlateSpec the box compositor and the GDS
    bake use (``blank_plan().faces``), so a die here is that face — every one
    a single ply whose inner ply is bare glass and takes no plate area.

    The order is the dicing plan (``export_witness.layout`` keeps a height
    class in the order given): the 32 mm class — lid, base plate, lid spare —
    then the 30.5 mm class with a FRONT (the die, then its spare) leading each
    row of three photographs, so two 32 mm fronts and six 27.5 mm sides fill
    two 117.5 mm rows exactly. ``SPARE_SIDES`` duplicates trail the list.
    """
    MM = 1000.0
    _, spec = blank_plan()

    def _face(face: str, cid: str, title: str, label: str, level: str, note: str) -> Cell:
        return _face_cell(face, cid, title, label=label, level=level, note=note)

    tall: list[Cell] = []          # the 32 mm class
    fronts: list[Cell] = []        # the 30.5 mm class, 32 mm wide
    sides: list[Cell] = []         # the 30.5 mm class, 27.5 mm wide
    for face, title in (("top", "lid"), ("front", "front"), ("back", "back")):
        ps = spec.faces[face]
        d = die_dims(face)
        what = (f"{ps.pattern_slug} {ps.pattern_params.get('image', '')} + garland"
                if ps.pattern_slug == "photo-halftone" else f"{ps.pattern_slug} + garland")
        c = _face(face, f"DIE-{face.upper()}", f"{title}: {what}",
                  f"{face.upper()} {ps.pattern_params.get('image', ps.pattern_slug)}", f"{face}",
                  f"one ply, {d['f_w']/MM:.1f} x {d['f_h']/MM:.1f} mm; "
                  + ("line screen with authored colour zones" if ps.pattern_slug == "photo-halftone"
                     else "single-layer diffraction mapping (colour by region)")
                  + "; mirrored for the chrome-down stack")
        (tall if face == "top" else fronts).append(c)
    # The solid gold base plate: a die with no openings, so it costs the mask
    # nothing but its ticks — and the box a gold floor under the ring.
    psb = spec.faces["bottom"]
    d = die_dims("bottom")
    tall.append(_face("bottom", "DIE-BOTTOM", f"base: {psb.pattern_slug}",
                      f"BOTTOM {psb.pattern_slug}", "bottom",
                      f"one ply, {d['f_w']/MM:.1f} x {d['f_h']/MM:.1f} mm; solid gold, no openings"))
    for face, cid, rot in SPARE_FACES:
        ps = spec.faces[face]
        c = _face_cell(face, cid, f"spare {face}: {ps.pattern_slug} + garland", rotated=rot,
                       label=f"{face.upper()} SPARE{' ROT' if rot else ''}", level=f"{face} spare",
                       note=(f"spare copy of DIE-{face.upper()}" + (", rotated +90 deg on the plate" if rot else "")))
        (tall if face == "top" else fronts).append(c)
    for image, mode, cid, seed in SIDE_PHOTOS:
        ps = side_photo_spec(image, mode, seed)
        sides.append(_face_cell(
            "left", cid, f"side: {ps.pattern_slug} {image} ({mode}) + garland", pspec=ps,
            label=f"SIDE photo {image}", level=f"side {image}",
            note="one ply: line screen with authored colour zones, leaf gratings on bare glass; "
                 "mirrored for the chrome-down stack"))
    for image, cid, seed in SPARE_SIDES:
        ps = side_photo_spec(image, "authored", seed)
        sides.append(_face_cell(
            "left", cid, f"spare side: {ps.pattern_slug} {image} + garland", pspec=ps,
            label=f"SIDE photo {image} SPARE", level=f"side spare {image}",
            note="duplicate side ply (own garland seed)"))
    cells: list[Cell] = list(tall)
    per_row = 3                     # 32 + 3 x 27.5 + 3 streets = 117.5 mm
    for i, fr in enumerate(fronts):
        cells.append(fr)
        cells.extend(sides[i * per_row:(i + 1) * per_row])
    cells.extend(sides[len(fronts) * per_row:])
    return cells
