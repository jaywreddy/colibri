"""The 5-inch witness plate: does the DoE fit, and is each cell what it claims?

Built from SMALL cells wherever a shipping-size one would be expensive. The real
plate is half a million rectangles, a 10 s build; what is worth pinning here is
the plate CONTRACTS — the packer fits, the layout is a dicing grid, the dies are
the box's own plies, the litho floor is reported honestly, and **metal and clear
are exact complements**, which on a darkfield write is the difference between
the part and its negative.

The optical IDENTITIES the plate's cells used to carry (beat, rotation,
harmonic, near field, swap angle) live in ``tests/test_optics_math.py`` since
2026-09-16, pinned against the formulas rather than against a cell's stats dict.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pytest

from app import witness_cells as wc
from app import witness_moire as wm
from app.export_witness import (
    C3_DUTY_LADDER,
    MM,
    PERIOD_LADDER_UM,
    RESOLUTION_LADDER_UM,
    _label_h,
    _plan,
    build_plate,
    clear_by_boolean,
    doe_cells,
    flat_rect_count,
    layout,
)
from app.patterns.bitmap import screenrects as sr
from app.patterns.bitmap.colourzone import MIN_FEATURE_UM
from app.witness_geom import (
    CLEAR,
    STREET_UM,
    METAL,
    BLANK_SIDE_UM,
    PLY_UM,
    USABLE_UM,
    Cell,
    _grating_rects,
    _text_rects,
    invert_grating,
)


def _area(r) -> float:
    r = np.asarray(r, dtype=np.float64)
    return float(((r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2])).sum()) if len(r) else 0.0


def _poly_area(pv) -> float:
    pv = np.asarray(pv, dtype=np.float64)
    x, y = pv[:, 0], pv[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _array_area(e) -> float:
    plan = sr.stripe_plan(
        e["rects"], e["period_um"],
        np.asarray(e["line_um"]) / np.asarray(e["period_um"]),
        phase_um=e.get("phase_um", 0.0))
    r = np.asarray(e["rects"], dtype=np.float64)
    return float((plan["n"] * plan["line_um"] * (r[:, 3] - r[:, 2])).sum())


def _art_area(art) -> float:
    return (_area(art.front)
            + sum(_poly_area(p) for p in art.free_polys)
            + sum(_poly_area(p) for p in art.polys)
            + sum(_array_area(e) for e in art.arrays))


# --- layout -----------------------------------------------------------------


def test_the_whole_design_of_experiments_fits_the_plate():
    _, lay = layout(doe_cells())
    assert lay["fits"], f"overflow: {lay['overflow']}"
    assert lay["height_used_mm"] <= lay["height_available_mm"]


def test_no_cell_escapes_the_usable_area_and_none_overlap():
    placed, _ = layout(doe_cells())
    half = USABLE_UM / 2.0
    boxes = []
    for p in placed:
        for cx, w, h in [(p.cx, p.cell.w_um, p.cell.h_um)]:
            assert cx - w / 2 >= -half - 1e-6, p.cell.cid
            assert cx + w / 2 <= half + 1e-6, p.cell.cid
            assert abs(p.cy) + h / 2 <= half + 1e-6, p.cell.cid
            boxes.append((p.cell.cid, cx - w / 2, cx + w / 2, p.cy - h / 2, p.cy + h / 2))
    for i in range(len(boxes)):
        ai, ax0, ax1, ay0, ay1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            bi, bx0, bx1, by0, by1 = boxes[j]
            assert not (ax0 < bx1 - 1e-6 and bx0 < ax1 - 1e-6
                        and ay0 < by1 - 1e-6 and by0 < ay1 - 1e-6), f"{ai} / {bi}"


def _written_mm2(c) -> float:
    return c.w_um * c.h_um / 1e6


def test_the_area_budget_matches_the_plan():
    """docs/production-plate-plan.md decision 10: the plate is the box's
    plies plus spares — the production dies are most of the field — and what
    is left carries the single-layer bench cells for THIS process. No cell on
    the plate is two-layer any more."""
    placed, _ = layout(doe_cells())
    area: dict[str, float] = {}
    for p in placed:
        area[p.cell.block] = area.get(p.cell.block, 0.0) + _written_mm2(p.cell)
    usable = (USABLE_UM / MM) ** 2
    assert 0.65 < area["production"] / usable < 0.80, area
    exp = {k: v for k, v in area.items() if k != "production"}
    assert sum(exp.values()) / usable < 0.12, exp
    cids = {p.cell.cid for p in placed}
    from app.witness_dies import SIDE_PHOTOS, SPARE_FACES, SPARE_SIDES
    assert {cid for _, _, cid, _ in SIDE_PHOTOS} <= cids
    assert {cid for _, cid, _ in SPARE_FACES} <= cids
    assert {cid for _, cid, _ in SPARE_SIDES} <= cids
    bench = {"M-POL", "M-CD-dense", "M-CD-iso", "M-DUTY10", "M-DUTY5", "D-PER", "WEDGE44", "H-ACU"}
    assert bench <= cids, bench - cids
    assert len(placed) <= 24, "the experiment set stays cut down"


def test_the_production_dies_are_the_panelized_box_plies():
    """A die on this plate must be cut to the box's own ply dimensions
    (``ply_cuts.pair_rects``), ONE ply per face (the inner plies are bare
    glass). Spares are the same die again; the rotated front spare is the
    front's dims swapped."""
    from app import ply_cuts as pc
    from app.witness_dies import (SIDE_PHOTOS, SPARE_FACES, SPARE_SIDES, blank_plan, die_dims,
                                  production_cells)

    (w_um, d_um, h_um), spec = blank_plan()
    assert spec.glass.thickness_um == PLY_UM
    assert (w_um, d_um, h_um) == (spec.width_um, spec.depth_um, spec.height_um)
    assert all(spec.faces[f].single_ply for f in spec.faces), "every face is one written ply"
    cells = {c.cid: c for c in production_cells()}
    want = ({"DIE-TOP", "DIE-FRONT", "DIE-BACK", "DIE-BOTTOM"} | {cid for _, cid, _ in SPARE_FACES}
            | {cid for _, _, cid, _ in SIDE_PHOTOS} | {cid for _, cid, _ in SPARE_SIDES})
    assert set(cells) == want
    assert len(SIDE_PHOTOS) >= 6, "every prepared photograph rides the plate"
    for c in cells.values():
        assert c.takes_polarity and c.block == "production"
    for face in ("top", "front", "back", "bottom"):
        c, d = cells[f"DIE-{face.upper()}"], die_dims(face)
        assert (c.w_um, c.h_um) == (d["f_w"], d["f_h"])
    for face, cid, rot in SPARE_FACES:
        c, d = cells[cid], die_dims(face)
        assert (c.w_um, c.h_um) == ((d["f_h"], d["f_w"]) if rot else (d["f_w"], d["f_h"]))
    d = die_dims("left")
    for _, _, cid, _ in SIDE_PHOTOS:
        assert (cells[cid].w_um, cells[cid].h_um) == (d["f_w"], d["f_h"])
    panel = {r.face: (r.width_um, r.height_um)
             for r in pc.pair_rects(w_um, d_um, h_um, PLY_UM)}
    assert panel["top:F"] == (cells["DIE-TOP"].w_um, cells["DIE-TOP"].h_um)
    assert panel["left:F"] == (cells["DIE-LEFT"].w_um, cells["DIE-LEFT"].h_um)


def _cut_crosses(x0, x1, y0, y1, cut_axis, cut_v, lo, hi) -> bool:
    """Does a cut line at ``cut_v`` along ``cut_axis`` between ``lo..hi`` pass
    through the box?"""
    if cut_axis == "x":
        return x0 < cut_v < x1 and not (hi <= y0 or lo >= y1)
    return y0 < cut_v < y1 and not (hi <= x0 or lo >= x1)


def test_the_layout_is_a_dicing_grid():
    """Straight streets: every horizontal cut spans the plate between two
    rows and crosses no die; every vertical cut spans its strip and crosses
    no die; the cuts are exactly the streets between neighbours, so every die
    is freed by the protocol (rows, then strips, then column pieces)."""
    placed, lay = layout(doe_cells())
    dc = lay["dicing"]
    boxes = {}
    for p in placed:
        c = p.cell
        boxes[c.cid] = (p.cx - c.w_um / 2, p.cx + c.w_um / 2, p.cy - c.h_um / 2, p.cy + c.h_um / 2)
    half = USABLE_UM / 2
    for y in dc["y_cuts_mm"]:
        for cid, (x0, x1, y0, y1) in boxes.items():
            assert not _cut_crosses(x0, x1, y0, y1, "y", y * MM, -half, half), (cid, y)
    strips = dc["strips"]
    assert [s["row"] for s in strips] == list(range(1, len(strips) + 1))
    for s in strips:
        yt, yb = s["y_top_mm"] * MM, s["y_bot_mm"] * MM
        for cid in s["cells"]:
            x0, x1, y0, y1 = boxes[cid]
            assert y1 <= yt + 1e-6 and y0 >= yb - 1e-6, (cid, "outside its strip")
        for x in s["x_cuts_mm"]:
            for cid, (x0, x1, y0, y1) in boxes.items():
                assert not _cut_crosses(x0, x1, y0, y1, "x", x * MM, yb, yt), (cid, x)
        # neighbours in a strip are separated by exactly one street
        xs = sorted((boxes[c][0], boxes[c][1]) for c in s["cells"])
        gaps = [b[0] - a[1] for a, b in zip(xs, xs[1:])]
        for g in gaps:
            assert g >= STREET_UM - 1e-6, gaps
        # every production die spans the full strip height (exact cut dims)
        for cid in s["cells"]:
            p = next(q for q in placed if q.cell.cid == cid)
            if p.cell.block == "production":
                assert p.cy + p.cell.h_um / 2 == pytest.approx(yt)
                assert s["height_mm"] * MM == pytest.approx(p.cell.h_um + _label_h(p.cell.h_um))
    # rows are stacked top-down with one street between them
    for a, b in zip(strips, strips[1:]):
        assert a["y_bot_mm"] - b["y_top_mm"] == pytest.approx(STREET_UM / MM)
    # EDGE cuts: the top and the bottom edge of every strip's dies
    assert len(dc["y_cuts_mm"]) == 2 * len(strips)
    for s in strips:
        assert s["die_top_mm"] in dc["y_cuts_mm"] and s["die_bot_mm"] in dc["y_cuts_mm"]
        # and every die edge in the strip is a vertical cut
        for cid in s["cells"]:
            x0, x1, _, _ = boxes[cid]
            assert round(x0 / MM, 3) in s["x_cuts_mm"] and round(x1 / MM, 3) in s["x_cuts_mm"], cid
    # the saw-lane marks sit on the cut lines, outside every die
    from app.export_witness import _dice_edge_marks
    marks = _dice_edge_marks(lay)
    for x0, x1, y0, y1 in marks:
        for cid, (bx0, bx1, by0, by1) in boxes.items():
            assert not (x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1), (cid, "mark inside a die")


def test_a_cell_with_no_row_budget_left_stacks_into_a_column():
    """Columns are the fallback, not the plan: cells open their own rows while
    height remains, and only stack beside a tall cell once it does not."""
    def g(cx, cy, w, h):
        return wc.build_grating_patch(cx, cy, w, h, period_um=10.0)
    tall = Cell("TALL", "t", "X", 20000.0, 113000.0, g)
    shorts = [Cell(f"S{i}", "t", "X", 30000.0, 4000.0, g) for i in range(3)]
    placed, lay = layout([[tall], shorts])
    assert lay["fits"], lay
    assert len(lay["rows"]) == 1
    col = lay["dicing"]["strips"][0]["columns"]
    assert len(col) == 1 and col[0]["cells"] == ["S0", "S1", "S2"]
    assert len(col[0]["y_cuts_mm"]) == 6, "top and bottom edge of each of the three cells"
    ys = sorted({round(p.cy) for p in placed if p.cell.cid.startswith("S")})
    assert len(ys) == 3


def test_the_die_inversion_is_the_exact_complement_of_its_metal():
    """``die box - metal`` through the klayout Region, decomposed to hole-free
    convex pieces: the pieces tile the box with the metal (area, to the few
    square micrometres the 2 µm finish takes off the triangle's acute tip and
    the nanometre slivers a decomposition leaves along its cuts), and no piece
    carries a hole or leaves the box."""
    from app.witness_dies import clear_field

    W, H = 3000.0, 2000.0
    tri = np.array([[-1400.0, -900.0], [-1000.0, -900.0], [-1200.0, -500.0]])
    metal = [_grating_rects(0.0, 0.0, 2000.0, 1200.0, 40.0, 0.5), tri]
    clear = clear_field(W, H, metal)
    a_metal = _area(metal[0]) + _poly_area(tri)
    a_clear = sum(_poly_area(p) for p in clear)
    assert a_metal + a_clear == pytest.approx(W * H, rel=1e-5)
    for p in clear:
        assert len(p) <= 6, "convex pieces only"
        assert p[:, 0].min() >= -W / 2 - 1e-6 and p[:, 0].max() <= W / 2 + 1e-6
        assert p[:, 1].min() >= -H / 2 - 1e-6 and p[:, 1].max() <= H / 2 + 1e-6
def test_short_cells_open_their_own_row_not_a_pocket():
    """A dicing grid: a 4 mm cell after a 28 mm one opens a 4 mm ROW under it
    (one more full-width cut) instead of nesting in the tall row's leftover,
    so every cell is freed by straight cuts."""
    def g(cx, cy, w, h):
        return wc.build_grating_patch(cx, cy, w, h, period_um=10.0)
    tall = Cell("TALL", "t", "X", 20000.0, 28000.0, g)
    shorts = [Cell(f"S{i}", "t", "X", 30000.0, 4000.0, g) for i in range(3)]
    placed, lay = layout([[tall], shorts])
    assert len(lay["rows"]) == 2, lay["rows"]
    assert lay["rows"][1]["cells"] == ["S0", "S1", "S2"]
    ys = {round(p.cy) for p in placed if p.cell.cid.startswith("S")}
    assert len(ys) == 1, "one row, one y"


def test_every_ladder_is_one_cell_not_one_cell_per_rung():
    """The shape correction that bought back a third of the plate: a 4 mm rung
    drawn as its own cell pays 2.4 mm of row overhead — 60%."""
    ids = {p.cell.cid for p in layout(doe_cells())[0]}
    assert {"M-CD-dense", "M-CD-iso", "D-PER", "H-ACU"} <= ids
    assert not [i for i in ids if i.startswith("CD0.8")], "CD rungs must not be cells"
    strip = wm.build_ladder_strip(0, 0, 30000, 5000,
                                  rungs=wm.cd_rungs(RESOLUTION_LADDER_UM))
    assert strip.stats["n_rungs"] == len(RESOLUTION_LADDER_UM)
    assert strip.stats["n_below_floor"] > 0, "the ladder must cross the floor"


def test_labels_are_left_anchored_and_carry_the_value():
    placed, _ = layout(doe_cells())
    for p in placed:
        c = p.cell
        assert c.label, c.cid
        if c.axis and c.level:
            # At least one NUMBER from the level must appear in the etched
            # label, so a cell can be identified under a microscope without the
            # map. Levels are phrased freely ("base 4 um, spread 1.20", "5/5 um",
            # "0.8 -> 8.0 um"), so match on the numbers rather than on the
            # wording; a purely named level (plain / hue / zones) matches as a
            # word instead.
            nums = re.findall(r"\d+(?:\.\d+)?", c.level)
            lab = c.label.lower()
            if nums:
                assert any(n in lab for n in nums), f"{c.cid}: {c.label!r} vs {c.level!r}"
            else:
                assert c.level.split()[0].lower() in lab, f"{c.cid}: {c.label!r}"
    t = _text_rects("SP 44um", 0.0, 0.0, 600.0, anchor="left")
    assert t[:, 0].min() >= -1e-9, "left-anchored text starts at the anchor"
    assert _label_h(4000.0) < _label_h(30000.0), "label band scales with the cell"


# --- polarity: this is a darkfield write ------------------------------------


def test_the_plate_defaults_to_CLEAR_polarity():
    """Darkfield mask, positive resist: the write says where the chrome comes
    OFF, so the file holds the clear regions, not the metal."""
    plate = build_plate([[Cell("X", "t", "C", 2000.0, 2000.0,
                               lambda cx, cy, w, h: wc.build_grating_patch(
                                   cx, cy, w, h, period_um=10.0))]], verbose=False)
    assert plate["polarity"] == CLEAR


@pytest.mark.parametrize("kw", [
    dict(period_um=10.0, duty=0.5),
    dict(period_um=7.0, duty=0.3),
])
def test_analytic_inverses_tile_their_cell_exactly(kw):
    W = H = 4000.0
    m = wc.build_grating_patch(0, 0, W, H, polarity=METAL, **kw)
    k = wc.build_grating_patch(0, 0, W, H, polarity=CLEAR, **kw)
    assert _art_area(m) + _art_area(k) == pytest.approx(W * H, rel=1e-6)


@pytest.mark.parametrize("mode", ["plain", "zones"])
def test_the_halftone_inverse_tiles_the_lines_it_covers(mode):
    """Plain bands AND coloured bands: the clear sub-grating (duty 1-c at phase
    c*d) must be the exact complement of the metal one, by area."""
    W = H = 4000.0
    kw = dict(plan=_plan(mode), line_period_um=44.0, tone_steps=22)
    m = wc.build_halftone(0, 0, W, H, polarity=METAL, **kw)
    k = wc.build_halftone(0, 0, W, H, polarity=CLEAR, **kw)
    tiled = round(W / 44.0) * 44.0 * W
    # Plain bands tile exactly. Coloured bands tile to the centre-in stripe rule:
    # each polarity keeps whole stripes whose centre is inside the band, so the
    # two sets can differ by one 2.5 um stripe per band end, unbiased — a few
    # parts per million over a cell (measured 39 um2 in 16 mm2).
    assert _art_area(m) + _art_area(k) == pytest.approx(tiled, rel=1e-9 if mode == "plain" else 1e-5)
    if mode == "zones":
        assert m.arrays and k.arrays, "the zones plan must colour some bands"
        band = _area(m.arrays[0]["rects"])
        assert _array_area(m.arrays[0]) / band == pytest.approx(0.5, abs=0.01)
        assert _array_area(k.arrays[0]) / band == pytest.approx(0.5, abs=0.01)


def test_a_grating_and_its_inverse_are_complements():
    m = _grating_rects(0, 0, 1000.0, 400.0, 7.0, 0.5)
    c = invert_grating(0, 0, 1000.0, 400.0, 7.0, 0.5)
    assert _area(m) + _area(c) == pytest.approx(1000.0 * 400.0, rel=1e-9)


def test_the_boolean_inverter_refuses_a_cell_that_is_too_large():
    """The guard that keeps a per-cell boolean from becoming a whole-plate one."""
    n = wm.MAX_BOOLEAN_SHAPES + 1
    rects = np.tile(np.array([[0.0, 1.0, 0.0, 1.0]]), (n, 1))
    with pytest.raises(ValueError, match="analytic"):
        clear_by_boolean(0, 0, 10.0, 10.0, rects, [])


# --- the bench cells: each reports what it measures --------------------------


def test_the_step_wedge_quantises_to_the_screens_own_ladder():
    """A wedge asking for round-number duties would measure tones the plate
    cannot make."""
    a = wm.build_step_wedge(0, 0, 30000, 4000, line_period_um=44.0, tone_steps=22)
    assert all(isinstance(l, int) for l in a.stats["levels"])
    assert a.stats["levels"] == sorted(a.stats["levels"])
    assert min(a.stats["levels"]) >= 1 and max(a.stats["levels"]) <= 21, (
        "a wedge cannot ask for 0% or 100%: neither is printable"
    )
    assert a.stats["duties"] == sorted(a.stats["duties"])


def test_the_period_ladder_spans_from_unprintable_to_colourless():
    strip = wm.build_ladder_strip(0, 0, 30000, 6000,
                                  rungs=wm.period_rungs(PERIOD_LADDER_UM))
    lines = [m["line_um"] for m in strip.stats["rungs"]]
    assert min(lines) < MIN_FEATURE_UM, "must reach below the floor"
    assert max(PERIOD_LADDER_UM) >= 20.0, "and out to where there is no colour left"


def test_the_duty_ladder_brackets_the_second_order_null():
    strip = wm.build_ladder_strip(0, 0, 26000, 5000,
                                  rungs=wm.duty_rungs(10.0, C3_DUTY_LADDER))
    duties = [m["duty"] for m in strip.stats["rungs"]]
    assert 0.50 in duties and min(duties) < 0.5 < max(duties)
    assert all(m["clears_floor"] for m in strip.stats["rungs"]), (
        "a 10 um period keeps every rung of a 0.30-0.70 sweep printable"
    )


def test_the_glass_constants_are_the_plate_compositors():
    """The witness cells beat against THE box's carrier and comb, which the plate
    compositor derives from the glass. One source of truth, pinned."""
    from app import plates as P
    from app.witness_dies import blank_plan
    from app.witness_geom import BOX_CARRIER_UM, BOX_COMB_UM, BOX_FRONT_LEAF_UM, BOX_MONO_UM

    _, spec = blank_plan()
    rd = P._carrier_recipe_data(spec.faces["top"])
    assert rd["carrier_period_um"] == pytest.approx(BOX_CARRIER_UM)
    assert rd["slit_period_um"] == pytest.approx(BOX_FRONT_LEAF_UM)
    assert P.fab_center_period_um(spec.faces["front"]) == pytest.approx(BOX_COMB_UM)
    assert P._carrier_recipe_data(spec.faces["top"])["fab_center_period_um"] == pytest.approx(BOX_MONO_UM, rel=1e-6)
    assert spec.glass.thickness_um == PLY_UM and spec.glass.n == pytest.approx(1.4585)


def test_the_polarity_witness_is_asymmetric():
    """It has to be unmistakable under inversion, so it must not be its own
    complement in outline."""
    a = wm.build_polarity_witness(0, 0, 6000, 6000)
    assert 0.1 < _area(a.front) / (6000 * 6000) < 0.5


# --- the plate --------------------------------------------------------------


def test_build_plate_places_geometry_where_the_manifest_says():
    cells = [[Cell("X", "t", "C", 2000.0, 2000.0,
                   lambda cx, cy, w, h: wc.build_grating_patch(
                       cx, cy, w, h, period_um=10.0))]]
    plate = build_plate(cells, verbose=False, polarity=METAL)
    m = plate["manifest"][0]
    r = plate["front"]
    # the cell's own geometry (the saw-lane marks ride ``front`` too, at the
    # field's edge and on the cut lines; leave them out of this check)
    from app.export_witness import DICE_MARK_LEN_UM, DICE_MARK_W_UM
    is_mark = np.isclose(np.minimum(r[:, 1] - r[:, 0], r[:, 3] - r[:, 2]), DICE_MARK_W_UM) &         np.isclose(np.maximum(r[:, 1] - r[:, 0], r[:, 3] - r[:, 2]), DICE_MARK_LEN_UM)
    r = r[~is_mark]
    assert r[:, 0].min() == pytest.approx(m["x_mm"] * MM - 1000.0, abs=12.0)
    assert r[:, 2].min() == pytest.approx(m["y_mm"] * MM - 1000.0, abs=1.0)


def test_the_plate_stays_inside_the_glass():
    plate = build_plate([[Cell("X", "t", "C", 2000.0, 2000.0,
                               lambda cx, cy, w, h: wc.build_grating_patch(
                                   cx, cy, w, h, period_um=10.0))]],
                        verbose=False, polarity=METAL)
    half = BLANK_SIDE_UM / 2.0
    for key in ("front", "outline", "labels"):
        r = plate[key]
        if len(r):
            assert r[:, 0].min() >= -half and r[:, 1].max() <= half
            assert r[:, 2].min() >= -half and r[:, 3].max() <= half


def test_gds_and_oasis_are_both_written_and_readable(tmp_path):
    import klayout.db as kdb

    from app.export_witness import write_mask

    cells = [[Cell("X", "t", "D", 1200.0, 1200.0,
                   lambda cx, cy, w, h: wc.build_colour_band(cx, cy, w, h))]]
    plate = build_plate(cells, verbose=False, polarity=METAL)
    paths = write_mask(plate, tmp_path / "w", formats=(".oas", ".gds"))
    assert len(paths) == 2 and all(p.is_file() for p in paths)
    sizes = plate["gds"]["size_mb_by_format"]
    assert sizes["oas"] <= sizes["gds"], "OASIS must not be the larger format"
    for p in paths:
        ly = kdb.Layout()
        ly.read(str(p))
        assert ly.top_cell().name == "WITNESS_5IN"
        assert ly.top_cell().bbox().area() > 0


def test_the_plate_reports_what_flattening_would_cost():
    cells = [[Cell("X", "t", "D", 2000.0, 2000.0,
                   lambda cx, cy, w, h: wc.build_colour_band(cx, cy, w, h))]]
    plate = build_plate(cells, verbose=False, polarity=METAL)
    assert flat_rect_count(plate) > len(plate["front"])
    assert plate["manifest"][0]["n_array_bands"] > 0
def test_the_band_cells_are_banded_not_continuous():
    """At tone 0.5 with tone held the band was the whole 44 um period — a
    continuous grating with a phase reset, not a banded one. At tone 0.25 the
    held band is 22 um (4.4 periods of 5 um) and the unheld one 10 um (2.0), so
    the pair measures what holding tone costs."""
    held = wc.build_colour_band(0, 0, 6000, 6000, period_um=5.0, tone=0.25, hold_tone=True)
    loose = wc.build_colour_band(0, 0, 6000, 6000, period_um=5.0, tone=0.25, hold_tone=False)
    assert held.stats["band_um"] == pytest.approx(22.0)
    assert loose.stats["band_um"] == pytest.approx(10.0)
    assert held.stats["band_um"] < 44.0, "a full-period band is not a band"
    # 10 um at 5 um pitch is exactly two periods — the boundary the cell probes,
    # so the stat sits on the threshold; only the ordering is pinned here.
    assert held.stats["periods_per_band"] > loose.stats["periods_per_band"]
    assert loose.stats["periods_per_band"] == pytest.approx(2.0)
