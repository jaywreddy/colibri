"""The 5-inch witness plate: does the DoE fit, and is each cell what it claims?

Built from SMALL cells wherever a shipping-size one would be expensive. The real
plate is 78 cells and half a million rectangles, a 10 s build; what is worth
pinning is the contracts — the packer fits, the physics formulas are the ones
``docs/witness-physics-plan.md`` states, the litho floor is reported honestly,
and **metal and clear are exact complements**, which on a darkfield write is the
difference between the part and its negative.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pytest

from app import witness_cells as wc
from app import witness_moire as wm
from app.export_witness import (
    BEAT_LADDER_UM,
    C3_DUTY_LADDER,
    MM,
    NEAR_FIELD_LADDER_UM,
    PERIOD_LADDER_UM,
    RESOLUTION_LADDER_UM,
    ROTATION_LADDER_DEG,
    _beat_w_mm,
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
    GUTTER_UM,
    METAL,
    PLATE_SIDE_UM,
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
        dies = [(p.cx, p.cell.w_um, p.cell.h_um)]
        if p.pair_cx is not None:
            dies.append((p.pair_cx, *p.cell.back_dims))
        for cx, w, h in dies:
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
    bw, bh = c.back_dims
    return (c.w_um * c.h_um + (bw * bh if c.two_layer else 0.0)) / 1e6


def test_the_area_budget_matches_the_plan():
    """docs/production-plate-plan.md section 2: the four production dies are
    about a third of the field, moiré is the largest EXPERIMENT block, there
    are no portrait cells (the colour sides are the portraits), and the
    experiments that need a bond stay a minority so a failed bond still
    returns most of the numbers."""
    placed, _ = layout(doe_cells())
    area: dict[str, float] = {}
    for p in placed:
        area[p.cell.block] = area.get(p.cell.block, 0.0) + _written_mm2(p.cell)
    usable = (USABLE_UM / MM) ** 2
    assert 0.25 < area["production"] / usable < 0.40, area
    exp = {k: v for k, v in area.items() if k != "production"}
    assert max(exp, key=exp.get) == "moire", exp
    assert not [p.cell.cid for p in placed if p.cell.cid.startswith(("PORT", "SZ"))]
    two = sum(_written_mm2(p.cell) for p in placed
              if p.cell.two_layer and p.cell.block != "production")
    assert two / sum(exp.values()) < 0.40, "most experiments must survive a missing bond"


def test_the_production_dies_are_the_panelized_box_plies():
    """A die on this plate must be interchangeable with one from the full
    ``export_blank`` panel: same solve, same cut dims, F and B for the two
    bonded faces, F only for the colour sides."""
    from app import export_blank as eb
    from app.witness_dies import blank_plan, die_dims, production_cells

    result, spec = blank_plan()
    assert result.plate_thickness_um == 1500.0
    cells = {c.cid: c for c in production_cells()}
    assert set(cells) == {"DIE-TOP", "DIE-FRONT", "DIE-LEFT", "DIE-RIGHT"}
    for face in ("top", "front"):
        c, d = cells[f"DIE-{face.upper()}"], die_dims(face)
        assert c.two_layer and c.takes_polarity
        assert (c.w_um, c.h_um) == (d["f_w"], d["f_h"])
        assert c.back_dims == (d["b_w"], d["b_h"])
        assert d["b_w"] == pytest.approx(d["f_w"] - 2 * 1500.0), "inner ply inset one ply per edge"
    for face in ("left", "right"):
        c = cells[f"DIE-{face.upper()}"]
        assert not c.two_layer and c.takes_polarity
        assert c.back_dims == (c.w_um, c.h_um)
    panel = {r.face: (r.width_um, r.height_um)
             for r in eb.pair_rects(result.width_um, result.depth_um,
                                    result.height_um, result.plate_thickness_um)}
    assert panel["top:F"] == (cells["DIE-TOP"].w_um, cells["DIE-TOP"].h_um)
    assert panel["left:F"] == (cells["DIE-LEFT"].w_um, cells["DIE-LEFT"].h_um)


def test_the_die_inversion_is_the_exact_complement_of_its_metal():
    """``die box - metal`` through the klayout Region, decomposed to hole-free
    trapezoids: the pieces tile the box with the metal exactly (area), and no
    piece carries a hole or leaves the box."""
    from app.witness_dies import clear_field

    W, H = 3000.0, 2000.0
    tri = np.array([[-1400.0, -900.0], [-1000.0, -900.0], [-1200.0, -500.0]])
    metal = [_grating_rects(0.0, 0.0, 2000.0, 1200.0, 40.0, 0.5), tri]
    clear = clear_field(W, H, metal)
    a_metal = _area(metal[0]) + _poly_area(tri)
    a_clear = sum(_poly_area(p) for p in clear)
    assert a_metal + a_clear == pytest.approx(W * H, rel=1e-6)
    for p in clear:
        assert len(p) <= 4, "trapezoids only"
        assert p[:, 0].min() >= -W / 2 - 1e-6 and p[:, 0].max() <= W / 2 + 1e-6
        assert p[:, 1].min() >= -H / 2 - 1e-6 and p[:, 1].max() <= H / 2 + 1e-6


def test_a_two_layer_cell_may_have_a_smaller_back_die():
    """A bonded box face: the inner ply is inset one ply per edge. The packer
    must span the pair at its true widths and the plate must frame, label and
    place the back die at its own size."""
    def build(cx, cy, w, h, polarity=METAL):
        art = wc.build_grating_patch(cx, cy, w, h, period_um=10.0)
        art.back = _grating_rects(cx, cy, w - 600.0, h - 600.0, 10.0, 0.5)
        art.back_polys = [np.array([[cx - 100, cy - 100], [cx + 100, cy - 100], [cx, cy + 100]])]
        return art
    c = Cell("PAIR", "t", "X", 3000.0, 3000.0, build, two_layer=True, takes_polarity=True,
             back_w_um=2400.0, back_h_um=2400.0)
    placed, lay = layout([[c]])
    p = placed[0]
    assert p.pair_cx == pytest.approx(p.cx + 1500.0 + GUTTER_UM + 1200.0)
    plate = build_plate([[c]], verbose=False, polarity=METAL)
    m = plate["manifest"][0]
    assert (m["back_w_mm"], m["back_h_mm"]) == (2.4, 2.4)
    dx = p.pair_cx - p.cx
    tri = [pv for pv in plate["free_polys"] if len(pv) == 3]
    assert len(tri) == 1 and tri[0][:, 0].mean() == pytest.approx(p.cx + dx)
    pm = plate["pair_marks"]
    assert pm[:, 0].min() == pytest.approx(p.pair_cx - 1200.0, abs=1.0)


def test_short_cells_stack_beside_tall_ones_instead_of_costing_the_row():
    """Pockets and columns: a 4 mm cell placed after a 28 mm one must not add
    28 mm of row height. Four 4 mm cells beside one tall cell fit in the tall
    cell's own row."""
    def g(cx, cy, w, h):
        return wc.build_grating_patch(cx, cy, w, h, period_um=10.0)
    tall = Cell("TALL", "t", "X", 20000.0, 28000.0, g)
    shorts = [Cell(f"S{i}", "t", "X", 30000.0, 4000.0, g) for i in range(4)]
    placed, lay = layout([[tall], shorts])
    assert len(lay["rows"]) == 1, lay["rows"]
    assert lay["height_used_mm"] < 31.0
    ys = sorted({round(p.cy) for p in placed if p.cell.cid.startswith("S")})
    assert len(ys) >= 2, "the short cells stacked in a column or pocket"


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


def test_the_crossed_grating_opens_exactly_the_square_of_its_gap():
    """Two orthogonal gratings OVERLAP, so their rectangle areas cannot simply
    be added: two 50% gratings sum to 100% of a cell whose union is 75%. The
    clear fraction is the one unambiguous number — ``(1 - c)^2`` — and it is
    also the check that the union identity behaves."""
    import klayout.db as kdb

    W = H = 4000.0
    c = 0.5
    m = wm.build_crossed(0, 0, W, H, period_x_um=20.0, period_y_um=25.0,
                         duty=c, polarity=METAL)
    k = wm.build_crossed(0, 0, W, H, period_x_um=20.0, period_y_um=25.0,
                         duty=c, polarity=CLEAR)
    assert _art_area(k) == pytest.approx((1 - c) ** 2 * W * H, rel=1e-3)
    reg = kdb.Region()
    for x0, x1, y0, y1 in m.front:
        reg.insert(kdb.Box(round(x0 * 1000), round(y0 * 1000),
                           round(x1 * 1000), round(y1 * 1000)))
    reg.merge()
    assert reg.area() * 1e-6 == pytest.approx((1 - (1 - c) ** 2) * W * H, rel=1e-3)


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


def test_the_boolean_inverse_tiles_its_cell_exactly():
    import klayout.db as kdb

    W = H = 4000.0
    m = wm.build_beat(0, 0, W, H, beat_um=1000.0)
    free = clear_by_boolean(0, 0, W, H, m.front, m.polys)
    reg = kdb.Region()
    for x0, x1, y0, y1 in m.front:
        reg.insert(kdb.Box(round(x0 * 1000), round(y0 * 1000),
                           round(x1 * 1000), round(y1 * 1000)))
    reg.merge()
    metal = reg.area() * 1e-6
    assert metal + sum(_poly_area(p) for p in free) == pytest.approx(W * H, rel=1e-6)


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


def test_rotated_geometry_is_CLIPPED_to_its_cell_not_merely_rejected():
    """A rotated grating's end lines overhang by up to half the diagonal. Left
    unclipped they spill into the neighbouring cell, and the clear-field inverse
    stops being the metal's complement — which the area check caught."""
    W = H = 4000.0
    a = wm.build_rotation_beat(0, 0, W, H, angle_deg=8.0)
    assert a.polys, "a rotated cell must produce polygons"
    for pv in a.polys:
        assert pv[:, 0].min() >= -W / 2 - 1e-6 and pv[:, 0].max() <= W / 2 + 1e-6
        assert pv[:, 1].min() >= -H / 2 - 1e-6 and pv[:, 1].max() <= H / 2 + 1e-6


def test_clipping_a_convex_polygon_is_exact():
    p = np.array([[-2.0, -2.0], [3.0, -2.0], [3.0, 3.0], [-2.0, 3.0]])
    assert _poly_area(wm._clip_convex(p, 0, 0, 2, 2)) == pytest.approx(4.0)


# --- physics ----------------------------------------------------------------


def test_the_beat_is_solved_from_the_beat_not_from_delta():
    for b in BEAT_LADDER_UM:
        d = wm.beat_delta_for(63.5, b)
        assert 63.5 * (63.5 + d) / d == pytest.approx(b, rel=1e-9)
    with pytest.raises(ValueError):
        wm.beat_delta_for(63.5, 50.0)


def test_the_beat_amplifies_a_pitch_error():
    """What makes B-BEAT a metrology cell and not just a pretty one: the beat is
    p/delta times the pitch difference, so it magnifies a pitch error by 25."""
    a = wm.build_beat(0, 0, 8000, 8000, beat_um=1635.0)
    assert a.stats["amplification"] == pytest.approx(24.7, abs=0.2)
    assert a.stats["single_layer"] is True


def test_rotation_and_vector_beats_agree_where_they_overlap():
    """``p/(2 sin(a/2))`` is the equal-pitch case of ``|k1 - k2|``; if they
    disagreed the perimeter frame's numbers would be wrong."""
    for a in ROTATION_LADDER_DEG:
        rot = 63.5 / (2 * math.sin(math.radians(a) / 2))
        assert wm.combined_beat_um(63.5, 63.5, a) == pytest.approx(rot, rel=1e-9)
    assert wm.combined_beat_um(63.5, 66.07, 0.0) == pytest.approx(
        63.5 * 66.07 / 2.57, rel=1e-3)


def test_the_even_harmonic_moire_exists_ONLY_under_duty_bias():
    """The best argument for the duty ladder. The (2,3) beat between the 44 µm
    screen and the 63.5 µm carrier is 559 µm — 6.4 arcmin, plainly visible — and
    its amplitude is identically zero at 50% duty, because even harmonics of a
    square wave vanish there. A process error conjures banding across the
    photograph that the nominal design does not have."""
    at50 = {(t["m"], t["n"]): t for t in wm.harmonic_beats(44.0, 63.5, 0.50)}
    at42 = {(t["m"], t["n"]): t for t in wm.harmonic_beats(44.0, 63.5, 0.42)}
    assert at50[(2, 3)]["arcmin"] == pytest.approx(6.42, abs=0.05)
    assert at50[(2, 3)]["amplitude"] == 0.0
    assert at42[(2, 3)]["amplitude"] > 0.0
    assert at50[(1, 1)]["arcmin"] == pytest.approx(1.65, abs=0.03)
    assert at50[(1, 1)]["amplitude"] > 0.1

    nominal = wm.build_harmonic(0, 0, 8000, 8000, duty=0.50)
    biased = wm.build_harmonic(0, 0, 8000, 8000, duty=0.42)
    assert nominal.stats["n_visible"] < biased.stats["n_visible"]


def test_the_near_field_ladder_brackets_the_fresnel_boundary():
    """Two-layer effects need a coarse pitch: a p/2 slit spreads by lambda*z/(n*p)
    across the gap, and the ladder must straddle p_min = sqrt(2*lambda*z/n) or it
    measures nothing. (Coherent Talbot self-imaging is NOT the criterion; its
    quarter distance is where a 50% grating's shadow vanishes.)"""
    verdicts = [wm.build_near_field(0, 0, 8000, 8000, period_um=p).stats["predicted"]
                for p in NEAR_FIELD_LADDER_UM]
    assert "washed out" in verdicts and "intact" in verdicts, verdicts
    ref = wm.build_near_field(0, 0, 8000, 8000, period_um=44.0).stats
    assert ref["p_min_um"] == pytest.approx(32.9, abs=0.5), "1.5 mm soda-lime pair, the box stock"
    assert ref["fresnel_number"] == pytest.approx(0.89, abs=0.02)
    fine = wm.build_near_field(0, 0, 8000, 8000, period_um=5.0).stats
    assert fine["fresnel_number"] < 0.05, "a colour grating is far past it"


def test_the_swatch_reports_when_its_blue_end_is_unprintable():
    ok = wm.build_swatch(0, 0, 6000, 6000, base_period_um=5.0, spread=1.45)
    assert ok.stats["all_printable"] is True
    bad = wm.build_swatch(0, 0, 6000, 6000, base_period_um=4.0, spread=1.90)
    assert bad.stats["all_printable"] is False
    assert bad.stats["finest_line_um"] < MIN_FEATURE_UM


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


def test_beat_cells_are_wide_not_square():
    """Beat fringes are spaced along ONE axis, so these cells need width, not
    area. Drawn square, the 6000 µm rung forced a 20.9 mm row that was a third
    full and cost the plate 10 mm of height for one cell."""
    placed, _ = layout(doe_cells())
    beats = [p.cell for p in placed if p.cell.cid.startswith("BEAT")]
    assert beats
    heights = {c.h_um for c in beats}
    assert len(heights) == 1, "every beat cell shares one height, so they pack"
    for c in beats:
        b = float(c.cid[4:])
        assert c.w_um / b >= 3.0, f"{c.cid} holds only {c.w_um/b:.1f} fringes"
    # width tracks the beat; height does not
    assert _beat_w_mm(6000.0) > _beat_w_mm(500.0)
    widest = max(beats, key=lambda c: c.w_um)
    assert widest.w_um > widest.h_um, "the coarsest beat must be wide, not tall"


def test_the_parallax_ruler_is_readable_by_hand():
    """At 17.2 µm/deg (1.5 mm soda lime) a 200 µm tooth needs 11.6 deg of tilt,
    so a hand-held read would cover one tooth. 60 µm gives 3.5 deg per tooth,
    five inside ±10 deg."""
    a = wm.build_parallax_ruler(0, 0, 10000, 6000)
    assert 2.0 < a.stats["deg_per_tooth"] < 4.0
    assert a.stats["parallax_um_per_deg"] == pytest.approx(17.22, abs=0.05)
    assert a.stats["single_layer"] is False


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
    assert r[:, 0].min() == pytest.approx(m["x_mm"] * MM - 1000.0, abs=12.0)
    assert r[:, 2].min() == pytest.approx(m["y_mm"] * MM - 1000.0, abs=1.0)


def test_the_plate_stays_inside_the_glass():
    plate = build_plate([[Cell("X", "t", "C", 2000.0, 2000.0,
                               lambda cx, cy, w, h: wc.build_grating_patch(
                                   cx, cy, w, h, period_um=10.0))]],
                        verbose=False, polarity=METAL)
    half = PLATE_SIDE_UM / 2.0
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


def test_the_barrier_switch_is_straddle_registered_wherever_the_cell_sits():
    """The comb must be anchored to the CELL, not the plate origin. Anchored to
    the origin, head-on registration was x0 mod p — zero for three combs and
    65 um for the shipping 173 um one, so that cell alone came out 25/75 and
    swapped at 0.79 and 2.37 deg instead of a symmetric pair. Found by a
    reviewer recomputing the cell, not by any test."""
    for comb in (100.0, 173.0, 250.0, 350.0):
        for x_left in (0.0, 16500.0, 12345.0):
            a = wc.build_barrier_switch(x_left + 4000, 0, 8000, 8000, comb_um=comb)
            assert a.stats["head_on_A_fraction"] == pytest.approx(0.5, abs=1e-6), (comb, x_left)
            assert a.stats["peak_shift_um"] == pytest.approx(comb / 4.0)


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
