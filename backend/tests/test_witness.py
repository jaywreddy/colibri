"""The 5-inch witness plate: does the DoE fit, and is each cell what it claims?

Deliberately built from SMALL cells rather than the shipping ones. The real
plate is 63 cells and 1.7 M rectangles, which is a 30 s build and gigabytes of
churn — exactly the kind of thing CLAUDE.md forbids putting in a test suite on
this host. What is worth pinning is the contracts: the packer fits, a cell lands
where the manifest says, the litho floor is reported honestly, and the array
path and the flat path describe the same geometry.
"""
from __future__ import annotations

import numpy as np
import pytest

from app import witness_cells as wc
from app.export_witness import (
    BASE_PERIOD_LADDER_UM,
    C3_DUTY_LADDER,
    COARSEN_LADDER_PX,
    DUTY_LADDER,
    SCALE_LADDER_MM,
    SCREEN_LADDER_UM,
    SPREAD_LADDER,
    STEPS_LADDER,
    MM,
    RESOLUTION_LADDER_UM,
    Placed,
    _plan,
    build_plate,
    doe_cells,
    flat_rect_count,
    layout,
)
from app.patterns.bitmap.colourzone import MIN_FEATURE_UM
from app.witness_geom import (
    GUTTER_UM,
    PLATE_SIDE_UM,
    USABLE_UM,
    Cell,
    CellArt,
    _frame_rects,
    _grating_rects,
    _text_rects,
)


def _rect_span(r: np.ndarray) -> tuple[float, float, float, float]:
    return r[:, 0].min(), r[:, 1].max(), r[:, 2].min(), r[:, 3].max()


# --- layout -----------------------------------------------------------------


def test_the_whole_design_of_experiments_fits_the_plate():
    """The reason the sizes are what they are. If a ladder gains a rung this is
    the test that says whether the plate still closes."""
    _, lay = layout(doe_cells())
    assert lay["fits"], f"overflow: {lay['overflow']}"
    assert lay["height_used_mm"] <= lay["height_available_mm"]
    assert lay["height_used_mm"] > 0.6 * lay["height_available_mm"], (
        "a plate this empty is wasting glass that could carry another ladder"
    )


def test_no_cell_escapes_the_usable_area():
    placed, _ = layout(doe_cells())
    half = USABLE_UM / 2.0
    for p in placed:
        right = (p.pair_cx if p.pair_cx is not None else p.cx) + p.cell.w_um / 2.0
        assert p.cx - p.cell.w_um / 2.0 >= -half - 1e-6, p.cell.cid
        assert right <= half + 1e-6, p.cell.cid
        assert abs(p.cy) + p.cell.h_um / 2.0 <= half + 1e-6, p.cell.cid


def test_cells_do_not_overlap():
    placed, _ = layout(doe_cells())
    boxes = []
    for p in placed:
        for cx in ([p.cx] if p.pair_cx is None else [p.cx, p.pair_cx]):
            boxes.append((p.cell.cid, cx - p.cell.w_um / 2, cx + p.cell.w_um / 2,
                          p.cy - p.cell.h_um / 2, p.cy + p.cell.h_um / 2))
    for i in range(len(boxes)):
        ai, ax0, ax1, ay0, ay1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            bi, bx0, bx1, by0, by1 = boxes[j]
            overlap = (ax0 < bx1 - 1e-6 and bx0 < ax1 - 1e-6
                       and ay0 < by1 - 1e-6 and by0 < ay1 - 1e-6)
            assert not overlap, f"{ai} overlaps {bi}"


def test_every_two_layer_cell_gets_a_pair_die():
    """One plate has nothing to register against, so Group A is emitted as a
    front die and a back die to be diced, flipped and bonded."""
    placed, _ = layout(doe_cells())
    two = [p for p in placed if p.cell.two_layer]
    assert two, "the plate should still carry the two-layer cells"
    for p in two:
        assert p.pair_cx is not None
        assert p.pair_cx > p.cx
        assert p.pair_cx - p.cx == pytest.approx(p.cell.w_um + GUTTER_UM)
    assert all(p.pair_cx is None for p in placed if not p.cell.two_layer)


def test_every_cell_id_is_unique():
    ids = [p.cell.cid for p in layout(doe_cells())[0]]
    assert len(ids) == len(set(ids))


def test_the_ladders_are_all_present():
    ids = {p.cell.cid for p in layout(doe_cells())[0]}
    assert {"B6a", "B6b", "B6c"} <= ids, "the three colour variants"
    for pre, ladder in (("C2", RESOLUTION_LADDER_UM), ("C3", C3_DUTY_LADDER),
                        ("CP", BASE_PERIOD_LADDER_UM), ("SP", SCREEN_LADDER_UM),
                        ("TS", STEPS_LADDER), ("DU", DUTY_LADDER),
                        ("SR", SPREAD_LADDER), ("GR", COARSEN_LADDER_PX),
                        ("SZ", SCALE_LADDER_MM)):
        got = [i for i in ids if i.startswith(pre)]
        assert len(got) == len(ladder), f"{pre}: {sorted(got)}"
    assert {"A1", "A2", "A4", "A6"} <= ids


def test_every_sweep_cell_is_labelled_with_its_VALUE():
    """A witness plate is read under a microscope, where every sweep cell looks
    like every other one and the map is elsewhere. "K3" is unreadable there."""
    import re

    for p in layout(doe_cells())[0]:
        if p.cell.axis and p.cell.level and p.cell.group == "B":
            assert p.cell.label, p.cell.cid
            head = p.cell.level.split()[0]
            digits = re.sub(r"[^0-9.]", "", head)
            # A numeric level must appear as a number; a named one (plain / hue /
            # zones) as the word.
            want = digits if digits else head
            assert want.lower() in p.cell.label.lower(), (
                f"{p.cell.cid}: label {p.cell.label!r} does not carry "
                f"level {p.cell.level!r}"
            )


# --- rect primitives --------------------------------------------------------


def test_a_grating_patch_fills_its_box_and_holds_its_duty():
    r = _grating_rects(0.0, 0.0, 1000.0, 400.0, 10.0, 0.5)
    x0, x1, y0, y1 = _rect_span(r)
    assert x0 >= -500.0 - 1e-9 and x1 <= 500.0 + 1e-9
    assert (y0, y1) == (-200.0, 200.0)
    area = ((r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2])).sum()
    assert area == pytest.approx(1000.0 * 400.0 * 0.5, rel=0.01)


def test_text_and_frame_stay_inside_what_they_annotate():
    t = _text_rects("B6c", 0.0, 0.0, 900.0)
    assert len(t) > 0
    x0, x1, y0, y1 = _rect_span(t)
    assert (y1 - y0) == pytest.approx(900.0, rel=0.35)
    f = _frame_rects(0.0, 0.0, 1000.0, 1000.0, 60.0)
    assert len(f) == 4
    assert _rect_span(f) == pytest.approx((-500.0, 500.0, -500.0, 500.0))


# --- cell builders ----------------------------------------------------------


def test_the_resolution_ladder_reports_which_rungs_break_the_floor():
    """C2 is the plate's highest-value cell precisely because some of its rungs
    are meant to fail; the manifest has to say which."""
    below = wc.build_grating_patch(0, 0, 2000, 2000, period_um=2.0)
    above = wc.build_grating_patch(0, 0, 2000, 2000, period_um=6.0)
    assert below.stats["clears_litho_floor"] is False
    assert below.stats["line_um"] < MIN_FEATURE_UM
    assert above.stats["clears_litho_floor"] is True


def test_the_duty_ladder_brackets_the_second_order_null():
    duties = [wc.build_grating_patch(0, 0, 2000, 2000, period_um=8.0,
                                     duty=c).stats for c in C3_DUTY_LADDER]
    assert [d["duty"] for d in duties] == list(C3_DUTY_LADDER)
    assert all(d["clears_litho_floor"] for d in duties), (
        "an 8 um period keeps every rung of a 0.30-0.70 duty sweep printable"
    )


def test_the_chirp_crosses_the_litho_floor_inside_the_patch():
    a = wc.build_chirp(0, 0, 5000, 2000)
    assert a.stats["period_end_um"] < a.stats["crosses_floor_at_um"] < a.stats[
        "period_start_um"], "the sweep must actually cross the floor to be a check"
    assert len(a.front) > 10


def test_a_colour_band_cell_reports_whether_it_can_hold_a_spectrum():
    """A band narrower than a couple of grating periods is a pair of slits, not
    a grating — the cell must say so rather than be discovered on glass."""
    wide = wc.build_colour_band(0, 0, 2000, 2000, tone=0.5, period_um=5.0)
    assert wide.stats["enough_periods_for_a_spectrum"] is True
    assert wide.stats["clears_litho_floor"] is True
    dark = wc.build_colour_band(0, 0, 2000, 2000, tone=0.05, period_um=5.0)
    assert dark.stats["enough_periods_for_a_spectrum"] is False


def test_holding_tone_widens_the_band():
    held = wc.build_colour_band(0, 0, 2000, 2000, tone=0.4, hold_tone=True)
    loose = wc.build_colour_band(0, 0, 2000, 2000, tone=0.4, hold_tone=False)
    assert held.stats["band_um"] > loose.stats["band_um"]


def test_the_barrier_switch_is_registered_a_quarter_period_over():
    """The free improvement: as first built the slit straddled a lane boundary
    at rest, so head-on showed a blend and nothing read until you tilted."""
    a = wc.build_barrier_switch(0, 0, 4000, 4000, comb_um=173.0)
    assert a.stats["quarter_period_registered"] is True
    assert a.stats["peak_shift_um"] == pytest.approx(173.0 / 4.0)
    assert len(a.front) and len(a.back)


def test_the_shading_moire_solves_the_front_pitch_from_the_BEAT():
    """Pinning delta instead of the beat is what made an earlier jamon cell read
    0.075 where it should have read 0.47."""
    a = wc.build_shading_moire(0, 0, 8000, 8000, back_period_um=63.5, beat_um=1635.0)
    p, d = a.stats["back_period_um"], a.stats["delta_um"]
    assert p * (p + d) / d == pytest.approx(1635.0, rel=1e-3)
    b = wc.build_shading_moire(0, 0, 8000, 8000, beat_um=3000.0)
    assert b.stats["delta_um"] < a.stats["delta_um"], "a longer beat is a finer detune"


def test_the_moire_magnifier_sampler_is_gold_WITH_holes():
    """Transmission is (1-f)(1-b), so sparse dots read as a grey field and
    nothing floats."""
    a = wc.build_moire_magnifier(0, 0, 3000, 3000, sampler_um=60.0, motif_um=62.0)
    assert a.stats["magnification"] == pytest.approx(-30.0, abs=0.5)
    area = ((a.front[:, 1] - a.front[:, 0]) * (a.front[:, 3] - a.front[:, 2])).sum()
    assert area > 0.6 * 3000 * 3000, "the sampler layer must be mostly opaque"


def test_the_vernier_amplifies_a_real_offset():
    a = wc.build_vernier(0, 0, 4000, 2000)
    assert a.stats["amplification"] == pytest.approx(10.0, abs=0.1)


# --- the halftone cell, small ------------------------------------------------


def test_a_plain_and_a_zoned_cell_differ_only_in_the_colour(tmp_path):
    """Same code path, same tone, so any difference on the plate is the colour."""
    kw = dict(line_period_um=44.0, tone_steps=22)
    plain = wc.build_halftone(0, 0, 2000, 2000, plan=_plan("plain"), **kw)
    zones = wc.build_halftone(0, 0, 2000, 2000, plan=_plan("zones"), **kw)
    assert plain.stats["mode"] == "plain" and not plain.arrays
    assert zones.stats["frac_coloured"] > 0.1
    assert zones.arrays, "colour must be deferred as arrays, not expanded"
    # the band geometry is the same screen; colour only SPLITS runs
    assert zones.stats["n_band_rects"] >= plain.stats["n_band_rects"]
    assert zones.stats["tone_steps"] == plain.stats["tone_steps"]


def test_tone_steps_are_clamped_to_the_litho_floor_not_trusted():
    a = wc.build_halftone(0, 0, 2000, 2000, plan=_plan("plain"),
                          line_period_um=20.0, tone_steps=22)
    assert a.stats["tone_steps"] == 10, "20 um / 2 um floor allows 10 levels"
    assert a.stats["finest_band_um"] >= MIN_FEATURE_UM - 1e-9


def test_the_asset_resolution_not_the_plate_bounds_the_rectangle_count():
    """The lever to reach for when a cell is too heavy: prep the asset smaller,
    not the plate. Doubling the extent must not double rects-per-line."""
    # Big enough that asset_px_for is not sitting on its 160 px floor, or the
    # two cells would prep the same asset and the test would prove nothing.
    small = wc.build_halftone(0, 0, 6000, 6000, plan=_plan("plain"))
    big = wc.build_halftone(0, 0, 12000, 12000, plan=_plan("plain"))
    assert big.stats["asset_px"] > small.stats["asset_px"]
    assert big.stats["eye_cells_across"] == pytest.approx(
        2 * small.stats["eye_cells_across"], abs=1)  # int() floor


def test_the_headline_cell_is_a_photograph_not_a_thumbnail():
    from app.export_witness import PORTRAIT_MM

    assert PORTRAIT_MM * MM / 87.0 > 150, (
        "under ~150 eye-cells the portrait reads as a thumbnail, which is the "
        "whole reason it needed more than the 3.95 mm die"
    )


# --- array path vs flat path -------------------------------------------------


def test_deferring_to_arrays_describes_the_same_geometry_as_expanding_it():
    kw = dict(line_period_um=44.0, tone_steps=22, plan=_plan("zones"))
    lazy = wc.build_halftone(0, 0, 1500, 1500, defer_arrays=True, **kw)
    flat = wc.build_halftone(0, 0, 1500, 1500, defer_arrays=False, **kw)
    assert lazy.stats["n_stripes_if_flat"] == flat.stats["n_stripes_if_flat"]
    assert len(flat.front) == len(lazy.front) + lazy.stats["n_stripes_if_flat"]


def test_the_plate_reports_what_flattening_would_cost():
    cells = [[Cell("X", "t", "D", 2000.0, 2000.0,
                   lambda cx, cy, w, h: wc.build_colour_band(cx, cy, w, h))]]
    plate = build_plate(cells, verbose=False)
    assert flat_rect_count(plate) > len(plate["front"])
    assert plate["manifest"][0]["n_array_bands"] > 0


def test_build_plate_places_geometry_where_the_manifest_says():
    cells = [[Cell("X", "t", "C", 2000.0, 2000.0,
                   lambda cx, cy, w, h: wc.build_grating_patch(
                       cx, cy, w, h, period_um=10.0))]]
    plate = build_plate(cells, verbose=False)
    m = plate["manifest"][0]
    x0, x1, y0, y1 = _rect_span(plate["front"])
    assert x0 == pytest.approx(m["x_mm"] * MM - 1000.0, abs=12.0)
    assert y0 == pytest.approx(m["y_mm"] * MM - 1000.0, abs=1.0)


def test_the_plate_stays_inside_the_glass():
    plate = build_plate([[Cell("X", "t", "C", 2000.0, 2000.0,
                               lambda cx, cy, w, h: wc.build_grating_patch(
                                   cx, cy, w, h, period_um=10.0))]], verbose=False)
    half = PLATE_SIDE_UM / 2.0
    for key in ("front", "outline", "labels"):
        r = plate[key]
        if len(r):
            assert r[:, 0].min() >= -half and r[:, 1].max() <= half
            assert r[:, 2].min() >= -half and r[:, 3].max() <= half


def test_gds_is_written_and_is_readable(tmp_path):
    from app.export_witness import write_gds

    cells = [[Cell("X", "t", "D", 1200.0, 1200.0,
                   lambda cx, cy, w, h: wc.build_colour_band(cx, cy, w, h))]]
    plate = build_plate(cells, verbose=False)
    out = write_gds(plate, tmp_path / "w.gds")
    assert out.is_file() and out.stat().st_size > 0
    assert plate["gds"]["n_array_instances"] > 0
    assert plate["gds"]["n_unit_cells"] > 0

    import klayout.db as kdb

    ly = kdb.Layout()
    ly.read(str(out))
    assert ly.top_cell().name == "WITNESS_5IN"
    # The arrays must actually contain geometry once flattened.
    assert ly.top_cell().bbox().area() > 0
