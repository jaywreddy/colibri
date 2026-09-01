"""Single-blank bonded-pair panelization (export_blank) — pure math + a tiny
fake-geometry GDS build. No real pattern generation (that is the CLI's heavy
path); everything here runs in well under a second.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.assembly import FoilSpec, bonded_overlap_um
from app.export_blank import (
    BLANK_EDGE_MARGIN_UM,
    BLANK_SIDE_UM,
    BLANK_STREET_UM,
    BLANK_TAPE_WIDTH_UM,
    DEFAULT_ASPECT_H_OVER_W,
    PLATE_THICKNESS_UM,
    SUBPLATE_SUFFIXES,
    VERNIER_BAR_LEN_UM,
    VERNIER_PITCH_B_UM,
    VERNIER_PITCH_F_UM,
    build_blank_gds,
    dice_tick_rects,
    id_tick_rects,
    mirror_rects,
    pack_blank,
    pair_rects,
    solve_blank_max_scale,
    subplate_id,
    vernier_band_offset_um,
    vernier_blocks,
    write_blank_layout_svg,
)
from app.patterns.base import LITHO_FLOOR_UM

PLY = PLATE_THICKNESS_UM
FOLD = bonded_overlap_um(FoilSpec(tape_width_um=BLANK_TAPE_WIDTH_UM), PLY)


# --- rect building ----------------------------------------------------------


def test_pair_rects_nested_shell_dims():
    rects = pair_rects(30_000.0, 30_000.0, 33_000.0, PLY)
    assert len(rects) == 12
    by_id = {r.face: r for r in rects}
    for fid in ("front", "back", "top", "bottom", "left", "right"):
        f = by_id[subplate_id(fid, "F")]
        b = by_id[subplate_id(fid, "B")]
        # The inner ply is the nested shell: exactly one ply smaller per edge.
        assert b.width_um == pytest.approx(f.width_um - 2 * PLY)
        assert b.height_um == pytest.approx(f.height_um - 2 * PLY)


# --- packer -----------------------------------------------------------------


def _no_overlap_and_street(placements):
    for i, a in enumerate(placements):
        for b in placements[i + 1 :]:
            dx = max(a.x0, b.x0) - min(a.x0 + a.width_um, b.x0 + b.width_um)
            dy = max(a.y0, b.y0) - min(a.y0 + a.height_um, b.y0 + b.height_um)
            # Separated by >= one street on at least one axis.
            assert max(dx, dy) >= BLANK_STREET_UM - 1e-6, (a.face, b.face)


def test_pack_blank_respects_square_and_streets():
    rects = pair_rects(24_000.0, 24_000.0, 26_400.0, PLY)
    usable = BLANK_SIDE_UM - 2 * BLANK_EDGE_MARGIN_UM
    placements = pack_blank(rects, usable_side_um=usable)
    assert placements is not None and len(placements) == 12
    half = usable / 2.0
    for p in placements:
        assert p.x0 >= -half - 1e-6 and p.x0 + p.width_um <= half + 1e-6
        assert p.y0 >= -half - 1e-6 and p.y0 + p.height_um <= half + 1e-6
    _no_overlap_and_street(placements)


def test_pack_blank_refuses_oversize():
    rects = pair_rects(90_000.0, 90_000.0, 99_000.0, PLY)
    usable = BLANK_SIDE_UM - 2 * BLANK_EDGE_MARGIN_UM
    assert pack_blank(rects, usable_side_um=usable) is None


# --- solver -----------------------------------------------------------------


def test_solver_finds_an_upright_box_and_all_plates_fit():
    # Default plan = 12 plates + the 2 default spare pairs (front, back) = 16.
    result = solve_blank_max_scale()
    assert result is not None
    assert len(result.placements) == 16
    # A 5-inch blank holds at least a 27 mm upright box from 1.5 mm plies
    # even with both spare pairs aboard.
    assert result.width_um >= 27_000.0
    assert result.height_um == pytest.approx(result.width_um * DEFAULT_ASPECT_H_OVER_W)
    _no_overlap_and_street(result.placements)
    dims = result.dims_mm()
    # Ring-fit readout present and consistent with the bonded wall.
    assert dims["interior_width_mm"] == pytest.approx(
        dims["width_mm"] - 2 * dims["wall_mm"], abs=1e-3
    )
    assert result.wall_thickness_um == 2 * PLATE_THICKNESS_UM


def test_solver_scales_with_blank_and_thickness():
    five = solve_blank_max_scale()
    four = solve_blank_max_scale(blank_side_um=101_600.0)
    assert five is not None and four is not None
    assert five.width_um > four.width_um
    thin = solve_blank_max_scale(plate_thickness_um=700.0)
    assert thin is not None
    # Thinner plies still solve (they pack a slightly SMALLER max box — the
    # inner plates are nearly outer-sized, so total glass area grows).
    assert thin.width_um >= 25_000.0


# --- bench marks --------------------------------------------------------------


def test_mirror_rects_is_an_involution_and_flips_x():
    rects = np.array([[1.0, 3.0, -2.0, 5.0], [-4.0, -1.0, 0.0, 1.0]])
    m = mirror_rects(rects)
    assert np.allclose(m[0], [-3.0, -1.0, -2.0, 5.0])
    assert np.allclose(mirror_rects(m), rects)


def test_vernier_blocks_live_in_the_both_plies_band():
    w, h = 30_000.0, 33_000.0
    off = vernier_band_offset_um(PLY, FOLD)
    for pitch in (VERNIER_PITCH_F_UM, VERNIER_PITCH_B_UM):
        rects = vernier_blocks(w, h, PLY, FOLD, pitch)
        assert rects.shape[0] >= 4 * 8  # 4 combs, >= 8 lines each
        widths = np.minimum(rects[:, 1] - rects[:, 0], rects[:, 3] - rects[:, 2])
        assert widths.min() >= LITHO_FLOOR_UM
        cx = (rects[:, 0] + rects[:, 1]) / 2
        cy = (rects[:, 2] + rects[:, 3]) / 2
        edge_dist = np.minimum(w / 2 - np.abs(cx), h / 2 - np.abs(cy))
        # Comb centers sit in the interior fold band [ply, ply + fold] — the
        # only rim zone where BOTH plies have glass.
        assert (edge_dist >= PLY - 1e-6).all()
        assert (edge_dist <= PLY + FOLD + 1e-6).all()
        assert edge_dist.min() == pytest.approx(off, abs=VERNIER_BAR_LEN_UM)
        # Every rect fully inside the INNER ply footprint (the smaller plate).
        assert (np.abs(rects[:, :2]) <= w / 2 - PLY + 1e-6).all()
        assert (np.abs(rects[:, 2:]) <= h / 2 - PLY + 1e-6).all()
    assert VERNIER_PITCH_B_UM != VERNIER_PITCH_F_UM


def test_vernier_blocks_layout_is_mirror_symmetric():
    """The stack mirror must map the comb layout onto itself, or the F/B
    combs would land in different places after the chrome-down flip."""
    rects = vernier_blocks(30_000.0, 33_000.0, PLY, FOLD, VERNIER_PITCH_F_UM)
    mirrored = mirror_rects(rects)
    key = lambda a: sorted(map(tuple, np.round(a, 6)))
    assert key(mirrored) == key(rects)


def test_vernier_blocks_drop_when_fold_too_narrow_or_plate_tiny():
    assert vernier_blocks(30_000.0, 33_000.0, PLY, 300.0, VERNIER_PITCH_F_UM).shape[0] == 0
    assert vernier_blocks(6_000.0, 6_000.0, PLY, FOLD, VERNIER_PITCH_F_UM).shape[0] == 0


def test_id_ticks_encode_face_and_layer():
    w, h = 30_000.0, 33_000.0
    for idx in range(6):
        f = id_tick_rects(idx, False, w, h, PLY, FOLD)
        b = id_tick_rects(idx, True, w, h, PLY, FOLD)
        assert f.shape[0] == idx + 1
        assert b.shape[0] == idx + 2  # + underline bar
        for rects in (f, b):
            widths = np.minimum(rects[:, 1] - rects[:, 0], rects[:, 3] - rects[:, 2])
            assert widths.min() >= LITHO_FLOOR_UM
            # Inside the inner ply footprint like the verniers.
            assert (np.abs(rects[:, 2:]) <= h / 2 - PLY + 1e-6).all()


def test_dice_ticks_stay_in_the_street():
    from app.export_wafer import Placement

    p = Placement("front:F", 1_000.0, 2_000.0, 20_000.0, 16_000.0, False)
    ticks = dice_tick_rects(p)
    assert ticks.shape[0] == 8  # 4 corners x 2 legs of the L
    for x0, x1, y0, y1 in ticks:
        outside = (
            x1 <= p.x0 or x0 >= p.x0 + p.width_um
            or y1 <= p.y0 or y0 >= p.y0 + p.height_um
        )
        assert outside
        assert x0 >= p.x0 - BLANK_STREET_UM and x1 <= p.x0 + p.width_um + BLANK_STREET_UM
        assert y0 >= p.y0 - BLANK_STREET_UM and y1 <= p.y0 + p.height_um + BLANK_STREET_UM


# --- GDS build (fake fine geometry — fast) --------------------------------------


class _FakeFine:
    """Stand-in for export_fine.PlateFine: one small triangle per layer."""

    def __init__(self, slug: str):
        self.slug = slug
        tri = np.array([[0.0, 0.0], [900.0, 0.0], [0.0, 700.0]])
        self.front_polys = [tri]
        self.back_polys = [tri + np.array([50.0, 50.0])]
        self.stats: dict = {}


def test_build_blank_gds_single_chrome_layer(tmp_path: Path):
    import klayout.db as kdb

    out = tmp_path / "blank.gds"
    summary = build_blank_gds(
        out, _build_plate_fine=lambda spec, fid: _FakeFine(f"fake-{fid}")
    )
    assert out.exists()
    assert summary["total_polygons"] > 0
    # 12 originals + the 2 default spare pairs.
    assert len(summary["plates"]) == 16
    assert summary["spare_pairs"] == ["front", "back"]
    assert summary["stack"]["mirrored_for_stack"] is True
    assert summary["stack"]["inner_ply_inset_um"] == PLATE_THICKNESS_UM
    assert summary["glass"]["ply_um"] == PLATE_THICKNESS_UM
    # Glass-derived optics: 1.5 mm soda lime is (1500/1.52)/(500/1.46) ≈ 2.88x
    # the baseline gap, so the 60 µm barrier bakes at ~173 µm.
    assert summary["optics"]["switch_barrier_period_um"] == pytest.approx(173.0, abs=1.0)
    assert summary["verniers"]["amplification"] == pytest.approx(11.0)
    assert summary["verniers"]["beat_um"] == pytest.approx(880.0)

    ly = kdb.Layout()
    ly.read(str(out))
    top = ly.top_cell()
    chrome = ly.layer(10, 0)
    n_chrome = sum(1 for _ in top.shapes(chrome).each())
    assert n_chrome == summary["total_polygons"]
    labels = ly.layer(3, 0)
    texts = [s.text.string for s in top.shapes(labels).each() if s.is_text()]
    expected = [
        subplate_id(f, s)
        for f in ("front", "back", "top", "bottom", "left", "right")
        for s in SUBPLATE_SUFFIXES
    ] + ["front:F:spare1", "front:B:spare1", "back:F:spare2", "back:B:spare2"]
    assert sorted(texts) == sorted(expected)


def test_build_blank_gds_mirror_flag_flips_geometry(tmp_path: Path):
    """The same fake triangle written mirrored vs not must produce different
    chrome coordinates (the stack mirror is real, not a no-op)."""
    import klayout.db as kdb

    def _bbox_set(path: Path) -> set:
        ly = kdb.Layout()
        ly.read(str(path))
        top = ly.top_cell()
        chrome = ly.layer(10, 0)
        return {
            (s.dbbox().left, s.dbbox().bottom, s.dbbox().right, s.dbbox().top)
            for s in top.shapes(chrome).each()
        }

    fake = lambda spec, fid: _FakeFine(f"fake-{fid}")
    a = tmp_path / "m.gds"
    b = tmp_path / "u.gds"
    build_blank_gds(a, _build_plate_fine=fake, mirror_for_stack=True)
    build_blank_gds(b, _build_plate_fine=fake, mirror_for_stack=False)
    assert _bbox_set(a) != _bbox_set(b)


def test_blank_box_spec_is_bonded_with_derived_periods():
    from app.export_blank import blank_box_spec
    from app.plates import fab_center_period_um, water_scan_fab_pitch_um

    result = solve_blank_max_scale()
    assert result is not None
    spec = blank_box_spec(result)
    assert spec.bonded is True
    assert spec.glass.thickness_um == PLATE_THICKNESS_UM
    face = spec.faces["front"]
    # Outer-ply cut dims stamped; margins follow the bonded formulas.
    assert face.width_um == pytest.approx(result.width_um)
    assert face.weld_margin_um == pytest.approx(FOLD + 500.0)  # + default safety
    assert face.back_margin_um == pytest.approx(PLY + FOLD)
    # The moiré/switch periods rescale with the glass; the DESIGN carrier
    # pitch stays 22 (the fabricated pitch derives from it per-face below).
    assert fab_center_period_um(face) == pytest.approx(173.0, abs=1.0)
    assert water_scan_fab_pitch_um(face) == pytest.approx(173.0, abs=1.0)
    assert face.carrier_pitch_um == 22.0


def test_carrier_scale_mode_per_face():
    """Per-face carrier policy: 'gap' (default) scales the FABRICATED carrier
    family with t/n — controlled reveals on thick stock, no-op at baseline —
    while 'fixed' keeps the fine pitch (refraction shimmer). The capybara
    body-shimmer accent stays fixed in both modes."""
    from dataclasses import replace as dc_replace

    from app.export_blank import blank_box_spec
    from app.plates import GlassSpec, PlateSpec, _carrier_recipe_data

    result = solve_blank_max_scale()
    assert result is not None
    spec = blank_box_spec(result)

    # Default 'gap' on the thick soda-lime ply: 22 µm scales ~2.88x -> 63.5.
    face = spec.faces["top"]  # monogram-jp carrier reveal
    assert face.carrier_scale_mode == "gap"
    rd = _carrier_recipe_data(face)
    assert rd["fab_back_period_um"] == pytest.approx(63.5)
    assert rd["fab_front_period_um"] == pytest.approx(63.5 * 1.09)

    # 'fixed' keeps the literal design pitch (shimmer face).
    fixed = dc_replace(face, carrier_scale_mode="fixed")
    rd_fixed = _carrier_recipe_data(fixed)
    assert rd_fixed["fab_back_period_um"] == pytest.approx(22.0)

    # Baseline invariance: at 500 µm fused silica 'gap' is EXACTLY a no-op.
    baseline = PlateSpec(pattern_slug=face.pattern_slug)
    assert baseline.glass == GlassSpec()
    rd_base = _carrier_recipe_data(baseline)
    assert rd_base["fab_back_period_um"] == 22.0

    # Body-shimmer accent never scales.
    capy = spec.faces["back"]
    rd_capy = _carrier_recipe_data(capy)
    assert rd_capy["water_body_carrier_period_um"] == pytest.approx(24.0)


def test_layout_svg_writes(tmp_path: Path):
    result = solve_blank_max_scale()
    assert result is not None
    svg = write_blank_layout_svg(
        result.placements, tmp_path / "blank.svg",
        usable_side_um=result.usable_side_um,
    )
    text = svg.read_text(encoding="utf-8")
    assert text.count("<rect") >= 14  # blank + usable + 12 plates
    assert "front:F" in text and "back:B" in text


# --- bonded flags must survive the API + export bundle --------------------------


def test_api_body_models_keep_bonded_and_carrier_mode():
    """The Pydantic request models silently DROP unknown fields — this is the
    regression that shipped a bonded UI box to the backend as bonded=False."""
    from app.api.boxes import BoxSpecBody
    from app.api.plates import PlateSpecBody

    assert BoxSpecBody(bonded=True).to_spec().bonded is True
    assert BoxSpecBody().to_spec().bonded is False
    p = PlateSpecBody(pattern_slug="monogram-jp", carrier_scale_mode="fixed").to_spec()
    assert p.carrier_scale_mode == "fixed"
    assert PlateSpecBody(pattern_slug="monogram-jp").to_spec().carrier_scale_mode == "gap"


def test_export_bundle_docs_render_bonded_shape():
    from app.api.export import _assembly_md, _cutlist_csv
    from app.assembly import FoilSpec, HingeSpec, bonded_assembly_summary

    assembly = bonded_assembly_summary(
        30_500.0, 30_500.0, 33_500.0, 1_500.0, FoilSpec(), HingeSpec()
    )
    csv = _cutlist_csv(assembly)
    lines = csv.strip().splitlines()
    assert len(lines) == 1 + 12  # header + 12 plies
    assert ",outer," in lines[1] and ",inner," in lines[2]
    assert ",1.5," in lines[1]  # sheet thickness = one ply

    manifest = {
        "name": "t",
        "spec": {
            "foil": FoilSpec().to_dict(),
            "hinge": HingeSpec().to_dict(),
            "glass": {"material": "soda lime"},
        },
        "dimensions_um": {"width": 30_500.0, "depth": 30_500.0, "height": 33_500.0},
    }
    md = _assembly_md(manifest, assembly)
    assert "Bond the ply pairs" in md
    assert "BONDED two-ply faces" in md
    assert "soda lime" in md
    # Foil estimate counts each bonded pair once (outer perimeters only).
    outer_total = sum(
        2.0 * (e["width_um"] + e["height_um"])
        for e in assembly["cut_list"] if e["ply"] == "outer"
    ) / 1000.0
    assert f"{outer_total:.0f} mm" in md


# --- spare bonded pairs -----------------------------------------------------


def test_spare_pairs_pack_and_shrink_the_box():
    base = solve_blank_max_scale(spare_faces=())
    two = solve_blank_max_scale(spare_faces=("front", "back"))
    assert base is not None and two is not None
    assert len(two.placements) == 16
    ids = {p.face for p in two.placements}
    assert "front:F:spare1" in ids and "back:B:spare2" in ids
    # Spares cost box size, but the sweet spot stays a ring-size box.
    assert two.width_um < base.width_um
    assert two.width_um >= 27_000.0
    # Spare dims identical to their originals (bench-interchangeable).
    by = {p.face: p for p in two.placements}
    for sid, spare in (("front:F", "front:F:spare1"), ("back:B", "back:B:spare2")):
        a, b = by[sid], by[spare]
        assert {a.width_um, a.height_um} == {b.width_um, b.height_um}


def test_build_blank_gds_places_spares(tmp_path: Path):
    out = tmp_path / "blank_spares.gds"
    summary = build_blank_gds(
        out,
        spare_faces=("front",),
        _build_plate_fine=lambda spec, fid: _FakeFine(f"fake-{fid}"),
    )
    assert summary["spare_pairs"] == ["front"]
    plates = [r["plate"] for r in summary["plates"]]
    assert len(plates) == 14
    assert "front:F:spare1" in plates and "front:B:spare1" in plates
