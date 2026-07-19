"""Assembly math for the stained-glass ring box (app/assembly.py).

Pure-geometry tests pin the contract formulas to exact numbers for the
default 50 x 50 x 40 mm ring box with 0.5 mm glass and 1/4" copper foil;
the materialize tests check the derived block lands in the box manifest.
"""
from __future__ import annotations

import json

import pytest

from app.assembly import (
    FACE_IDS,
    FoilSpec,
    HingeSpec,
    assembly_summary,
    cut_list,
    face_cut_dims,
    hinge_layout,
    keepout_um,
    overlap_um,
    seam_list,
    validate_assembly,
)


# Default ring box: W=D=50 mm, H=40 mm, t=0.5 mm.
W, D, H, T = 50000.0, 50000.0, 40000.0, 500.0


# ----- foil keep-out ----------------------------------------------------------


def test_overlap_quarter_inch_tape_on_half_mm_glass():
    # (6350 - 500) / 2 = 2925 um folds onto each plate face.
    assert overlap_um(FoilSpec(), T) == pytest.approx(2925.0)


def test_keepout_is_overlap_plus_safety():
    # 2925 overlap + 500 safety = 3425 um blank rim per edge.
    assert keepout_um(FoilSpec(), T) == pytest.approx(3425.0)


def test_overlap_clamps_at_zero_for_thick_glass():
    foil = FoilSpec(tape_width_um=4763.0)
    assert overlap_um(foil, 5000.0) == 0.0
    assert keepout_um(foil, 5000.0) == foil.safety_um


# ----- cut list ---------------------------------------------------------------


def test_cut_list_exact_numbers_for_default_ring_box():
    entries = {e["face"]: e for e in cut_list(W, D, H, T)}
    assert len(entries) == 6
    # bottom / top (lid): full W x D footprint.
    for fid in ("bottom", "top"):
        assert (entries[fid]["width_um"], entries[fid]["height_um"]) == (50000.0, 50000.0)
        assert (entries[fid]["width_mm"], entries[fid]["height_mm"]) == (50.0, 50.0)
    # front / back walls: W wide, lose a glass thickness top AND bottom.
    for fid in ("front", "back"):
        assert (entries[fid]["width_um"], entries[fid]["height_um"]) == (50000.0, 39000.0)
        assert (entries[fid]["width_mm"], entries[fid]["height_mm"]) == (50.0, 39.0)
    # left / right walls: fit between front/back, same wall height.
    for fid in ("left", "right"):
        assert (entries[fid]["width_um"], entries[fid]["height_um"]) == (49000.0, 39000.0)
        assert (entries[fid]["width_mm"], entries[fid]["height_mm"]) == (49.0, 39.0)


def test_face_cut_dims_rejects_unknown_face():
    with pytest.raises(ValueError, match="Unknown face id"):
        face_cut_dims("inside", W, D, H, T)


# ----- hinge ------------------------------------------------------------------


def test_hinge_layout_default_ring_box():
    layout = hinge_layout(HingeSpec(), W)
    # run = 0.8 * 50000 = 40000; 4 gaps of 400 leave 38400 over 5 segments.
    assert layout["run_length_um"] == pytest.approx(40000.0)
    assert layout["segment_length_um"] == pytest.approx(7680.0)
    assert layout["gap_um"] == 400.0
    # rod sticks out one rod-OD past each end: 40000 + 2*1600.
    assert layout["rod_length_um"] == pytest.approx(43200.0)
    # 5 segments alternate body,lid,body,lid,body.
    assert layout["body_segments"] == 3
    assert layout["lid_segments"] == 2


def test_hinge_layout_scales_with_width_and_segments():
    layout = hinge_layout(HingeSpec(segments=3, coverage=0.5), 30000.0)
    assert layout["run_length_um"] == pytest.approx(15000.0)
    assert layout["segment_length_um"] == pytest.approx((15000.0 - 2 * 400.0) / 3)
    assert layout["body_segments"] == 2
    assert layout["lid_segments"] == 1


# ----- seams ------------------------------------------------------------------


def test_seam_list_eight_seams_with_joint_lengths():
    seams = seam_list(W, D, H, T)
    assert len(seams) == 8
    by_id = {s["id"]: s for s in seams}
    assert by_id["bottom-front"]["length_um"] == W
    assert by_id["bottom-left"]["length_um"] == D
    # corners run the wall height: H - 2t.
    for cid in ("corner-front-left", "corner-front-right", "corner-back-left", "corner-back-right"):
        assert by_id[cid]["length_um"] == pytest.approx(39000.0)
    assert sum(1 for s in seams if s["kind"] == "bottom") == 4
    assert sum(1 for s in seams if s["kind"] == "corner") == 4


# ----- validation -------------------------------------------------------------


def test_validate_default_ring_box_passes():
    validate_assembly(W, D, H, T, FoilSpec(), HingeSpec())


def test_validate_rejects_keepout_swallowing_plate():
    # 10 mm box: left/right plates are 9 mm/side; 2*3425 + 3000 > 9000.
    with pytest.raises(ValueError, match="keep-out swallows"):
        validate_assembly(10000.0, 10000.0, 10000.0, T, FoilSpec(), HingeSpec())


def test_validate_rejects_even_or_too_few_hinge_segments():
    with pytest.raises(ValueError, match="odd number"):
        validate_assembly(W, D, H, T, FoilSpec(), HingeSpec(segments=4))
    with pytest.raises(ValueError, match="odd number"):
        validate_assembly(W, D, H, T, FoilSpec(), HingeSpec(segments=1))


def test_validate_rejects_bad_hinge_coverage():
    with pytest.raises(ValueError, match="coverage"):
        validate_assembly(W, D, H, T, FoilSpec(), HingeSpec(coverage=0.0))
    with pytest.raises(ValueError, match="coverage"):
        validate_assembly(W, D, H, T, FoilSpec(), HingeSpec(coverage=1.5))


def test_validate_rejects_uncuttable_hinge_segments():
    # 99 segments over a 40 mm run -> ~0 mm pieces, shorter than the tube OD.
    with pytest.raises(ValueError, match="segments must be an odd|uncuttable"):
        validate_assembly(W, D, H, T, FoilSpec(), HingeSpec(segments=99))


def test_validate_rejects_rod_wider_than_tube():
    with pytest.raises(ValueError, match="rod"):
        validate_assembly(W, D, H, T, FoilSpec(), HingeSpec(rod_od_um=2400.0, tube_od_um=2400.0))


def test_validate_rejects_height_swallowed_by_glass():
    with pytest.raises(ValueError, match="walls"):
        validate_assembly(W, D, 900.0, T, FoilSpec(), HingeSpec())


# ----- summary + manifest integration ----------------------------------------


def test_assembly_summary_shape():
    asm = assembly_summary(W, D, H, T, FoilSpec(), HingeSpec())
    assert asm["keepout_um"] == pytest.approx(3425.0)
    assert asm["overlap_um"] == pytest.approx(2925.0)
    assert asm["glass_thickness_um"] == T
    assert {e["face"] for e in asm["cut_list"]} == set(FACE_IDS)
    assert len(asm["seams"]) == 8
    assert asm["hinge"]["segments"] == 5
    assert asm["hinge"]["run_length_um"] == pytest.approx(40000.0)
    assert asm["hinge"]["segment_length_um"] == pytest.approx(7680.0)


def test_foil_and_hinge_spec_roundtrip_and_tolerance():
    foil = FoilSpec.from_dict({"tape_width_um": 4763.0})
    assert foil.tape_width_um == 4763.0
    assert foil.safety_um == 500.0  # missing fields -> defaults
    assert FoilSpec.from_dict(foil.to_dict()) == foil
    hinge = HingeSpec.from_dict(None)
    assert hinge == HingeSpec()
    assert HingeSpec.from_dict(hinge.to_dict()) == hinge


def test_box_manifest_carries_assembly_block(isolated_data):
    from app.boxes import BoxSpec, materialize_box
    from app.plates import FrameSpec, PlateSpec

    spec = BoxSpec(width_um=24000.0, depth_um=24000.0, height_um=24000.0)
    spec.faces["top"] = PlateSpec(pattern_slug="wayuu-kanasu-moire", frame=FrameSpec(seed=42))
    manifest = materialize_box(spec, box_id="asm-block")
    asm = manifest["assembly"]
    assert asm["keepout_um"] == pytest.approx(3425.0)
    assert asm["overlap_um"] == pytest.approx(2925.0)
    cut = {e["face"]: e for e in asm["cut_list"]}
    assert cut["front"]["height_mm"] == pytest.approx(23.0)
    assert cut["left"]["width_mm"] == pytest.approx(23.0)
    assert asm["hinge"]["rod_length_um"] == pytest.approx(0.8 * 24000.0 + 3200.0)


def test_frame_scene_kept_out_of_manifests_sidecar_on_disk(isolated_data):
    from app.boxes import materialize_box, BoxSpec
    from app.plates import FrameSpec, PlateSpec, PLATES_ROOT

    spec = BoxSpec(width_um=24000.0, depth_um=24000.0, height_um=24000.0)
    spec.faces["front"] = PlateSpec(pattern_slug="wayuu-kanasu-moire", frame=FrameSpec(seed=7))
    manifest = materialize_box(spec, box_id="lean-faces")
    face = manifest["faces"]["front"]
    assert "frame_scene" not in face["recipe_data"]
    # The plate manifest on disk is lean too; the frame scene lives in the
    # scene.json sidecar (live-preview/debug artifact, excluded from fab zips).
    on_disk = json.loads((PLATES_ROOT / face["id"] / "manifest.json").read_text())
    assert "frame_scene" not in on_disk["recipe_data"]
    scene = json.loads((PLATES_ROOT / face["id"] / "scene.json").read_text())
    assert "segments" in scene


# ---------------------------------------------------------------------------
# Golden contract fixture — shared with frontend/tests/unit/assemblyGolden.test.ts
# ---------------------------------------------------------------------------

def test_golden_fixture_matches_backend() -> None:
    """The shared fixture (tools/fixtures/assembly_golden.json) must agree
    with the live backend math. The frontend runs the SAME cases through
    src/assembly.ts, so a formula change that lands on only one side fails
    one of the two suites instead of silently diverging.

    Regenerate intentionally with tools/dev/gen_assembly_golden.py.
    """
    import json
    from pathlib import Path

    from app.assembly import (
        FoilSpec,
        HingeSpec,
        cut_list,
        hinge_layout,
        keepout_um,
        overlap_um,
        seam_list,
        validate_assembly,
    )

    fixture_path = (
        Path(__file__).resolve().parents[2] / "tools" / "fixtures" / "assembly_golden.json"
    )
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 5

    for case in cases:
        spec = case["spec"]
        exp = case["expected"]
        foil = FoilSpec.from_dict(spec["foil"])
        hinge = HingeSpec.from_dict(spec["hinge"])
        args = (
            spec["width_um"],
            spec["depth_um"],
            spec["height_um"],
            spec["glass_thickness_um"],
        )

        raised = False
        try:
            validate_assembly(*args, foil, hinge)
        except ValueError:
            raised = True
        assert raised == (not exp["valid"]), f"{case['name']}: validity flipped"
        if not exp["valid"]:
            continue

        assert overlap_um(foil, args[3]) == exp["overlap_um"], case["name"]
        assert keepout_um(foil, args[3]) == exp["keepout_um"], case["name"]
        assert cut_list(*args) == exp["cut_list"], case["name"]
        assert {s["id"]: s["length_um"] for s in seam_list(*args)} == exp["seams"], case["name"]
        assert hinge_layout(hinge, spec["width_um"]) == exp["hinge"], case["name"]
