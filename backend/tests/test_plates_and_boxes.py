"""Plate composition and ring-box assembly (BoxSpec v2).

Exercises both the Python API and the HTTP endpoints. The shared
``isolated_data`` fixture (conftest.py) points the per-test data directory
at tmp_path so cache hashes don't leak between runs.

Box geometry note: with the default 1/4" foil + 0.5 mm safety on 0.5 mm
glass, the keep-out is 3.425 mm per edge, so test boxes need min plate
sides comfortably above ~10 mm — we use 20-30 mm boxes throughout.
"""
from __future__ import annotations


import pytest
from fastapi.testclient import TestClient


def test_weld_margin_zeroes_border_in_raster(isolated_data):
    """No gold pixels may appear inside the weld zone of either layer."""
    from PIL import Image
    from app.plates import FrameSpec, PlateSpec, PLATES_ROOT, materialize_plate

    spec = PlateSpec(
        pattern_slug="monogram-jp",
        frame=FrameSpec(seed=11),
        width_um=8000.0,
        height_um=8000.0,
        weld_margin_um=600.0,
    )
    m = materialize_plate(spec)
    pitch = m["pixel_pitch_um"]
    weld_px = int(round(spec.weld_margin_um / pitch))

    for fname in ("front.png", "back.png"):
        img = Image.open(PLATES_ROOT / m["id"] / fname).convert("L")
        w, h = img.size
        # Check each border strip — sum of pixel values must be 0.
        # Top, bottom, left, right rectangles.
        top = img.crop((0, 0, w, weld_px))
        bottom = img.crop((0, h - weld_px, w, h))
        left = img.crop((0, 0, weld_px, h))
        right = img.crop((w - weld_px, 0, w, h))
        for region, name in [(top, "top"), (bottom, "bottom"), (left, "left"), (right, "right")]:
            stats = region.getextrema()
            assert stats == (0, 0), f"{fname} {name} weld zone has gold: extrema={stats}"


def test_materialize_plate_writes_manifest(isolated_data):
    from app.plates import FrameSpec, PlateSpec, materialize_plate, PLATES_ROOT

    spec = PlateSpec(
        pattern_slug="monogram-jp",
        frame=FrameSpec(seed=3),
        width_um=8000.0,
        height_um=6000.0,
        weld_margin_um=400.0,
    )
    manifest = materialize_plate(spec)
    assert manifest["kind"] == "plate"
    assert manifest["extent_um"] == [8000.0, 6000.0]
    # Files exist on disk.
    pid = manifest["id"]
    assert (PLATES_ROOT / pid / "manifest.json").exists()
    assert (PLATES_ROOT / pid / "front.png").exists()
    assert (PLATES_ROOT / pid / "back.png").exists()
    # SVG is lazy — empty in the manifest until the export path requests it.
    assert manifest["files"]["front_svg"] == ""
    from app.plates import ensure_plate_svg

    pair = ensure_plate_svg(pid)
    assert pair is not None
    front_svg, back_svg = pair
    assert front_svg.exists() and back_svg.exists()
    # frame_scene lives in a scene.json sidecar, NOT in the manifest — the
    # embedded blob made manifests MBs and nothing at runtime consumed it.
    assert "frame_scene" not in manifest["recipe_data"]
    import json

    scene = json.loads((PLATES_ROOT / pid / "scene.json").read_text())
    assert "segments" in scene and "max_t" in scene


def test_materialize_plate_is_cached(isolated_data):
    from app.plates import FrameSpec, PlateSpec, materialize_plate

    spec = PlateSpec(
        pattern_slug="monogram-jp",
        frame=FrameSpec(seed=4),
        width_um=6000.0,
        height_um=6000.0,
        weld_margin_um=300.0,
    )
    m1 = materialize_plate(spec)
    m2 = materialize_plate(spec)
    assert m1["id"] == m2["id"]


# ----- Box model (v2) ---------------------------------------------------------


def _box_spec(width=30000.0, depth=20000.0, height=25000.0, faces=None):
    """A v2 box — small enough to materialize fast, large enough to clear
    the default foil keep-out and hinge validation. ``faces`` limits which
    plates are populated (each render costs ~a minute; the cut list and
    assembly block always cover all six regardless)."""
    from app.boxes import BoxSpec, FACE_IDS
    from app.plates import FrameSpec, PlateSpec

    spec = BoxSpec(width_um=width, depth_um=depth, height_um=height)
    for i, fid in enumerate(FACE_IDS):
        if faces is not None and fid not in faces:
            continue
        spec.faces[fid] = PlateSpec(
            pattern_slug="monogram-jp",
            frame=FrameSpec(seed=100 + i),
            # dims/glass/weld are re-stamped by normalize_face_dims.
            width_um=0,
            height_um=0,
        )
    return spec


def test_normalize_stamps_cut_dims_keepout_and_glass(isolated_data):
    from app.boxes import FACE_IDS

    spec = _box_spec()
    spec.glass.thickness_um = 500.0
    spec.normalize_face_dims()
    t = 500.0
    # Cut list per the contract: walls sit on the bottom, lid on the wall rim.
    assert (spec.faces["bottom"].width_um, spec.faces["bottom"].height_um) == (30000.0, 20000.0)
    assert (spec.faces["top"].width_um, spec.faces["top"].height_um) == (30000.0, 20000.0)
    assert (spec.faces["front"].width_um, spec.faces["front"].height_um) == (30000.0, 25000.0 - 2 * t)
    assert (spec.faces["left"].width_um, spec.faces["left"].height_um) == (20000.0 - 2 * t, 25000.0 - 2 * t)
    for fid in FACE_IDS:
        # 1/4" tape on 0.5 mm glass: overlap 2925, +500 safety = 3425.
        assert spec.faces[fid].weld_margin_um == pytest.approx(3425.0)
        assert spec.faces[fid].glass.thickness_um == 500.0
        assert spec.faces[fid].glass.material == "fused silica"


def test_materialize_box_fans_out(isolated_data):
    from app.boxes import materialize_box

    # One face of each cut-dim class (front/back, top/bottom, left/right)
    # keeps the fan-out honest without rendering all six plates.
    spec = _box_spec(width=30000.0, depth=20000.0, height=25000.0,
                     faces=("front", "top", "left"))
    box = materialize_box(spec, box_id="testbox")
    assert box["kind"] == "box"
    assert box["id"] == "testbox"
    assert set(box["faces"].keys()) == {"front", "top", "left"}
    # Cut list extents: top/bottom W×D, front/back W×(H-2t), left/right (D-2t)×(H-2t).
    assert box["faces"]["front"]["extent_um"] == [30000.0, 24000.0]
    assert box["faces"]["top"]["extent_um"] == [30000.0, 20000.0]
    assert box["faces"]["left"]["extent_um"] == [19000.0, 24000.0]
    # The manifest carries the derived assembly block.
    asm = box["assembly"]
    assert asm["keepout_um"] == pytest.approx(3425.0)
    assert asm["overlap_um"] == pytest.approx(2925.0)
    assert asm["glass_thickness_um"] == 500.0
    assert len(asm["cut_list"]) == 6
    assert len(asm["seams"]) == 8
    assert asm["hinge"]["run_length_um"] == pytest.approx(0.8 * 30000.0)


def test_box_manifest_faces_are_lean_and_scene_lives_in_sidecar(isolated_data):
    import json
    from app.boxes import materialize_box
    from app.plates import PLATES_ROOT

    spec = _box_spec(faces=("front",))
    box = materialize_box(spec, box_id="lean-1")
    for fid, face in box["faces"].items():
        assert "frame_scene" not in face.get("recipe_data", {}), (
            f"box manifest face {fid} should not embed the frame_scene blob"
        )
        # The per-plate manifest on disk is lean too; the full frame scene
        # is kept in a scene.json sidecar for debugging/inspection.
        on_disk = json.loads((PLATES_ROOT / face["id"] / "manifest.json").read_text())
        assert "frame_scene" not in on_disk["recipe_data"]
        assert (PLATES_ROOT / face["id"] / "scene.json").exists()


def test_box_persists_and_loads(isolated_data):
    from app.boxes import get_box, materialize_box
    from app.plates import FrameSpec, PlateSpec

    spec = _box_spec(width=20000.0, depth=20000.0, height=20000.0)
    spec.faces = {
        "front": PlateSpec(
            pattern_slug="monogram-jp",
            frame=FrameSpec(seed=1),
            width_um=0,
            height_um=0,
        )
    }
    m = materialize_box(spec, box_id="persist-1")
    loaded = get_box("persist-1")
    assert loaded is not None
    assert loaded["id"] == "persist-1"
    assert "front" in loaded["faces"]
    assert "assembly" in loaded


def test_anonymous_generate_reuses_scratch_and_is_not_listed(isolated_data):
    from app.boxes import BOXES_ROOT, SCRATCH_BOX_ID, list_boxes, materialize_box

    m1 = materialize_box(_box_spec(faces=("front",)))
    m2 = materialize_box(_box_spec(faces=("front",)))
    # Anonymous (live-preview) generates overwrite ONE scratch slot...
    assert m1["id"] == m2["id"] == SCRATCH_BOX_ID
    assert m1["saved"] is False
    assert sum(1 for d in BOXES_ROOT.iterdir() if (d / "box.json").exists()) == 1
    # ...and never show up as presets.
    assert list_boxes() == []
    saved = materialize_box(_box_spec(faces=("front",)), box_id="kept")
    assert saved["saved"] is True
    assert [b["id"] for b in list_boxes()] == ["kept"]


def test_box_id_path_traversal_is_rejected(isolated_data):
    from app.boxes import delete_box, get_box, materialize_box

    for bad in ("../evil", "..", "a/b", "a\\b", "-leading", ".hidden", "x" * 65):
        with pytest.raises(ValueError, match="box_id"):
            materialize_box(_box_spec(faces=()), box_id=bad)
        assert get_box(bad) is None
        assert delete_box(bad) is False


def test_list_boxes_skips_old_format_and_corrupt(isolated_data):
    import json
    from app.boxes import BOXES_ROOT, list_boxes, materialize_box

    materialize_box(_box_spec(faces=("front",)), box_id="good-1")
    # Old-format (pre-v2) manifest: parses as JSON but has no assembly block.
    old = BOXES_ROOT / "old-1"
    old.mkdir(parents=True)
    (old / "box.json").write_text(json.dumps({"kind": "box", "id": "old-1", "spec": {}}))
    # Corrupt file.
    bad = BOXES_ROOT / "bad-1"
    bad.mkdir(parents=True)
    (bad / "box.json").write_text("{not json")

    boxes = list_boxes()
    assert [b["id"] for b in boxes] == ["good-1"]


def test_box_spec_from_dict_tolerates_missing_fields():
    from app.boxes import BoxSpec

    spec = BoxSpec.from_dict({"width_um": 42000.0})
    assert spec.width_um == 42000.0
    assert spec.depth_um == 50000.0
    assert spec.height_um == 40000.0
    assert spec.glass.thickness_um == 500.0
    assert spec.foil.tape_width_um == 6350.0
    assert spec.foil.finish == "bright"
    assert spec.hinge.segments == 5
    assert spec.hinge.coverage == 0.8
    assert spec.faces == {}


def test_default_box_spec_matches_contract():
    from app.boxes import FACE_IDS, default_box_spec

    spec = default_box_spec()
    # THE PRODUCTION BOX: six SINGLE 2.25 mm fused-quartz plies, butt-jointed
    # with 1/4" foil (2026-09-16 — no inner plies, no bonding). These numbers
    # are mirrored term for term by the frontend's defaultBoxSpec(); the two
    # must move together or the live preview stops describing the real part.
    assert (spec.width_um, spec.depth_um, spec.height_um) == (32000.0, 32000.0, 35000.0)
    assert spec.bonded is False
    assert spec.foil.tape_width_um == 6350.0
    assert spec.glass.thickness_um == 2250.0
    assert spec.glass.material == "fused quartz"
    assert spec.glass.n == pytest.approx(1.4585)
    assert set(spec.faces) == set(FACE_IDS)
    # Four written faces and two of bare glass — see boxes.default_box_spec.
    expected_slug = {
        "front": "globe-atlantic",
        "back": "photo-halftone",
        "top": "monogram-jp",
        "bottom": "solid-gold",
        "left": "photo-halftone",
        "right": "photo-halftone",
    }
    for i, fid in enumerate(FACE_IDS):
        assert spec.faces[fid].pattern_slug == expected_slug[fid]
        # Per-face seed 100..105 in FACE_IDS order (distinct moiré carrier angle).
        assert spec.faces[fid].frame.seed == 100 + i
        # Finer, lacier foliage at the same band width, on every face.
        assert spec.faces[fid].frame.motif_scale == 0.68
        assert spec.faces[fid].frame.band_um == 2400.0
    # The two photographs carry their own AUTHORED colour plans, and every face
    # is a single written ply (2026-09-15: the two-ply moire effects are gone).
    assert spec.faces["left"].pattern_params == {"image": "beach", "colour_mode": "authored"}
    assert spec.faces["right"].pattern_params == {"image": "sunset", "colour_mode": "authored"}
    assert [fid for fid in FACE_IDS if spec.faces[fid].single_ply] == list(FACE_IDS)


def test_the_production_art_rim_is_pinned_not_derived_from_the_foil():
    """The 2026-09-15 plate was WRITTEN with a 3.6375 mm rim on every face,
    chosen when the box was bonded (one ply + the interior foil fold). The box
    is six single plies now and its 1/4" foil implies a 2.55 mm rim instead —
    but the mask does not move for a construction change, so the rim is PINNED.

    Both layers carry it: on a single ply the front keep-out is the smaller of
    the two rims (plates._raster_compose_plate), so a narrower back window would
    pull the garland in just as surely as a narrower weld margin."""
    from app.assembly import keepout_um
    from app.boxes import FACE_IDS, default_box_spec
    from app.production import ART_RIM_UM

    spec = default_box_spec()
    assert ART_RIM_UM == 3637.5
    assert spec.art_rim_um == ART_RIM_UM
    # ...and it is NOT what this box's foil would give.
    assert keepout_um(spec.foil, spec.glass.thickness_um) == pytest.approx(2550.0)
    for fid in FACE_IDS:
        face = spec.faces[fid]
        assert face.weld_margin_um == ART_RIM_UM, fid
        assert face.back_margin_um == ART_RIM_UM, fid
    # Re-normalizing (materialize_box does it every time) must not undo it.
    spec.normalize_face_dims()
    assert spec.faces["top"].weld_margin_um == ART_RIM_UM


def test_a_box_that_pins_no_rim_still_derives_one_from_its_foil():
    """The pin is opt-in: any box the user builds keeps the derived rim."""
    from app.assembly import back_window_um, keepout_um
    from app.boxes import BoxSpec
    from app.plates import PlateSpec

    spec = BoxSpec(width_um=24000.0, depth_um=24000.0, height_um=24000.0)
    spec.faces["top"] = PlateSpec(pattern_slug="monogram-jp")
    assert spec.art_rim_um is None
    spec.normalize_face_dims()
    t = spec.glass.thickness_um
    assert spec.faces["top"].weld_margin_um == keepout_um(spec.foil, t)
    assert spec.faces["top"].back_margin_um == back_window_um(spec.foil, t)


def test_the_ring_fits_a_single_ply_wall():
    """One ply per face means a 2.25 mm wall, not a 4.5 mm bonded stack: the
    same 32 x 32 x 35 mm outer box opens from a 23 mm interior to
    27.5 x 27.5 x 30.5 mm, and the ring gains 4.5 mm on every axis."""
    from app.boxes import ring_fit, ring_interior_um
    from app.production import PLY_UM

    fit = ring_fit(32000.0, 32000.0, 35000.0, PLY_UM)
    assert fit["interior_um"] == [27500.0, 27500.0, 30500.0]
    assert fit["needed_um"] == list(ring_interior_um())
    assert fit["clearance_um"] == [4500.0, 4500.0, 4500.0]
    assert fit["fits"] is True


def test_the_single_ply_cut_dims_are_the_dies_that_were_written():
    """The inner plies leaving must not move a die. A face's OUTER ply was
    always cut at the PLY thickness, so the single-ply ``face_cut_dims`` and the
    bonded panelizer's ``face:F`` rect are the same rectangle — 32 x 32 for the
    lid and base, 32 x 30.5 front and back, 27.5 x 30.5 for the sides, which is
    what the 2026-09-15 plate was diced to."""
    from app import ply_cuts as pc
    from app.assembly import FACE_IDS, face_cut_dims
    from app.boxes import default_box_spec
    from app.production import PLY_UM

    spec = default_box_spec()
    dims = (spec.width_um, spec.depth_um, spec.height_um)
    pair = {r.face: (r.width_um, r.height_um)
            for r in pc.pair_rects(*dims, PLY_UM)}
    want = {
        "top": (32000.0, 32000.0), "bottom": (32000.0, 32000.0),
        "front": (32000.0, 30500.0), "back": (32000.0, 30500.0),
        "left": (27500.0, 30500.0), "right": (27500.0, 30500.0),
    }
    for fid in FACE_IDS:
        single = face_cut_dims(fid, *dims, PLY_UM)
        assert single == pair[pc.subplate_id(fid, "F")], fid
        assert single == want[fid], fid
        assert (spec.faces[fid].width_um, spec.faces[fid].height_um) == want[fid], fid


# ----- production face types --------------------------------------------------


def test_photo_halftone_registers_with_a_valid_recipe(isolated_data):
    from app.patterns.base import RECIPE_NAMES, registry

    cls = registry["photo-halftone"]
    assert cls.render_recipe in RECIPE_NAMES
    # The choice list is built from the prepared assets and must exclude the
    # ``.subject`` / ``.fade`` sidecars — offering one as a "photograph" would
    # screen a matte.
    images = dict((p.name, p) for p in cls.params)["image"].choices
    assert "beach" in images
    assert not any("." in name for name in images)

    gp = cls.generate(image="beach", extent_um=1000.0)
    assert not gp.front.is_empty, "the line screen should carry bands"
    # Front layer only: gold on the inner ply would show through the gaps
    # between bands and lift every shadow in the picture.
    assert gp.back.is_empty
    assert gp.min_feature_um >= 2.0


def test_photo_coverage_fades_to_the_carrier_field_but_keeps_the_centre(isolated_data):
    import numpy as np
    from app.patterns.bitmap.photo import CARRIER_COV, box_edge, photo_coverage

    gate = 0.94
    cov, ids, _periods = photo_coverage("beach", 0.50, gate, 22, 160, colour_mode="faces")
    d = box_edge(cov.shape[0])

    # Past the gate the picture is GONE — the field the art box dissolves into
    # (bare glass since 2026-09-10: CARRIER_COV = 0, a single ply carries no
    # carrier) instead of ending at a border. (Corners are the only place d
    # exceeds the gate on the rounded square metric, which is the point of it.)
    outside = d > gate
    assert outside.any(), "the rounded-square metric must exceed the gate somewhere"
    assert np.allclose(cov[outside], CARRIER_COV, atol=1e-6)

    # The middle is still a photograph: the fade must not have flattened it.
    n = cov.shape[0]
    q = n // 4
    centre = cov[q : n - q, q : n - q]
    assert centre.std() > 0.05, f"centre went flat (std {centre.std():.4f})"
    assert centre.max() > CARRIER_COV + 0.3, "the picture's darks must stand well off the field"
    assert centre.min() < centre.max() - 0.3, "the picture keeps its tonal range"

    # Colour dies with the picture — no coloured band may survive out into the
    # field, where a sub-grating would advertise the dissolve.
    assert not ids[outside].any()


def test_blank_face_composes_to_empty_layers(isolated_data):
    import numpy as np
    from PIL import Image
    from app.plates import PLATES_ROOT, FrameSpec, PlateSpec, ensure_plate_svg, materialize_plate

    spec = PlateSpec(
        pattern_slug="blank",
        frame=FrameSpec(seed=21),
        width_um=8000.0,
        height_um=8000.0,
        weld_margin_um=500.0,
    )
    m = materialize_plate(spec)
    for fname in ("front.png", "back.png"):
        img = Image.open(PLATES_ROOT / m["id"] / fname).convert("L")
        assert img.getextrema() == (0, 0), f"{fname} carries gold on a blank face"
    assert m["recipe_data"]["blank"] is True
    # The fab pair still EXISTS — a blank face reads as "blank", not "missing".
    pair = ensure_plate_svg(m["id"])
    assert pair is not None
    for svg in pair:
        body = svg.read_text(encoding="utf-8")
        assert "<path" not in body, "a blank face must bake no geometry"


def test_single_ply_face_has_no_carrier_on_either_layer(isolated_data):
    """One ply: the leaves (and the centerpiece) on the front, NO carrier
    anywhere — the ring between the art box and the frame band is bare glass —
    and nothing on the back. (2026-09-10: photo + frame only on the sides.)"""
    import numpy as np
    from PIL import Image
    from app.plates import (
        CENTERPIECE_FILL,
        FRAME_LEVEL,
        PLATES_ROOT,
        FrameSpec,
        PlateSpec,
        _aperture,
        materialize_plate,
    )

    def _spec(single_ply: bool) -> PlateSpec:
        return PlateSpec(
            pattern_slug="monogram-jp",
            frame=FrameSpec(seed=22),
            width_um=12000.0,
            height_um=12000.0,
            weld_margin_um=500.0,
            back_margin_um=1200.0,
            single_ply=single_ply,
        )

    one = materialize_plate(_spec(True))
    two = materialize_plate(_spec(False))
    assert one["id"] != two["id"], "single_ply must be part of the plate hash"
    assert one["recipe_data"]["single_ply"] is True
    assert two["recipe_data"]["single_ply"] is False

    # A sample point in the ring between the centerpiece art box and the frame
    # band — clean carrier territory on a two-ply face, so it isolates the
    # carrier from the foliage.
    spec = _spec(True)
    pitch = one["pixel_pitch_um"]
    aperture = _aperture(spec)
    r_um = 0.5 * (CENTERPIECE_FILL * aperture / 2.0 + aperture / 2.0)
    w, h = Image.open(PLATES_ROOT / one["id"] / "front.png").size
    px = (w // 2 + int(round(r_um / pitch)), h // 2)

    one_front = Image.open(PLATES_ROOT / one["id"] / "front.png").convert("L")
    one_back = Image.open(PLATES_ROOT / one["id"] / "back.png").convert("L")
    two_front = Image.open(PLATES_ROOT / two["id"] / "front.png").convert("L")
    two_back = Image.open(PLATES_ROOT / two["id"] / "back.png").convert("L")

    assert one_back.getextrema() == (0, 0), "a single ply has no back layer to write"
    assert one_front.getpixel(px) == 0, "a single ply carries no carrier: bare glass between art and frame"
    # The two-ply face is the control: same point, carrier on the BACK only.
    assert two_back.getpixel(px) == FRAME_LEVEL
    assert two_front.getpixel(px) == 0


def test_single_ply_leaves_are_fine_gratings_one_period_per_family(isolated_data):
    """One ply diffracts instead of beating: the garland is written as fine 50%
    gratings with one PERIOD per motif family (the colour ladder), and the
    recipe advertises that period so the preview's sheen knows it. A two-ply
    face keeps its moire louvres."""
    from app.export_fine import build_plate_fine
    from app.plates import (
        FrameSpec,
        PlateSpec,
        _carrier_recipe_data,
        single_ply_leaf_period_um,
    )

    def _spec(single_ply: bool) -> PlateSpec:
        return PlateSpec(
            pattern_slug="monogram-jp",
            frame=FrameSpec(seed=31),
            width_um=12000.0,
            height_um=12000.0,
            weld_margin_um=500.0,
            back_margin_um=1200.0,
            single_ply=single_ply,
        )

    from app.plates import SINGLE_PLY_LEAF_FILL, SINGLE_PLY_LEAF_HUE_PERIODS_UM

    one = build_plate_fine(_spec(True), "one")
    leaves = one.stats.get("single_ply_leaves")
    assert leaves, "a single-ply face must report its diffractive leaf gratings"
    assert leaves["fill"] == SINGLE_PLY_LEAF_FILL == "hue"
    assert leaves["n_families"] >= 2, "more than one family, or nothing flashes separately"
    # one PERIOD per family, every one of them a legal rung of the colour ladder
    assert tuple(leaves["hue_periods_um"]) == tuple(SINGLE_PLY_LEAF_HUE_PERIODS_UM)
    assert min(SINGLE_PLY_LEAF_HUE_PERIODS_UM) >= 4.0, "2 um lines and gaps: the litho floor"
    assert leaves["min_legal_period_um"] <= min(SINGLE_PLY_LEAF_HUE_PERIODS_UM)
    assert not one.back_polys
    # ONE helper answers for both the manifest key and the preview period map,
    # and under the "hue" fill that answer is the LADDER'S MEAN — not the
    # single-period "lines" constant, which this face never writes.
    ladder_mean = sum(SINGLE_PLY_LEAF_HUE_PERIODS_UM) / len(SINGLE_PLY_LEAF_HUE_PERIODS_UM)
    assert single_ply_leaf_period_um(_spec(True)) == pytest.approx(ladder_mean)
    assert _carrier_recipe_data(_spec(True))["single_ply_leaf_period_um"] == pytest.approx(ladder_mean)
    assert min(SINGLE_PLY_LEAF_HUE_PERIODS_UM) <= ladder_mean <= max(SINGLE_PLY_LEAF_HUE_PERIODS_UM)
    two = build_plate_fine(_spec(False), "two")
    assert "single_ply_leaves" not in two.stats
    assert single_ply_leaf_period_um(_spec(False)) == 0.0
    assert _carrier_recipe_data(_spec(False))["single_ply_leaf_period_um"] == 0.0


def test_photo_face_composes_bands_on_the_front_only(isolated_data):
    import numpy as np
    from PIL import Image
    from app.plates import (
        ART_LEVEL,
        PLATES_ROOT,
        RAINBOW_LEVEL,
        FrameSpec,
        PlateSpec,
        materialize_plate,
        photo_band_rects,
    )

    spec = PlateSpec(
        pattern_slug="photo-halftone",
        pattern_params={"image": "beach", "colour_mode": "faces"},
        frame=FrameSpec(seed=23),
        width_um=12000.0,
        height_um=12000.0,
        weld_margin_um=500.0,
        back_margin_um=1200.0,
        single_ply=True,
    )
    m = materialize_plate(spec)
    front = np.asarray(Image.open(PLATES_ROOT / m["id"] / "front.png").convert("L"))
    back = Image.open(PLATES_ROOT / m["id"] / "back.png").convert("L")

    assert (front == ART_LEVEL).any(), "no halftone bands in the composed front mask"
    # Coloured bands get RAINBOW_LEVEL, where the shader adds the spectral sheen
    # that stands in for the sub-grating the fab writes for real.
    assert (front == RAINBOW_LEVEL).any(), "colour_mode='faces' coloured nothing"
    assert back.getextrema() == (0, 0)
    # The shader must fill ART with solid gold, not the procedural switch
    # carrier — the band height already IS the tone.
    assert m["recipe_data"]["art_solid"] is True

    # The exact fab geometry the SVG and the fine GDS both bake, in plate µm.
    rects = photo_band_rects(spec)
    assert rects.shape[0] > 100
    assert (rects[:, 1] > rects[:, 0]).all() and (rects[:, 3] > rects[:, 2]).all()

    # PERIOD MAP: the colour sub-grating is ~5 µm lines, far under a pixel at
    # 2048 px, so literal_front can only carry its coverage — the colour it
    # diffracts comes from period_front (period µm × LITERAL_PERIOD_SCALE).
    from app.plates import LITERAL_PERIOD_SCALE, photo_colour_band_periods

    assert m["files"]["period_front"] == f"/data/plates/{m['id']}/period_front.png"
    period_img = Image.open(PLATES_ROOT / m["id"] / "period_front.png")
    assert period_img.mode == "L"
    period = np.asarray(period_img)
    assert period.max() > 0, "coloured bands published an empty period map"
    _bands, periods_um = photo_colour_band_periods(spec)
    assert periods_um.size > 0
    # Every painted value INSIDE THE ART BOX decodes to a period the fab
    # actually writes. (Outside it, a single-ply face's garland leaves carry
    # their own diffractive period — see _single_ply_leaf_period_raster.)
    from app.plates import CENTERPIECE_FILL, _aperture

    h, w = period.shape
    side = CENTERPIECE_FILL * _aperture(spec)
    sx, sy = w / spec.width_um, h / spec.height_um
    x0, x1 = int((spec.width_um / 2 - side / 2) * sx) + 2, int((spec.width_um / 2 + side / 2) * sx) - 2
    y0, y1 = int((spec.height_um / 2 - side / 2) * sy) + 2, int((spec.height_um / 2 + side / 2) * sy) - 2
    inside = period[y0:y1, x0:x1]
    painted = np.unique(inside[inside > 0]) / LITERAL_PERIOD_SCALE
    assert painted.size > 0
    assert painted.min() >= periods_um.min() - 0.05
    assert painted.max() <= periods_um.max() + 0.05
    # The period field lives inside the picture, not over the whole plate.
    assert 0.0 < np.count_nonzero(period) / period.size < 0.5


def test_literal_rasters_publish_the_fabricated_chrome(isolated_data):
    """Every composed face publishes coverage rasters of its REAL litho geometry.

    The renderer samples these on the two pattern planes instead of
    synthesising gratings, so the contract is exact: mode L, 2048 on the long
    side, plate aspect, and a ``files`` URL in the same form as ``front_png``.
    An empty layer is an all-zero raster, never an absent file.
    """
    import numpy as np
    from PIL import Image
    from app.plates import (
        LITERAL_RASTER_PX,
        PLATES_ROOT,
        FrameSpec,
        PlateSpec,
        materialize_plate,
    )

    art = materialize_plate(
        PlateSpec(
            pattern_slug="monogram-jp",
            frame=FrameSpec(seed=31),
            width_um=8000.0,
            height_um=6000.0,
            weld_margin_um=400.0,
        )
    )
    blank = materialize_plate(
        PlateSpec(
            pattern_slug="blank",
            frame=FrameSpec(seed=32),
            width_um=8000.0,
            height_um=6000.0,
            weld_margin_um=400.0,
        )
    )

    for m in (art, blank):
        pid = m["id"]
        assert m["recipe_data"]["literal"] is True
        # The frontend refuses any other recipe; literal rasters do not change it.
        assert m["render_recipe"] == "foliage_moire"
        for key in ("literal_front", "literal_back"):
            assert m["files"][key] == f"/data/plates/{pid}/{key}.png", key
            path = PLATES_ROOT / pid / f"{key}.png"
            assert path.exists(), f"{key} missing on disk"
            img = Image.open(path)
            assert img.mode == "L", f"{key} is {img.mode}, not a coverage mask"
            assert max(img.size) == LITERAL_RASTER_PX
            # Plate aspect, and the same extent/orientation as front.png/back.png
            # so the two rasters overlay under one uniform scale.
            mask_w, mask_h = Image.open(PLATES_ROOT / pid / "front.png").size
            assert img.size[0] * mask_h == pytest.approx(img.size[1] * mask_w, rel=2e-3)

    def _arr(m, key):
        return np.asarray(Image.open(PLATES_ROOT / m["id"] / f"{key}.png"))

    # BARE GLASS on both plies: an all-zero raster is a valid literal layer.
    assert _arr(blank, "literal_front").max() == 0
    assert _arr(blank, "literal_back").max() == 0
    # A written face carries chrome on both plies (foliage front, carrier back).
    assert _arr(art, "literal_front").max() > 0
    assert _arr(art, "literal_back").max() > 0
    # Sub-pixel grating lines must AVERAGE in rather than drop out — a raster
    # that only ever hit 0 or 255 would mean the coverage estimate collapsed to
    # a binary in/out fill.
    front = _arr(art, "literal_front")
    assert ((front > 0) & (front < 255)).any()

    # period_front is published only where a sub-grating period field exists;
    # a plain face omits the file AND the key rather than shipping zeros.
    for m in (art, blank):
        assert "period_front" not in m["files"]
        assert not (PLATES_ROOT / m["id"] / "period_front.png").exists()


def _grating_rings(period_um, duty, angle_deg, w_um, h_um):
    """CCW µm rings for a grating covering the whole plate, plate frame, y up."""
    import numpy as np

    th = np.radians(angle_deg)
    ux, uy = np.cos(th), np.sin(th)          # along the lines
    nx, ny = -np.sin(th), np.cos(th)         # across them
    half = period_um * duty / 2.0
    reach = 0.5 * (w_um * abs(nx) + h_um * abs(ny)) + period_um
    span = w_um + h_um
    rings = []
    for k in range(-int(reach / period_um) - 1, int(reach / period_um) + 2):
        c = k * period_um
        cx, cy = nx * c, ny * c
        rings.append(
            np.array(
                [
                    (cx + ux * su * span + nx * sv * half,
                     cy + uy * su * span + ny * sv * half)
                    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                ]
            )
        )
    return rings


def test_literal_raster_coverage_is_unbiased(isolated_data):
    """A 50%-duty grating must raster to 50% coverage — at ANY angle, and at a
    period far below a texel.

    ``literal_front``/``literal_back`` are what the renderer samples in place of
    a procedural grating, so a systematic error in them is a systematic error in
    the preview's brightness. The features are all sub-texel (a 27.5 mm face is
    ~13.4 µm/texel; the colour sub-gratings are 2.5 µm lines on a 5 µm period),
    so a boundary-inclusive polygon fill fattens every line by a whole sample:
    the pre-v16 2× supersampled draw read 0.538 on the 99 µm carrier and a
    solid 1.000 on the 5 µm colour stripes. ``literal_raster`` accumulates the
    exact area instead — the axis-aligned rectangles analytically, everything
    else by signed-area accumulation — so both paths land on the true duty.
    """
    import numpy as np
    from app.plates import PlateSpec, _literal_layer_raster, _literal_raster_dims

    spec = PlateSpec(pattern_slug="blank", width_um=29100.0, height_um=29100.0)
    w_px, h_px = _literal_raster_dims(spec)

    def coverage(period_um, duty, angle_deg):
        rings = _grating_rings(period_um, duty, angle_deg, spec.width_um, spec.height_um)
        img = _literal_layer_raster(rings, spec, w_px, h_px)
        assert img.mode == "L" and img.size == (w_px, h_px)
        # Trim the plate edge, where the grating is cut off mid-period.
        return np.asarray(img, dtype=np.float64)[64:-64, 64:-64] / 255.0

    # ANGLED (the carrier / leaf gratings): goes down the polygon path.
    angled = coverage(99.0, 0.5, 148.0)
    assert angled.mean() == pytest.approx(0.5, abs=0.01)
    assert 0.0 < (angled >= 254 / 255).mean() < 0.6, "an angled 50% duty is not solid"

    # AXIS-ALIGNED (the colour stripes, halftone bands, comb, switch lanes):
    # goes down the exact-rectangle path.
    assert coverage(44.0, 0.5, 0.0).mean() == pytest.approx(0.5, abs=0.01)

    # SUB-TEXEL and axis-aligned: 2.5 µm lines under a 14 µm texel. The whole
    # point of the coverage raster — every texel reads the local duty, and NONE
    # of them saturates into solid gold.
    fine = coverage(5.0, 0.5, 0.0)
    assert fine.mean() == pytest.approx(0.5, abs=0.01)
    assert fine.max() < 254 / 255, "sub-texel 50% stripes rastered as solid chrome"
    assert fine.min() > 1 / 255, "sub-texel 50% stripes dropped out entirely"

    # The raster carries DUTY, not just presence: a quarter-duty grating is a
    # quarter as bright, at both a coarse and a sub-texel period.
    assert coverage(99.0, 0.25, 148.0).mean() == pytest.approx(0.25, abs=0.01)
    assert coverage(5.0, 0.25, 0.0).mean() == pytest.approx(0.25, abs=0.01)


# ----- HTTP layer ------------------------------------------------------------


@pytest.fixture
def app_client(isolated_data):
    from app.main import create_app

    return TestClient(create_app())


def test_http_plates_roundtrip(app_client):
    body = {
        "spec": {
            "pattern_slug": "monogram-jp",
            "pattern_params": {},
            "frame": {"seed": 12},
            "glass": {},
            "width_um": 1800.0,
            "height_um": 1800.0,
            "label": "test plate",
        },
        "force": False,
    }
    r = app_client.post("/plates/generate", json=body)
    assert r.status_code == 200, r.text
    plate = r.json()
    pid = plate["id"]
    r2 = app_client.get(f"/plates/{pid}")
    assert r2.status_code == 200
    assert r2.json()["id"] == pid

    r3 = app_client.get("/plates")
    assert r3.status_code == 200
    assert any(p["id"] == pid for p in r3.json())


def _http_box_body(**overrides):
    face = {
        "pattern_slug": "monogram-jp",
        "pattern_params": {},
        "frame": {"seed": 1},
        "glass": {},
        "label": "",
    }
    body = {
        "width_um": 24000.0,
        "depth_um": 24000.0,
        "height_um": 24000.0,
        "glass": {"thickness_um": 500.0, "material": "fused silica", "n": 1.46},
        "foil": {"tape_width_um": 6350.0, "safety_um": 500.0, "bead_um": 2000.0, "finish": "bright"},
        "hinge": {"style": "tube", "tube_od_um": 2400.0, "rod_od_um": 1600.0, "segments": 5, "coverage": 0.8},
        "faces": {
            fid: {**face, "frame": {"seed": 100 + i}}
            for i, fid in enumerate(["front", "back", "top", "bottom", "left", "right"])
        },
        "label": "test box",
        "box_id": "http-box-1",
        "force": False,
    }
    body.update(overrides)
    return body


def test_http_boxes_roundtrip(app_client):
    body = _http_box_body()
    # Two faces keep the test fast; the assembly block still covers all six.
    body["faces"] = {fid: body["faces"][fid] for fid in ("front", "left")}
    r = app_client.post("/boxes/generate", json=body)
    assert r.status_code == 200, r.text
    box = r.json()
    assert box["id"] == "http-box-1"
    assert "assembly" in box
    assert len(box["assembly"]["cut_list"]) == 6

    r2 = app_client.get("/boxes/http-box-1")
    assert r2.status_code == 200

    r3 = app_client.get("/boxes")
    assert r3.status_code == 200
    assert any(b["id"] == "http-box-1" for b in r3.json())


def test_http_box_rejects_unsafe_box_id(app_client):
    body = _http_box_body(box_id="../evil")
    body["faces"] = {}
    r = app_client.post("/boxes/generate", json=body)
    assert r.status_code == 400
    assert "box_id" in r.json()["detail"]


def test_http_box_validation_error_is_actionable(app_client):
    # 10 mm box: left/right plates are 9 mm — swallowed by the 3.425 mm rim.
    body = _http_box_body(width_um=10000.0, depth_um=10000.0, height_um=10000.0, box_id="tiny")
    r = app_client.post("/boxes/generate", json=body)
    assert r.status_code == 400
    assert "keep-out" in r.json()["detail"]


def test_plate_svg_central_scaled_to_aperture(isolated_data):
    """Regression (fab-critical): the SVG masks must carry the SAME
    aperture-scaled central pattern as the preview PNGs. The SVG path used to
    emit the central at its native extent (~2 mm) inside the full plate — a
    wrong lithography mask that no test caught."""
    import re

    from app.plates import FrameSpec, PlateSpec, ensure_plate_svg, materialize_plate

    spec = PlateSpec(
        pattern_slug="monogram-jp",
        frame=FrameSpec(seed=5),
        width_um=8000.0,
        height_um=8000.0,
        weld_margin_um=600.0,
    )
    m = materialize_plate(spec)
    aperture_um = float(m["extra"]["aperture_um"])
    assert aperture_um > 0
    pair = ensure_plate_svg(m["id"])
    assert pair is not None
    front_svg, back_svg = pair

    # Post-merge the SVG writer rasterizes the SAME composed plate the
    # preview shows (budget-aware fab pitch), so the "central at native
    # extent" failure mode is structurally impossible — the writer-agnostic
    # invariants that remain worth pinning: current version marker, true-mm
    # physical size, and real geometry that spans the plate (not a stub) yet
    # never overshoots the half-extent.
    from app.plates import PLATE_SVG_VERSION

    half_w = spec.width_um / 2.0
    for svg_path in (front_svg, back_svg):
        svg = svg_path.read_text(encoding="utf-8")
        assert PLATE_SVG_VERSION in svg[:256], f"{svg_path.name}: stale writer version"
        # Physical size is explicit mm; user units stay um via the viewBox.
        assert 'width="8.0000mm"' in svg and 'height="8.0000mm"' in svg, svg[:300]

        coords: list[float] = []
        for d_attr in re.findall(r'd="([^"]+)"', svg):
            coords.extend(
                abs(float(v))
                for v in re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", d_attr)
            )
        assert len(coords) > 200, f"{svg_path.name}: implausibly little geometry"
        reach_um = max(coords)
        assert reach_um <= half_w * 1.01 + 1.0, (
            f"{svg_path.name}: geometry overshoots the plate "
            f"({reach_um:.1f} um > {half_w:.1f} um)"
        )
        assert reach_um >= aperture_um / 2.0 * 0.5, (
            f"{svg_path.name}: geometry does not reach the aperture "
            f"({reach_um:.1f} um) — composite missing?"
        )


def test_stale_plate_svg_regenerates_on_version_bump(isolated_data):
    """A cached SVG without the current PLATE_SVG_VERSION marker must be
    rebuilt, not served — formula fixes change output under unchanged spec
    hashes."""
    from app.plates import (
        FrameSpec,
        PLATES_ROOT,
        PlateSpec,
        PLATE_SVG_VERSION,
        ensure_plate_svg,
        materialize_plate,
    )

    spec = PlateSpec(
        pattern_slug="monogram-jp",
        frame=FrameSpec(seed=6),
        width_um=8000.0,
        height_um=8000.0,
        weld_margin_um=600.0,
    )
    m = materialize_plate(spec)
    plate_dir = PLATES_ROOT / m["id"]
    # Simulate a pre-fix cache: valid files, no version marker.
    (plate_dir / "front.svg").write_text("<svg>stale</svg>", encoding="utf-8")
    (plate_dir / "back.svg").write_text("<svg>stale</svg>", encoding="utf-8")

    pair = ensure_plate_svg(m["id"])
    assert pair is not None
    front_svg, _ = pair
    text = front_svg.read_text(encoding="utf-8")
    assert PLATE_SVG_VERSION in text
    assert "stale" not in text


def test_recipe_data_keys_flow_to_box_faces(isolated_data):
    """Every recipe_data key the frontend shader binding reads
    (BoxScene.tsx) must be emitted by the pattern, survive the plate
    compose, AND survive the box manifest slimming (_lean_face_manifest
    strips only frame_scene). A missing key silently degrades the 3D
    preview to fallback values."""
    from app.boxes import materialize_box
    from app.plates import FrameSpec, PlateSpec

    # The barrier exemplar on one face and a written production face on the
    # other: between them they exercise both halves of the merge — a central
    # pattern that publishes its OWN recipe keys (the barrier's interlaced view
    # PNGs and slit lattice) and one whose keys come from the compositor.
    spec = _box_spec(faces=[])
    spec.faces["front"] = PlateSpec(
        pattern_slug="globe-duo-phase",
        frame=FrameSpec(seed=201),
        width_um=0,
        height_um=0,
    )
    spec.faces["back"] = PlateSpec(
        pattern_slug="monogram-jp",
        frame=FrameSpec(seed=202),
        width_um=0,
        height_um=0,
    )
    manifest = materialize_box(spec)

    # Post-merge contract: EVERY composed plate advertises the two-plane
    # foliage_moire recipe (the plate is a box-first carrier), but the central
    # pattern's own recipe_data keys must still merge through.
    stereo = manifest["faces"]["front"]
    assert stereo["render_recipe"] == "foliage_moire"
    rd = stereo["recipe_data"]
    for key in ("view_a_png", "view_b_png", "slit_axis_deg", "slit_period_um"):
        assert key in rd, f"barrier recipe_data missing {key!r} (frontend reads it)"
    for key in ("view_a_png", "view_b_png"):
        assert isinstance(rd[key], str) and rd[key].startswith("/data/"), (
            f"{key} must be a servable URL, got {rd[key]!r}"
        )
    assert "frame_scene" not in rd

    assert "frame_scene" not in rd

    mono = manifest["faces"]["back"]
    assert mono["render_recipe"] == "foliage_moire"
    rd = mono["recipe_data"]
    for key in ("switch_axis_deg", "carrier_period_um"):
        assert key in rd, f"recipe_data missing {key!r}"
    assert "frame_scene" not in rd
