"""Plate composition and ring-box assembly (BoxSpec v2).

Exercises both the Python API and the HTTP endpoints. The shared
``isolated_data`` fixture (conftest.py) points the per-test data directory
at tmp_path so cache hashes don't leak between runs.

Box geometry note: with the default 1/4" foil + 0.5 mm safety on 0.5 mm
glass, the keep-out is 3.425 mm per edge, so test boxes need min plate
sides comfortably above ~10 mm — we use 20-30 mm boxes throughout.
"""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient


def test_compose_plate_unions_frame_and_pattern(isolated_data):
    from app.plates import FrameSpec, PlateSpec, compose_plate

    spec = PlateSpec(
        pattern_slug="colibri-globe-phase",
        pattern_params={"period_um": 30.0, "extent_um": 1000.0},
        frame=FrameSpec(seed=7),
        width_um=10000.0,
        height_um=8000.0,
        weld_margin_um=500.0,
    )
    composed = compose_plate(spec)
    assert composed.extent_um == (10000.0, 8000.0)
    # Front layer = central pattern + frame band. The frame hugs the *active*
    # rectangle (inset by weld), so its outer edge lands at width - 2·weld.
    fx0, fy0, fx1, fy1 = composed.front.bounds
    active_w = spec.width_um - 2 * spec.weld_margin_um
    assert fx1 - fx0 > active_w * 0.6, "front mask should span most of the active rect"
    # No front pixel may overshoot the active rect (no gold in weld zone).
    assert fx0 >= -active_w / 2 - 1.0, "front mask leaks past the weld boundary on the left"
    assert fx1 <= active_w / 2 + 1.0, "front mask leaks past the weld boundary on the right"
    # Back layer is unchanged from the central pattern — bounded by its extent.
    bx0, by0, bx1, by1 = composed.back.bounds
    assert bx1 - bx0 < 1100.0, "back layer should stay within the central pattern extent"


def test_weld_margin_zeroes_border_in_raster(isolated_data):
    """No gold pixels may appear inside the weld zone of either layer."""
    from PIL import Image
    from app.plates import FrameSpec, PlateSpec, PLATES_ROOT, materialize_plate

    spec = PlateSpec(
        pattern_slug="wayuu-kanasu-moire",
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
        pattern_slug="wayuu-kanasu-moire",
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
        pattern_slug="wayuu-kanasu-moire",
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
            pattern_slug="wayuu-kanasu-moire",
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
            pattern_slug="wayuu-kanasu-moire",
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
    assert (spec.width_um, spec.depth_um, spec.height_um) == (50000.0, 50000.0, 40000.0)
    assert set(spec.faces) == set(FACE_IDS)
    # Confirmed six-face plan (rev: round-8 user decisions — front carries the
    # rotating CA↔Colombia duo-globe, left the jamón tray; colibri-globe-phase
    # and food-pair-chirp remain in the catalog).
    expected_slug = {
        "front": "globe-duo-phase",
        "back": "capybara-scanimation",
        "top": "monogram-jp",
        "bottom": "inscription-line",
        "left": "jamon-tray",
        "right": "gear-quill-switch",
    }
    for i, fid in enumerate(FACE_IDS):
        assert spec.faces[fid].pattern_slug == expected_slug[fid]
        # Per-face seed 100..105 in FACE_IDS order (distinct moiré carrier angle).
        assert spec.faces[fid].frame.seed == 100 + i


# ----- HTTP layer ------------------------------------------------------------


@pytest.fixture
def app_client(isolated_data):
    from app.main import create_app

    return TestClient(create_app())


def test_http_plates_roundtrip(app_client):
    body = {
        "spec": {
            "pattern_slug": "wayuu-kanasu-moire",
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
        "pattern_slug": "wayuu-kanasu-moire",
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


def test_http_box_fab_export_has_cutlist_and_assembly(app_client):
    # One face keeps the export fast; the cut list always covers all six.
    body = _http_box_body(box_id="export-1")
    body["faces"] = {"front": body["faces"]["front"]}
    r = app_client.post("/boxes/generate", json=body)
    assert r.status_code == 200, r.text

    rz = app_client.get("/export/box/export-1/fab.zip")
    assert rz.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(rz.content))
    names = set(zf.namelist())
    assert "box.json" in names
    assert "CUTLIST.csv" in names
    assert "ASSEMBLY.md" in names
    assert "front/front.svg" in names
    assert "front/manifest.json" in names
    # The frame-scene sidecar is a live-preview artifact, not fab data.
    assert "front/scene.json" not in names

    csv_text = zf.read("CUTLIST.csv").decode()
    lines = csv_text.strip().splitlines()
    assert lines[0] == "face,width_mm,height_mm,thickness_mm,width_um,height_um"
    assert len(lines) == 7  # header + 6 plates
    assert any(line.startswith("bottom,24.0,24.0,0.5") for line in lines)
    assert any(line.startswith("left,23.0,23.0,0.5") for line in lines)

    md = zf.read("ASSEMBLY.md").decode()
    # Real numbers: keep-out, foil width, hinge segment length, rod length.
    assert "3.425 mm" in md
    assert "6.350 mm copper foil" in md
    # Hinge: run = 0.8*24000 = 19200; seg = (19200-1600)/5 = 3520; rod = 22400.
    assert "3.52 mm" in md
    assert "22.4 mm" in md

    # Embedded face manifests stay lean inside the archive too.
    import json as _json
    embedded = _json.loads(zf.read("front/manifest.json").decode())
    assert "frame_scene" not in embedded.get("recipe_data", {})
    box_json = _json.loads(zf.read("box.json").decode())
    for face in box_json["faces"].values():
        assert "frame_scene" not in face.get("recipe_data", {})
