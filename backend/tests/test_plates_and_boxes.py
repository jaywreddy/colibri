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
        pattern_slug="colibri-globe-lenticular",
        # A barrier pattern — its params are the slit family.
        pattern_params={"slit_period_um": 30.0, "extent_um": 1000.0},
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
    # rotating CA↔Colombia duo-globe, left the jamón tray; food-pair-chirp
    # remains in the catalog; colibri-globe-phase was later removed as a
    # redundant twin of colibri-globe-lenticular).
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
    # The `ply` column is empty for single-plate construction and carries
    # outer/inner for bonded boxes (12 rows there).
    assert lines[0] == "face,ply,width_mm,height_mm,thickness_mm,width_um,height_um"
    assert len(lines) == 7  # header + 6 plates
    assert any(line.startswith("bottom,,24.0,24.0,0.5") for line in lines)
    assert any(line.startswith("left,,23.0,23.0,0.5") for line in lines)

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


def test_plate_svg_central_scaled_to_aperture(isolated_data):
    """Regression (fab-critical): the SVG masks must carry the SAME
    aperture-scaled central pattern as the preview PNGs. The SVG path used to
    emit the central at its native extent (~2 mm) inside the full plate — a
    wrong lithography mask that no test caught."""
    import re

    from app.plates import FrameSpec, PlateSpec, ensure_plate_svg, materialize_plate

    spec = PlateSpec(
        pattern_slug="wayuu-kanasu-moire",
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
        pattern_slug="wayuu-kanasu-moire",
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

    # Uses the curated test patterns (see frontend effectsCatalog TEST_PATTERNS):
    # the spinning globe (stereo) and the J+P monogram reveal (phase).
    spec = _box_spec(faces=[])
    spec.faces["front"] = PlateSpec(
        pattern_slug="globe-rotation-stereo",
        frame=FrameSpec(seed=201),
        width_um=0,
        height_um=0,
    )
    spec.faces["back"] = PlateSpec(
        # The honest carrier reveal — its carrier_period_um must survive the
        # box slimming for the Pattern Lab's zone quick-sets.
        pattern_slug="monogram-carrier-reveal",
        frame=FrameSpec(seed=202),
        width_um=0,
        height_um=0,
    )
    manifest = materialize_box(spec)

    # Post-merge contract: EVERY composed plate advertises the two-plane
    # foliage_moire recipe (the plate is a box-first moire carrier), but the
    # central pattern's own recipe_data keys must still merge through — the
    # Pattern Lab zone UI and any central-recipe consumers read them there.
    stereo = manifest["faces"]["front"]
    assert stereo["render_recipe"] == "foliage_moire"
    rd = stereo["recipe_data"]
    for key in ("view_a_png", "view_b_png", "slit_axis_deg", "slit_period_um"):
        assert key in rd, f"stereo recipe_data missing {key!r} (frontend reads it)"
    for key in ("view_a_png", "view_b_png"):
        assert isinstance(rd[key], str) and rd[key].startswith("/data/"), (
            f"{key} must be a servable URL, got {rd[key]!r}"
        )
    assert "frame_scene" not in rd

    reveal = manifest["faces"]["back"]
    assert reveal["render_recipe"] == "foliage_moire"
    rd = reveal["recipe_data"]
    for key in ("switch_axis_deg", "carrier_period_um"):
        assert key in rd, f"reveal recipe_data missing {key!r} (lab zone UI reads it)"
    assert "frame_scene" not in rd


# ----- capybara waterline: one resolver feeds mask + manifest + fab ----------


def test_composed_water_band_follows_the_waterline_param(isolated_data):
    """The capybara ``waterline`` param must move the COMPOSED MASK and the
    ``water_waterline_y`` the shader binds together — one resolver, no drift.

    Every plate-side consumer used to hardwire ``capybara_scanimation.WATERLINE_Y``
    (0.66), so a face that dialled the param got its band, calm patch and depth
    shear baked at the default anyway while the preview had nothing to read.
    ``plates._water_waterline_y`` is now the single resolver behind the centerpiece
    mask, the fab SVG bake, ``export_fine``'s fine-GDS band/wake and this manifest
    key. This pins the preview end of that chain to exact rows.

    Hand-derived geometry for the 12 mm square plate below (all µm unless px):
        active   = 12000 - 2·600                  = 10800
        band     = 0.12 · 10800                   =  1296
        aperture = 10800 - 2·1296                 =  8208
        art side = CENTERPIECE_FILL · 8208        =  7058.88
        central cell (extent 800, n_grid 384)     =     2.0833
        plate pitch = max(2.0833, 12000/1500)     =     8.0  → 1500 × 1500 px
        side_px  = round(7058.88 / 8)             =   882 → y0 = 750 - 441 = 309
        band top = y0 + ceil(waterline · side_px)
                 = 309 + ceil(0.45 · 882 = 396.90) = 706   ← this face
                 = 309 + ceil(0.66 · 882 = 582.12) = 892   ← the 0.66 default
    i.e. a 0.45 waterline lifts the band 186 px (~1.5 mm) up the plate.
    """
    import math

    import numpy as np
    from PIL import Image

    from app.plates import (
        ART_LEVEL,
        CENTERPIECE_FILL,
        FrameSpec,
        PLATES_ROOT,
        PlateSpec,
        materialize_plate,
    )
    from app.patterns.artistic.capybara_scanimation import WATERLINE_Y

    waterline = 0.45
    assert waterline != WATERLINE_Y, "the pin needs a non-default waterline"
    spec = PlateSpec(
        pattern_slug="capybara-scanimation",
        # extent_um at the small end of the published slider keeps the central
        # generate light (384² lattice); it does not touch the plate geometry —
        # the composed centerpiece is sized by the plate aperture, not the
        # pattern's own extent.
        pattern_params={"waterline": waterline, "extent_um": 800.0},
        frame=FrameSpec(seed=41),
        width_um=12000.0,
        height_um=12000.0,
        weld_margin_um=600.0,
    )
    m = materialize_plate(spec)

    # 1. The manifest advertises the EFFECTIVE waterline the mask was built at.
    assert m["recipe_data"]["water_waterline_y"] == pytest.approx(waterline)

    # 2. The composed BACK mask's water band moved with it. On the back layer
    #    ART_LEVEL is *only* the water band (the carrier window is FRAME_LEVEL and
    #    the foliage band is a front-layer feature), so its topmost ART row IS the
    #    baked waterline.
    pitch = m["pixel_pitch_um"]
    assert pitch == pytest.approx(8.0), f"plate pitch drifted: {pitch}"
    aperture_um = float(m["extra"]["aperture_um"])
    assert aperture_um == pytest.approx(8208.0)

    back = np.asarray(
        Image.open(PLATES_ROOT / m["id"] / "back.png").convert("L")
    )
    plate_h, plate_w = back.shape
    assert (plate_w, plate_h) == (1500, 1500)

    side_px = max(8, int(round(CENTERPIECE_FILL * aperture_um / pitch)))
    assert side_px == 882
    y0 = plate_h // 2 - side_px // 2
    assert y0 == 309

    art_rows = np.flatnonzero((back == ART_LEVEL).any(axis=1))
    assert art_rows.size > 0, "back mask carries no ART_LEVEL water band at all"
    top = int(art_rows[0])
    assert top == y0 + math.ceil(waterline * side_px) == 706, (
        f"water band starts at row {top}, expected 706 for waterline {waterline}"
    )
    # ...and strictly ABOVE where the 0.66 default would have put it (row 892) —
    # the failure mode was a band frozen at the module constant.
    assert top < y0 + math.ceil(WATERLINE_Y * side_px)
    assert y0 + math.ceil(WATERLINE_Y * side_px) == 892


def test_default_waterline_resolves_to_the_module_constant(isolated_data):
    """A face that sets no ``waterline`` — and one that sets nonsense — resolves
    to ``capybara_scanimation.WATERLINE_Y``, in the manifest key AND in the mask.

    Deliberately compose-free (the resolver + the centerpiece mask are the whole
    contract), so the default case costs no pattern generation on the 13.7 GB
    host. Mask rows: ``_centerpiece_masks`` builds the band as
    ``row/n >= waterline``, so its first row is ``ceil(waterline · n)`` —
    ceil(0.66·256) = 169 by default vs ceil(0.45·256) = 116 at 0.45.
    """
    import math

    import numpy as np

    from app.plates import (
        FrameSpec,
        PlateSpec,
        _carrier_recipe_data,
        _centerpiece_masks,
        _water_waterline_y,
    )
    from app.patterns.artistic.capybara_scanimation import WATERLINE_Y

    spec = PlateSpec(
        pattern_slug="capybara-scanimation",
        pattern_params={"extent_um": 800.0},   # no waterline key
        frame=FrameSpec(seed=42),
        width_um=12000.0,
        height_um=12000.0,
        weld_margin_um=600.0,
    )
    assert _carrier_recipe_data(spec)["water_waterline_y"] == pytest.approx(WATERLINE_Y)

    # The resolver never raises on junk — the API validates against the ParamSpec
    # but direct callers (fab CLIs, tests) do not, and the flow field divides by
    # ``1 - waterline_y``.
    assert _water_waterline_y(None) == WATERLINE_Y
    assert _water_waterline_y({}) == WATERLINE_Y
    for junk in (0.0, 1.0, -0.2, 1.5, "wet", None, float("nan")):
        assert _water_waterline_y({"waterline": junk}) == WATERLINE_Y, junk

    # And the composed centerpiece mask splits at the same line. masks[1] is the
    # water band (below the line, minus the submerged body).
    n_px = 256
    band_default = _centerpiece_masks("capybara-scanimation", n_px, {})[1]
    band_high = _centerpiece_masks("capybara-scanimation", n_px, {"waterline": 0.45})[1]
    assert int(np.flatnonzero(band_default.any(axis=1))[0]) == math.ceil(
        WATERLINE_Y * n_px
    ) == 169
    assert int(np.flatnonzero(band_high.any(axis=1))[0]) == math.ceil(0.45 * n_px) == 116
    # A higher waterline means MORE water: the 0.45 band strictly contains more
    # rows than the default one.
    assert band_high.sum() > band_default.sum()


def test_fab_water_band_carves_the_submerged_capybara(isolated_data):
    """No fab writer may print ripple or barrier gold over the submerged body.

    The convention is the composed preview's: the scanimation band is the water
    AROUND the animal, and the submerged body keeps the plain carrier.
    ``_centerpiece_masks`` has always built the preview's back water zone as
    ``below & ~capy``, but both FAB writers used to treat the band as pure
    ``below`` — ``ensure_plate_svg`` consumed ``_capybara_and_water``'s
    ``water_band`` (then a copy of ``below``) and ``export_fine``'s exact vector
    builders swept the full-width band — so front.svg striped 45 µm slit bars
    across the capybara and back.svg / fine.gds ran the ripple interleave under it.

    Deliberately compose-free (no materialize, no export): the writers' geometry
    comes from the shared builder ``capybara_scanimation._build`` (what
    ``ensure_plate_svg`` bakes) and the two exact vector rect builders (what
    ``build_plate_fine`` emits), so pinning those three is the whole contract at a
    fraction of a compose. Each half also runs the UNCARVED construction and
    asserts it WOULD have hit the animal, so the pin cannot go quietly vacuous if
    the crest field or the comb ever stops reaching the body.
    """
    import numpy as np

    from app.patterns.artistic import capybara_scanimation as capyscan
    from app.plates import _centerpiece_masks

    n = 192
    wl = 0.6  # more submerged body than the 0.66 default → a bigger carve to check
    scene = capyscan._capybara_and_water(n, wl)
    capy, submerged = scene["capy"], scene["capy_below"]
    assert submerged.any(), "the pin needs a half-submerged body at this waterline"

    # 1. ONE band definition. The pattern's band is the water minus the animal...
    assert not (scene["water_band"] & capy).any()
    rows = np.arange(n)[:, None] / n
    below = np.broadcast_to(rows >= wl, (n, n))
    assert np.array_equal(scene["water_band"], below & ~capy)
    # ...and it is byte-identical to the composed preview's back art, which is the
    # authority here (same silhouette source, same waterline resolver).
    assert np.array_equal(
        _centerpiece_masks("capybara-scanimation", n, {"waterline": wl})[1],
        scene["water_band"],
    )

    # 2. The fab SVG bake (ensure_plate_svg consumes exactly these masks).
    b = capyscan._build(
        extent_um=1200.0,
        frame_pitch_um=60.0,
        n_phases=4,
        carrier_period_um=24.0,
        waterline_y=wl,
        n_grid=n,
    )
    for k, ph in enumerate(b["phase_masks"]):
        assert not (ph & submerged).any(), f"ripple phase {k} prints on the animal"
    # back.svg's interleave: the atomic slot fill must not bleed onto the body
    # either (a straddling slot is dropped, see _interleave_phases' keep gate).
    assert not (b["back_water"] & submerged).any(), "back.svg interleaves under the body"
    # front.svg's slit-barrier bars over the band.
    barrier_bars = (~b["barrier"]) & b["water_band"]
    assert barrier_bars.any(), "no barrier bars at all — the front pin is vacuous"
    assert not (barrier_bars & submerged).any(), "front.svg bars the submerged body"
    # Non-vacuity: with an UNCARVED band the very same crest field lands on the
    # animal, so the assertions above are testing the carve, not an empty mask.
    naive = capyscan._ripple_phase_masks(n, b["cell_um"], 60.0, 4, below, wl)
    assert any((ph & submerged).any() for ph in naive), (
        "an uncarved band puts no crest on the animal — the pin proves nothing"
    )

    # 3. The fine GDS (export_fine's EXACT vector builders — no plate cache
    #    needed: they read only the spec's aperture arithmetic).
    from app.export_fine import (
        _art_box_um,
        _sample_art_mask,
        _scanimation_back_frame_rects,
        _scanimation_barrier_bar_rects,
    )
    from app.plates import FrameSpec, PlateSpec

    spec = PlateSpec(
        pattern_slug="capybara-scanimation",
        pattern_params={"waterline": wl, "extent_um": 800.0},
        frame=FrameSpec(seed=43),
        width_um=12000.0,
        height_um=12000.0,
        weld_margin_um=600.0,
    )
    art = _art_box_um(spec)
    side = art[2] - art[0]

    def _on_body(rects):
        """How many emitted rects sample the capybara silhouette.

        Probed at each rect's own x samples (left edge, center, right edge — the
        SAME three the carve tests) crossed with three y's inside the rect, mapped
        to art-box normalized coords with the builders' own transform. The carve
        clears whole y-sample cells, so any y inside a surviving rect is one the
        carve already cleared: this probe cannot be stricter than the carve.

        The FULL silhouette, like the carve: every rect here lives inside the water
        band, so a hit is the submerged body — and probing ``capy & below`` instead
        would let the raster row that STRADDLES the waterline off the hook.
        """
        if rects.shape[0] == 0:
            return 0
        eps = 1e-6
        # Probe the SAME points the carve sampled: the 2 um ladder CELL CENTERS
        # (top-anchored at the band top, exactly _carve_submerged_from_bars'
        # ladder — rect edges snap to cell boundaries, so any other y, e.g. a
        # rect edge +/- eps, can nearest-neighbour into the adjacent 37 um mask
        # pixel and flag body where the carve honestly saw water).
        band_y1 = float(rects[:, 3].max())
        band_y0 = float(rects[:, 2].min())
        n_y = max(2, int(round((band_y1 - band_y0) / 2.0)))
        dy = (band_y1 - band_y0) / n_y
        centers = band_y1 - (np.arange(n_y) + 0.5) * dy
        xs = (rects[:, 0] + eps, 0.5 * (rects[:, 0] + rects[:, 1]), rects[:, 1] - eps)
        hit = np.zeros(rects.shape[0], dtype=bool)
        inside = (centers[None, :] > rects[:, 2:3]) & (centers[None, :] < rects[:, 3:4])
        for x in xs:
            on = _sample_art_mask(
                capy, ((x - art[0]) / side)[:, None], ((art[3] - centers) / side)[None, :]
            )
            hit |= (on & inside).any(axis=1)
        return int(hit.sum())

    for name, build in (
        ("front slit-barrier comb", _scanimation_barrier_bar_rects),
        ("back ripple interleave", _scanimation_back_frame_rects),
    ):
        carved = build(spec, wl, 60.0, 4, capy)
        assert carved.shape[0] > 0, f"{name}: emitted nothing at all"
        assert _on_body(carved) == 0, f"{name}: {_on_body(carved)} rects on the animal"
        # Non-vacuity again: without the silhouette the builder DOES cover it.
        assert _on_body(build(spec, wl, 60.0, 4, None)) > 0, (
            f"{name}: even the uncarved build misses the animal — pin proves nothing"
        )
