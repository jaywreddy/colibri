"""Phase I.5 + J: plate composition and box assembly.

Exercises both the Python API and the HTTP endpoints. Uses ``tmp_path`` to
isolate the per-test data directory so cache hashes don't leak between runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    """Point all data writers at a fresh tmp dir for one test."""
    # Patch the three modules that snapshot DATA_ROOT at import time.
    from app import service, plates, boxes

    monkeypatch.setattr(service, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(plates, "PLATES_ROOT", tmp_path / "plates")
    monkeypatch.setattr(boxes, "BOXES_ROOT", tmp_path / "boxes")
    return tmp_path


def test_compose_plate_unions_frame_and_pattern(isolated_data):
    from app.plates import FrameSpec, PlateSpec, compose_plate

    spec = PlateSpec(
        pattern_slug="colibri-globe-phase",
        pattern_params={"period_um": 30.0, "extent_um": 1000.0},  # overridden by aperture
        frame=FrameSpec(seed=7),
        width_um=2400.0,
        height_um=1600.0,
    )
    composed = compose_plate(spec)
    assert composed.extent_um == (2400.0, 1600.0)
    # Front layer should now include both the central pattern and the frame band.
    # The frame band hugs the perimeter; central pattern lives in the aperture.
    # So the front mask's bounds should span the full plate.
    fx0, fy0, fx1, fy1 = composed.front.bounds
    assert fx1 - fx0 > 1800.0, "front mask should span the plate width (frame extends to edges)"
    # Back layer is unchanged from the central pattern — bounded by the aperture.
    bx0, by0, bx1, by1 = composed.back.bounds
    assert bx1 - bx0 < 1700.0, "back layer should stay within the aperture, not span the frame"


def test_materialize_plate_writes_manifest(isolated_data):
    from app.plates import FrameSpec, PlateSpec, materialize_plate, PLATES_ROOT

    spec = PlateSpec(
        pattern_slug="wayuu-kanasu-moire",
        frame=FrameSpec(seed=3),
        width_um=2000.0,
        height_um=1500.0,
    )
    manifest = materialize_plate(spec)
    assert manifest["kind"] == "plate"
    assert manifest["extent_um"] == [2000.0, 1500.0]
    # Files exist on disk.
    pid = manifest["id"]
    assert (PLATES_ROOT / pid / "manifest.json").exists()
    assert (PLATES_ROOT / pid / "front.png").exists()
    assert (PLATES_ROOT / pid / "back.png").exists()
    assert (PLATES_ROOT / pid / "front.svg").exists()
    # frame_scene rides along in recipe_data so the frontend can paint preview.
    assert "frame_scene" in manifest["recipe_data"]


def test_materialize_plate_is_cached(isolated_data):
    from app.plates import FrameSpec, PlateSpec, materialize_plate

    spec = PlateSpec(
        pattern_slug="wayuu-kanasu-moire",
        frame=FrameSpec(seed=4),
        width_um=1800.0,
        height_um=1800.0,
    )
    m1 = materialize_plate(spec)
    m2 = materialize_plate(spec)
    assert m1["id"] == m2["id"]


def test_box_face_dimensions():
    from app.boxes import face_dimensions

    W, H, D = 30000.0, 20000.0, 15000.0
    assert face_dimensions("front", W, H, D) == (W, H)
    assert face_dimensions("back", W, H, D) == (W, H)
    assert face_dimensions("top", W, H, D) == (W, D)
    assert face_dimensions("bottom", W, H, D) == (W, D)
    assert face_dimensions("left", W, H, D) == (D, H)
    assert face_dimensions("right", W, H, D) == (D, H)


def test_materialize_box_fans_out(isolated_data):
    from app.boxes import BoxSpec, FACE_IDS, materialize_box
    from app.plates import FrameSpec, PlateSpec

    # Build 6 faces all sharing the same central pattern slug but different
    # frame seeds, so the per-face plate hashes differ and the cache key
    # actually goes through fan-out.
    spec = BoxSpec(width_um=20000.0, height_um=15000.0, depth_um=10000.0)
    for i, fid in enumerate(FACE_IDS):
        spec.faces[fid] = PlateSpec(
            pattern_slug="wayuu-kanasu-moire",
            frame=FrameSpec(seed=100 + i),
            # dims are re-stamped by normalize_face_dims; we set placeholders.
            width_um=0,
            height_um=0,
        )

    box = materialize_box(spec, box_id="testbox")
    assert box["kind"] == "box"
    assert box["id"] == "testbox"
    assert set(box["faces"].keys()) == set(FACE_IDS)
    # Top/bottom should be 20000×10000, left/right 10000×15000, front/back 20000×15000.
    assert box["faces"]["front"]["extent_um"] == [20000.0, 15000.0]
    assert box["faces"]["top"]["extent_um"] == [20000.0, 10000.0]
    assert box["faces"]["left"]["extent_um"] == [10000.0, 15000.0]


def test_box_persists_and_loads(isolated_data):
    from app.boxes import BoxSpec, get_box, materialize_box
    from app.plates import PlateSpec, FrameSpec

    spec = BoxSpec(width_um=12000.0, height_um=12000.0, depth_um=12000.0)
    spec.faces["front"] = PlateSpec(
        pattern_slug="wayuu-kanasu-moire",
        frame=FrameSpec(seed=1),
        width_um=0,
        height_um=0,
    )
    m = materialize_box(spec, box_id="persist-1")
    loaded = get_box("persist-1")
    assert loaded is not None
    assert loaded["id"] == "persist-1"
    assert "front" in loaded["faces"]


# ----- HTTP layer ------------------------------------------------------------


@pytest.fixture
def client(isolated_data):
    from app.main import create_app

    return TestClient(create_app())


def test_http_plates_roundtrip(client):
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
    r = client.post("/plates/generate", json=body)
    assert r.status_code == 200, r.text
    plate = r.json()
    pid = plate["id"]
    r2 = client.get(f"/plates/{pid}")
    assert r2.status_code == 200
    assert r2.json()["id"] == pid

    r3 = client.get("/plates")
    assert r3.status_code == 200
    assert any(p["id"] == pid for p in r3.json())


def test_http_boxes_roundtrip(client):
    face = {
        "pattern_slug": "wayuu-kanasu-moire",
        "pattern_params": {},
        "frame": {"seed": 1},
        "glass": {},
        "width_um": 0,
        "height_um": 0,
        "label": "",
    }
    body = {
        "width_um": 12000.0,
        "height_um": 12000.0,
        "depth_um": 12000.0,
        "faces": {fid: {**face, "frame": {"seed": i}} for i, fid in enumerate(
            ["front", "back", "top", "bottom", "left", "right"]
        )},
        "label": "test box",
        "box_id": "http-box-1",
        "force": False,
    }
    r = client.post("/boxes/generate", json=body)
    assert r.status_code == 200, r.text
    box = r.json()
    assert box["id"] == "http-box-1"

    r2 = client.get("/boxes/http-box-1")
    assert r2.status_code == 200
