"""Cache-integrity contracts for the disposable ``backend/data`` cache.

Four classes of damage are pinned here, none of them reachable from the other
test files: those all run against a fresh data root, so a slot written by an
OLDER build — or by a build that died mid-publish — never exists for them.

  1. **Stale version marker.** Cache keys hash user spec/params ONLY, so the
     version markers on the hit path (``service.PATTERN_GEN_VERSION``,
     ``plates.PLATE_COMPOSE_VERSION``) are the only thing that invalidates a
     warm cache after a code or constant change. A slot carrying an older
     marker must regenerate in place, not be served.
  2. **Truncated manifest.** A crash between the payload files and the manifest
     rename must read as a MISS (regenerate over it), never as a permanently
     poisoned slot that raises for every later request.
  3. **Wiped plate cache.** ``data/`` is disposable, so ``just clean`` between a
     box generate and its fab export must not make a saved box un-exportable —
     fab.zip is the artifact that goes to the mask shop.
  4. **Rename-based publishing.** No ``*.tmp`` staging file may survive a
     completed materialize.

Everything runs on deliberately cheap geometry — the coarsest legal kanasü
weave at the smallest legal extent (a 20x20 px central raster) and the smallest
plates the foil keep-out allows. Cache plumbing is what is under test, not
optics, and this host cannot afford a second heavy chunk (CLAUDE.md). The
concurrency half of the contract is asserted structurally (the per-id lock
registry, and which key each generate path takes) rather than by racing real
generates in threads, for the same reason.
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Cheapest central pattern in the catalog: coarsest weave period at the
# smallest legal extent -> ~3x3 diamonds on a 20x20 px raster. Every value is
# inside its ParamSpec bounds (service.validate_params runs before the hash) and
# clears the litho floor (200 um * 0.29 = 58 um features).
CHEAP_SLUG = "wayuu-kanasu-moire"
CHEAP_PARAMS = {"period_um": 200.0, "duty": 0.5, "rotation_deg": 2.0, "extent_um": 500.0}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_no_tmp(*roots: Path) -> None:
    """Every publish is a rename, so a finished slot holds no staging file."""
    for root in roots:
        if not root.exists():
            continue
        leftovers = sorted(str(p) for p in root.rglob("*.tmp"))
        assert leftovers == [], f"staging files survived a completed materialize: {leftovers}"


def _cheap_plate_spec(seed: int):
    """A small plate whose aperture still clears the frame band (5 mm - 2*0.3 mm
    active, ~0.53 mm band, ~3.3 mm aperture)."""
    from app.plates import FrameSpec, PlateSpec

    return PlateSpec(
        pattern_slug=CHEAP_SLUG,
        pattern_params=dict(CHEAP_PARAMS),
        frame=FrameSpec(seed=seed),
        width_um=5000.0,
        height_um=5000.0,
        weld_margin_um=300.0,
    )


def _cheap_box_spec():
    """A 16 mm cube — near the smallest that clears the default 3.425 mm foil
    keep-out plus the 3 mm minimum aperture (assembly.validate_assembly) — with
    ONE face populated. Cache integrity is per-slot, and six faces would be six
    plate composes plus six fab-mask builds for no extra coverage.
    """
    from app.boxes import BoxSpec
    from app.plates import FrameSpec, PlateSpec

    # Width drives hinge segment length; 16 mm leaves the default hinge's
    # segments shorter than the tube OD (uncuttable, validation rejects).
    spec = BoxSpec(width_um=20000.0, depth_um=16000.0, height_um=16000.0)
    spec.faces["front"] = PlateSpec(
        pattern_slug=CHEAP_SLUG,
        pattern_params=dict(CHEAP_PARAMS),
        frame=FrameSpec(seed=903),
        # dims/glass/weld are re-stamped by normalize_face_dims.
        width_um=0,
        height_um=0,
    )
    return spec


@pytest.fixture
def app_client(isolated_data) -> TestClient:
    """The full app (plates/boxes/export routers) on the isolated cache roots."""
    from app.main import create_app

    return TestClient(create_app())


# ----- 1. stale version markers ----------------------------------------------


def test_stale_gen_version_regenerates_the_variant(client: TestClient, isolated_data_root: Path):
    from app.service import PATTERN_GEN_VERSION

    body = {"slug": CHEAP_SLUG, "params": CHEAP_PARAMS}
    r = client.post("/patterns/generate", json=body)
    assert r.status_code == 200, r.text
    first = r.json()
    assert first["gen_version"] == PATTERN_GEN_VERSION
    slot = isolated_data_root / first["slug"] / first["variant"]

    # Forge the slot a pre-bump build would have left behind: identical params
    # (so the key still hits), older generation code, stale payload.
    manifest = json.loads((slot / "manifest.json").read_text())
    manifest["gen_version"] = PATTERN_GEN_VERSION - 1
    manifest["stale_marker"] = "pre-bump"
    (slot / "manifest.json").write_text(json.dumps(manifest))
    (slot / "front.png").write_bytes(b"stale bytes, not a png")

    r2 = client.post("/patterns/generate", json=body)
    assert r2.status_code == 200, r2.text
    fresh = r2.json()
    # The variant hash covers params only, so the slot is the same one — the
    # point is that it was rebuilt, not re-served.
    assert fresh["variant"] == first["variant"]
    assert fresh["gen_version"] == PATTERN_GEN_VERSION
    assert "stale_marker" not in fresh, "the stale manifest was served verbatim"
    on_disk = json.loads((slot / "manifest.json").read_text())
    assert on_disk["gen_version"] == PATTERN_GEN_VERSION
    assert "stale_marker" not in on_disk
    assert (slot / "front.png").read_bytes()[:8] == PNG_MAGIC, (
        "payload PNGs must be rewritten on a version-mismatch regenerate"
    )
    _assert_no_tmp(slot)


def test_stale_compose_version_regenerates_the_plate(isolated_data, monkeypatch):
    from app import plates as P

    # The per-plate compute lock is asserted here rather than in its own test:
    # proving materialize_plate takes it needs a real compose, and this test
    # already pays for two (CLAUDE.md — no spare heavy composes on this host).
    keys: list[str] = []
    real_lock = P.cache_lock
    monkeypatch.setattr(P, "cache_lock", lambda key: (keys.append(key), real_lock(key))[1])

    spec = _cheap_plate_spec(seed=901)
    first = P.materialize_plate(spec)
    pid = first["id"]
    slot = P.PLATES_ROOT / pid
    assert first["compose_version"] == P.PLATE_COMPOSE_VERSION
    assert keys == [f"plate:{pid}"], "compose must serialize on the plate's cache id"

    manifest = json.loads((slot / "manifest.json").read_text())
    manifest["compose_version"] = P.PLATE_COMPOSE_VERSION - 1
    manifest["stale_marker"] = "pre-bump"
    (slot / "manifest.json").write_text(json.dumps(manifest))
    (slot / "front.png").write_bytes(b"stale bytes, not a png")

    fresh = P.materialize_plate(spec)
    assert fresh["id"] == pid
    assert fresh["compose_version"] == P.PLATE_COMPOSE_VERSION
    assert "stale_marker" not in fresh, "the stale manifest was served verbatim"
    on_disk = json.loads((slot / "manifest.json").read_text())
    assert on_disk["compose_version"] == P.PLATE_COMPOSE_VERSION
    assert "stale_marker" not in on_disk
    assert (slot / "front.png").read_bytes()[:8] == PNG_MAGIC, (
        "composed PNGs must be rewritten on a compose-version regenerate"
    )
    # The cache-HIT path holds the lock too, so a second request cannot start a
    # compose into a slot another thread is still publishing.
    assert keys == [f"plate:{pid}", f"plate:{pid}"]
    _assert_no_tmp(slot)


# ----- 2. corrupt manifests are misses, not poisoned slots --------------------


def test_truncated_variant_manifest_is_a_miss(client: TestClient, isolated_data_root: Path):
    from app.service import PATTERN_GEN_VERSION

    body = {"slug": CHEAP_SLUG, "params": CHEAP_PARAMS}
    r = client.post("/patterns/generate", json=body)
    assert r.status_code == 200, r.text
    first = r.json()
    slot = isolated_data_root / first["slug"] / first["variant"]
    manifest_path = slot / "manifest.json"

    # Killed between the payload PNGs and the manifest rename.
    raw = manifest_path.read_text()
    manifest_path.write_text(raw[: len(raw) // 2])
    with pytest.raises(json.JSONDecodeError):
        json.loads(manifest_path.read_text())  # the setup really is corrupt

    r2 = client.post("/patterns/generate", json=body)
    # Not a 400 "Generation failed: JSONDecodeError(...)" and not a 500.
    assert r2.status_code == 200, r2.text
    fresh = r2.json()
    assert fresh["variant"] == first["variant"]
    assert fresh["gen_version"] == PATTERN_GEN_VERSION
    assert json.loads(manifest_path.read_text())["variant"] == first["variant"]
    _assert_no_tmp(slot)


def test_truncated_plate_manifest_is_a_miss_not_a_500(app_client: TestClient, isolated_data):
    from app import plates as P

    spec = _cheap_plate_spec(seed=902)
    first = P.materialize_plate(spec)
    pid = first["id"]
    manifest_path = P.PLATES_ROOT / pid / "manifest.json"
    raw = manifest_path.read_text()
    manifest_path.write_text(raw[: len(raw) // 2])
    with pytest.raises(json.JSONDecodeError):
        json.loads(manifest_path.read_text())

    # Readers report corruption as absence, so the route 404s (recoverable)
    # instead of raising JSONDecodeError out of the handler as a 500...
    assert P.get_plate(pid) is None
    r = app_client.get(f"/plates/{pid}")
    assert r.status_code == 404, r.text
    # ...the listing drops the slot instead of failing wholesale...
    assert all(p.get("id") != pid for p in P.list_plates())
    # ...and the next generate rebuilds over it.
    fresh = P.materialize_plate(spec)
    assert fresh["id"] == pid
    assert fresh["compose_version"] == P.PLATE_COMPOSE_VERSION
    r2 = app_client.get(f"/plates/{pid}")
    assert r2.status_code == 200, r2.text
    assert r2.json()["id"] == pid
    _assert_no_tmp(P.PLATES_ROOT / pid)


# ----- 3. fab export survives a wiped plate cache -----------------------------


def test_box_fab_export_rebuilds_a_wiped_plate_cache(app_client: TestClient, isolated_data):
    from app import plates as P
    from app.boxes import BOXES_ROOT, materialize_box

    box = materialize_box(_cheap_box_spec(), box_id="cache-wipe-1")
    pid = box["faces"]["front"]["id"]
    assert (P.PLATES_ROOT / pid / "manifest.json").exists()

    # `just clean`, or any cache eviction, between generate and export.
    shutil.rmtree(P.PLATES_ROOT)
    assert not P.PLATES_ROOT.exists()

    r = app_client.get("/export/box/cache-wipe-1/fab.zip")
    assert r.status_code == 200, r.content[:400]
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"box.json", "CUTLIST.csv", "ASSEMBLY.md", "FINE_MASKS.json"} <= names
    for member in ("front/manifest.json", "front/front.png", "front/front.svg"):
        assert member in names, f"rebuilt face is missing {member} from the archive"

    # The rebuild recomposes from the face's own saved spec, whose hash IS the
    # plate id — so it lands back in the slot box.json already points at.
    assert (P.PLATES_ROOT / pid / "manifest.json").exists()
    assert json.loads(zf.read("box.json"))["faces"]["front"]["id"] == pid
    _assert_no_tmp(P.PLATES_ROOT, BOXES_ROOT)


# ----- 4. atomic publishing + the per-id lock registry -----------------------


def test_publishing_a_variant_leaves_no_staging_files(
    client: TestClient, isolated_data_root: Path
):
    body = {"slug": CHEAP_SLUG, "params": CHEAP_PARAMS}
    r = client.post("/patterns/generate", json=body)
    assert r.status_code == 200, r.text
    m = r.json()
    slot = isolated_data_root / m["slug"] / m["variant"]
    for name in ("front.png", "back.png", "thumbnail.png", "manifest.json"):
        assert (slot / name).exists(), f"missing published payload {name}"
    _assert_no_tmp(isolated_data_root)

    # The lazy SVG pair publishes the same way (write_text_atomic + a manifest
    # rewrite), so it must not leave staging files either.
    r2 = client.get(f"/patterns/{m['slug']}/{m['variant']}/svg")
    assert r2.status_code == 200, r2.text
    assert (slot / "front.svg").exists() and (slot / "back.svg").exists()
    _assert_no_tmp(isolated_data_root)


def test_cache_lock_is_one_lock_per_cache_id():
    from app.service import cache_lock

    lock = cache_lock("plate:deadbeefcafe")
    assert cache_lock("plate:deadbeefcafe") is lock, (
        "two callers on the same cache id must get the SAME lock or they don't serialize"
    )
    assert cache_lock("plate:0123456789ab") is not lock, "distinct ids must not block each other"
    # Keys are namespaced, so a variant hash can never share a lock with a plate
    # hash that happens to be the same string.
    assert cache_lock(f"pattern:{CHEAP_SLUG}:deadbeefcafe") is not lock
    assert lock.acquire(blocking=False) is True
    lock.release()


def test_variant_generate_takes_the_variant_cache_lock(isolated_data_root: Path, monkeypatch):
    from app import service

    keys: list[str] = []
    real_lock = service.cache_lock
    monkeypatch.setattr(service, "cache_lock", lambda key: (keys.append(key), real_lock(key))[1])

    first = service.materialize(CHEAP_SLUG, dict(CHEAP_PARAMS))
    service.materialize(CHEAP_SLUG, dict(CHEAP_PARAMS))  # cache hit
    expected = f"pattern:{CHEAP_SLUG}:{first['variant']}"
    assert keys == [expected, expected], (
        "both the generate and the cache-hit path must hold the variant's lock"
    )
