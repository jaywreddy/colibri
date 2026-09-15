"""Shared pytest fixtures. We isolate DATA_ROOT to a tmp_path per test so the
on-disk pattern cache never leaks between tests or between runs."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Cheapest central pattern in the catalog, for tests that only assert MANIFEST
# SHAPE / API contracts and not geometry: the coarsest legal kanasü weave at
# the smallest legal extent -> ~3x3 diamonds on a 20x20 px raster, a few ms
# instead of the ~90 s the default variant costs. Every value sits inside its
# ParamSpec bounds (service.validate_params runs before the variant hash) and
# clears the litho floor (200 um * 0.29 = 58 um features). Same geometry
# test_cache_integrity.py uses; tests that genuinely pin wayuu's DEFAULT
# geometry must keep passing `{}`.
CHEAP_SLUG = "wayuu-kanasu-moire"
CHEAP_PARAMS: dict[str, float] = {
    "period_um": 200.0,
    "duty": 0.5,
    "rotation_deg": 2.0,
    "extent_um": 500.0,
}


@pytest.fixture
def cheap_pattern() -> tuple[str, dict[str, float]]:
    """(slug, params) of the cheapest generate the catalog allows."""
    return CHEAP_SLUG, dict(CHEAP_PARAMS)


@pytest.fixture(scope="session")
def shared_pattern_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-wide central-pattern cache shared by all plate/box tests.

    Pattern variants are content-addressed (slug + params hash), so sharing
    the cache across tests is safe — and essential for speed: generating the
    wayuu central pattern from scratch costs ~90 s, and without sharing every
    plate/box test pays it again in its own tmp dir.
    """
    return tmp_path_factory.mktemp("pattern-cache")


@pytest.fixture
def isolated_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_pattern_cache: Path,
) -> Path:
    """Per-test plate/box roots + the session-shared central-pattern cache.

    PLATES_ROOT/BOXES_ROOT stay per-test (listing/caching tests rely on a
    fresh dir), while DATA_ROOT points at the shared pattern cache so the
    expensive central patterns generate once per session, not once per test.
    Patches the modules that snapshot DATA_ROOT-derived paths at import time
    so reads land where service.materialize writes.
    """
    from app import boxes, plates, service

    monkeypatch.setattr(service, "DATA_ROOT", shared_pattern_cache)
    monkeypatch.setattr(plates, "DATA_ROOT", shared_pattern_cache)
    monkeypatch.setattr(plates, "PLATES_ROOT", tmp_path / "plates")
    monkeypatch.setattr(boxes, "BOXES_ROOT", tmp_path / "boxes")
    return tmp_path


@pytest.fixture
def isolated_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point DATA_ROOT at a per-test tmp directory AND patch the module-level
    references that were already imported elsewhere. FastAPI static mount is
    set up in create_app(), so we must patch before importing main."""
    from app import service as service_mod
    from app.api import sim as sim_mod

    new_root = tmp_path / "data"
    new_root.mkdir()
    monkeypatch.setattr(service_mod, "DATA_ROOT", new_root)
    monkeypatch.setattr(sim_mod, "DATA_ROOT", new_root)
    return new_root


@pytest.fixture
def client(isolated_data_root: Path) -> TestClient:
    """TestClient with the lifespan disabled (so we don't seed every pattern
    on every test — that would take ~20 seconds per test)."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles

    from app.api import patterns as patterns_api
    from app.api import sim as sim_api

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(patterns_api.router)
    app.include_router(sim_api.router)
    app.mount("/data", StaticFiles(directory=str(isolated_data_root)), name="data")

    with TestClient(app) as c:
        yield c


@pytest.fixture
def seeded_wayuu_moire(client: TestClient) -> tuple[str, str]:
    """POST /patterns/generate for a Wayuu kanasü moiré variant and yield
    (slug, variant). Downstream sim tests reuse this instead of re-seeding."""
    r = client.post("/patterns/generate", json={"slug": "wayuu-kanasu-moire", "params": {}})
    assert r.status_code == 200, r.text
    m = r.json()
    return m["slug"], m["variant"]
