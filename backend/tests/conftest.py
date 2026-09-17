"""Shared pytest fixtures. We isolate DATA_ROOT to a tmp_path per test so the
on-disk pattern cache never leaks between tests or between runs."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Cheapest central pattern in the catalog, for tests that only assert MANIFEST
# SHAPE / API contracts and not geometry: the J+P monogram at the smallest legal
# extent -> two glyphs on a small raster, a few ms. Every value sits inside its
# ParamSpec bounds (service.validate_params runs before the variant hash). Same
# geometry test_cache_integrity.py uses; tests that genuinely pin the monogram's
# DEFAULT geometry must keep passing `{}`.
CHEAP_SLUG = "monogram-jp"
CHEAP_PARAMS: dict[str, float] = {
    "extent_um": 500.0,
    "overlap": 0.76,
    "j_period_um": 4.47,
    "p_period_um": 6.02,
}


@pytest.fixture
def cheap_pattern() -> tuple[str, dict[str, float]]:
    """(slug, params) of the cheapest generate the catalog allows."""
    return CHEAP_SLUG, dict(CHEAP_PARAMS)


@pytest.fixture(scope="session")
def shared_pattern_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-wide central-pattern cache shared by all plate/box tests.

    Pattern variants are content-addressed (slug + params hash), so sharing
    the cache across tests is safe — and essential for speed: a cold central
    pattern costs seconds, and without sharing every plate/box test pays it
    again in its own tmp dir.
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
    from app import boxes, service
    from app.plates import spec as plate_spec

    monkeypatch.setattr(service, "DATA_ROOT", shared_pattern_cache)
    # app.plates.spec is where the roots are DEFINED; every submodule reads
    # them through the module object and the package forwards ``PLATES_ROOT``
    # to it (see app/plates/__init__.py::__getattr__), so patching here is the
    # one place that moves them for readers and writers alike.
    monkeypatch.setattr(plate_spec, "DATA_ROOT", shared_pattern_cache)
    monkeypatch.setattr(plate_spec, "PLATES_ROOT", tmp_path / "plates")
    monkeypatch.setattr(boxes, "BOXES_ROOT", tmp_path / "boxes")
    return tmp_path


@pytest.fixture
def isolated_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point DATA_ROOT at a per-test tmp directory AND patch the module-level
    references that were already imported elsewhere. FastAPI static mount is
    set up in create_app(), so we must patch before importing main."""
    from app import service as service_mod

    new_root = tmp_path / "data"
    new_root.mkdir()
    monkeypatch.setattr(service_mod, "DATA_ROOT", new_root)
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


