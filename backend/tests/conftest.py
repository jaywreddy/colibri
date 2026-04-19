"""Shared pytest fixtures. We isolate DATA_ROOT to a tmp_path per test so the
on-disk pattern cache never leaks between tests or between runs."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


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
