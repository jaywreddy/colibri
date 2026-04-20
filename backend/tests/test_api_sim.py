"""Backend API contract for /sim — atlas shape, caching, and error codes."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_fft_returns_atlas_png(client: TestClient, seeded_wayuu_moire) -> None:
    slug, variant = seeded_wayuu_moire
    r = client.post(
        "/sim/fft",
        json={
            "slug": slug,
            "variant": variant,
            "wavelengths_um": [0.65, 0.55, 0.45],
            "n_angles": 64,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slug"] == slug and body["variant"] == variant
    assert body["atlas_png"].endswith(".png")
    assert body["atlas_png"].startswith(f"/data/{slug}/{variant}/")


def test_fft_invalid_variant_returns_404(client: TestClient) -> None:
    r = client.post(
        "/sim/fft",
        json={"slug": "wayuu-kanasu-moire", "variant": "deadbeef42"},
    )
    assert r.status_code == 404


def test_propagate_returns_tiled_atlas(client: TestClient, seeded_wayuu_moire) -> None:
    slug, variant = seeded_wayuu_moire
    r = client.post(
        "/sim/propagate",
        json={
            "slug": slug,
            "variant": variant,
            "wavelengths_um": [0.55],
            "view_angles_deg": [0.0, 15.0],
            "downsample": 8,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == 2  # two view angles
    assert body["cols"] == 1  # one wavelength
    assert isinstance(body["tile"], list) and len(body["tile"]) == 2


def test_propagate_cached_second_call(client: TestClient, seeded_wayuu_moire) -> None:
    slug, variant = seeded_wayuu_moire
    payload = {
        "slug": slug,
        "variant": variant,
        "wavelengths_um": [0.55],
        "view_angles_deg": [0.0],
        "downsample": 8,
    }
    r1 = client.post("/sim/propagate", json=payload)
    r2 = client.post("/sim/propagate", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    # first call may or may not be cached depending on whether a prior test wrote
    # the same atlas hash; the second must be cached regardless.
    assert r2.json()["cached"] is True


def test_propagate_unknown_variant_404(client: TestClient) -> None:
    r = client.post(
        "/sim/propagate",
        json={"slug": "wayuu-kanasu-moire", "variant": "nope0000ff"},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Phase D — /sim/carpet contract. Used by the near_field_carpet recipe for
# tairona-talbot and muzo-emerald-zone. Atlas is a vertical stack of 2D
# tiles, one per z-slice; the shader picks a row via uZSlice.
# ---------------------------------------------------------------------------
@pytest.fixture
def seeded_tairona_talbot(client: TestClient) -> tuple[str, str]:
    r = client.post(
        "/patterns/generate", json={"slug": "tairona-talbot", "params": {}}
    )
    assert r.status_code == 200, r.text
    m = r.json()
    return m["slug"], m["variant"]


def test_carpet_returns_z_stack_atlas(
    client: TestClient, seeded_tairona_talbot
) -> None:
    slug, variant = seeded_tairona_talbot
    r = client.post(
        "/sim/carpet",
        json={
            "slug": slug,
            "variant": variant,
            "wavelength_um": 0.55,
            "z_min_um": 0.0,
            "z_max_um": 4000.0,
            "n_slices": 16,
            "downsample": 8,
            "tile_size": 64,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == 16
    assert body["cols"] == 1
    assert body["tile"] == [64, 64]
    assert body["atlas_png"].endswith(".png")
    assert body["atlas_png"].startswith(f"/data/{slug}/{variant}/")


def test_carpet_cache_key_includes_z_range(
    client: TestClient, seeded_tairona_talbot
) -> None:
    slug, variant = seeded_tairona_talbot
    base = {
        "slug": slug,
        "variant": variant,
        "wavelength_um": 0.55,
        "z_min_um": 0.0,
        "n_slices": 8,
        "downsample": 8,
        "tile_size": 32,
    }
    r1 = client.post("/sim/carpet", json={**base, "z_max_um": 2000.0})
    r2 = client.post("/sim/carpet", json={**base, "z_max_um": 2000.0})
    r3 = client.post("/sim/carpet", json={**base, "z_max_um": 4000.0})
    assert r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 200
    # Same params → same atlas; second call cached.
    assert r1.json()["atlas_png"] == r2.json()["atlas_png"]
    assert r2.json()["cached"] is True
    # Different z_max_um → different atlas (hash must include z_max).
    assert r1.json()["atlas_png"] != r3.json()["atlas_png"]


def test_carpet_rejects_bad_z_range(
    client: TestClient, seeded_tairona_talbot
) -> None:
    slug, variant = seeded_tairona_talbot
    r = client.post(
        "/sim/carpet",
        json={
            "slug": slug,
            "variant": variant,
            "wavelength_um": 0.55,
            "z_min_um": 1000.0,
            "z_max_um": 1000.0,  # equal → must reject
            "n_slices": 8,
            "downsample": 8,
            "tile_size": 32,
        },
    )
    assert r.status_code == 400


def test_carpet_unknown_variant_404(client: TestClient) -> None:
    r = client.post(
        "/sim/carpet",
        json={
            "slug": "tairona-talbot",
            "variant": "nope0000ff",
            "wavelength_um": 0.55,
            "z_min_um": 0.0,
            "z_max_um": 2000.0,
            "n_slices": 8,
        },
    )
    assert r.status_code == 404
