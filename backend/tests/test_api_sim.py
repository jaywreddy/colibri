"""Backend API contract for /sim — atlas shape, caching, and error codes."""
from __future__ import annotations

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
