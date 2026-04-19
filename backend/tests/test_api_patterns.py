"""Backend API contract for /patterns — shape, idempotence, error codes."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


EXPECTED_SLUGS = {
    "wayuu-kanasu-moire",
    "emerald-facet-moire",
    "sombrero-vueltiao-parallax",
    "cafetero-iridescence",
    "colibri-hologram",
    "muzo-emerald-zone",
    "tairona-talbot",
    "caravel-latent",
    "compass-rose-spiral",
    "meridian-speckle",
}


def test_list_patterns_returns_all_registered_slugs(client: TestClient) -> None:
    r = client.get("/patterns")
    assert r.status_code == 200
    got = {entry["slug"] for entry in r.json()}
    assert got == EXPECTED_SLUGS


def test_get_manifest_roundtrips_after_generate(client: TestClient) -> None:
    r = client.post("/patterns/generate", json={"slug": "wayuu-kanasu-moire", "params": {}})
    assert r.status_code == 200
    m = r.json()
    assert m["slug"] == "wayuu-kanasu-moire"
    assert m["pixel_pitch_um"] > 0
    assert len(m["extent_um"]) == 2
    assert "params" in m and isinstance(m["params"], dict)
    assert "substrate" in m and m["substrate"]["thickness_um"] > 0


def test_generate_is_idempotent_on_same_params(client: TestClient) -> None:
    r1 = client.post("/patterns/generate", json={"slug": "wayuu-kanasu-moire", "params": {}})
    r2 = client.post("/patterns/generate", json={"slug": "wayuu-kanasu-moire", "params": {}})
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["variant"] == r2.json()["variant"]


def test_generate_writes_front_back_svg_png_and_thumbnail(
    client: TestClient, isolated_data_root: Path
) -> None:
    r = client.post("/patterns/generate", json={"slug": "wayuu-kanasu-moire", "params": {}})
    m = r.json()
    variant_dir = isolated_data_root / m["slug"] / m["variant"]
    for name in ("front.png", "back.png", "front.svg", "back.svg", "thumbnail.png", "manifest.json"):
        assert (variant_dir / name).exists(), f"missing {name}"


def test_unknown_slug_returns_404(client: TestClient) -> None:
    r = client.get("/patterns/does-not-exist")
    assert r.status_code == 404


def test_invalid_param_type_returns_error(client: TestClient) -> None:
    # period_um is a float in wayuu-kanasu-moire. A string should either 422 at pydantic
    # or 400 at the generator. Either is acceptable — just not 200.
    r = client.post(
        "/patterns/generate",
        json={"slug": "wayuu-kanasu-moire", "params": {"period_um": "not-a-number"}},
    )
    assert r.status_code in (400, 422)
