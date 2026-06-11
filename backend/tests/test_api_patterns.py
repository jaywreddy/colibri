"""Backend API contract for /patterns — shape, idempotence, error codes."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


EXPECTED_SLUGS = {
    "wayuu-kanasu-moire",
    "emerald-facet-moire",
    "colibri-globe-lenticular",
    "colibri-globe-moire",
    "colibri-globe-phase",
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


def test_generate_writes_pngs_and_thumbnail_with_lazy_svg(
    client: TestClient, isolated_data_root: Path
) -> None:
    r = client.post("/patterns/generate", json={"slug": "wayuu-kanasu-moire", "params": {}})
    m = r.json()
    variant_dir = isolated_data_root / m["slug"] / m["variant"]
    for name in ("front.png", "back.png", "thumbnail.png", "manifest.json"):
        assert (variant_dir / name).exists(), f"missing {name}"
    # SVG is lazy (mirrors the plate manifest contract): empty URL fields and
    # no files on disk until the on-demand endpoint is hit.
    assert m["files"]["front_svg"] == ""
    assert m["files"]["back_svg"] == ""
    assert not (variant_dir / "front.svg").exists()
    assert not (variant_dir / "back.svg").exists()

    r2 = client.get(f"/patterns/{m['slug']}/{m['variant']}/svg")
    assert r2.status_code == 200, r2.text
    m2 = r2.json()
    assert m2["files"]["front_svg"] == f"/data/{m['slug']}/{m['variant']}/front.svg"
    assert m2["files"]["back_svg"] == f"/data/{m['slug']}/{m['variant']}/back.svg"
    assert (variant_dir / "front.svg").exists()
    assert (variant_dir / "back.svg").exists()


def test_svg_endpoint_404s_for_unknown_variant(client: TestClient) -> None:
    r = client.get("/patterns/wayuu-kanasu-moire/no-such-variant/svg")
    assert r.status_code == 404


def test_unknown_slug_returns_404(client: TestClient) -> None:
    r = client.get("/patterns/does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Render-recipe manifest round-trip — every manifest must stamp its class
# render_recipe and the recipe must be one of the three names the shader
# actively switches on.
# ---------------------------------------------------------------------------

_VALID_RECIPES = {
    "stereo_lenticular",
    "moire_interactive",
    "phase_shift_overlay",
}


def test_manifest_carries_render_recipe_field(client: TestClient) -> None:
    r = client.post("/patterns/generate", json={"slug": "wayuu-kanasu-moire", "params": {}})
    assert r.status_code == 200
    m = r.json()
    assert "render_recipe" in m, "manifest missing render_recipe"
    assert m["render_recipe"] in _VALID_RECIPES
    assert "recipe_data" in m and isinstance(m["recipe_data"], dict)


def test_descriptor_advertises_render_recipe_for_every_pattern(client: TestClient) -> None:
    r = client.get("/patterns")
    assert r.status_code == 200
    descriptors = r.json()
    assert descriptors, "no descriptors returned"
    for d in descriptors:
        assert d.get("render_recipe") in _VALID_RECIPES, (
            f"{d['slug']} advertises unknown recipe {d.get('render_recipe')!r}"
        )


def test_lenticular_manifest_ships_view_a_view_b_urls(client: TestClient) -> None:
    """stereo_lenticular patterns must publish view_a / view_b PNG URLs in
    recipe_data so the PlateScene can fetch the interlaced scenes."""
    r = client.post(
        "/patterns/generate",
        json={"slug": "colibri-globe-lenticular", "params": {}},
    )
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["render_recipe"] == "stereo_lenticular"
    rd = m["recipe_data"]
    assert "view_a_png" in rd and "view_b_png" in rd
    assert "slit_period_um" in rd


def test_phase_overlay_manifest_ships_carrier_period(client: TestClient) -> None:
    r = client.post(
        "/patterns/generate",
        json={"slug": "colibri-globe-phase", "params": {}},
    )
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["render_recipe"] == "phase_shift_overlay"
    assert "carrier_period_um" in m["recipe_data"]


def test_invalid_param_type_returns_error(client: TestClient) -> None:
    r = client.post(
        "/patterns/generate",
        json={"slug": "wayuu-kanasu-moire", "params": {"period_um": "not-a-number"}},
    )
    assert r.status_code in (400, 422)
