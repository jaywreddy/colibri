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


# ---------------------------------------------------------------------------
# Phase A — render_recipe manifest round-trip.
# ---------------------------------------------------------------------------
# Each pattern must stamp its class-level render_recipe onto every manifest
# it materializes, and the recipe must be one of the six known names. A
# regression here (e.g. a pattern forgetting to override render_recipe on a
# class rename, or service.py dropping the field during migration) would
# immediately surface as flat-gold rendering in the frontend with no error.

_VALID_RECIPES = {
    "iridescent_grating",
    "stereo_lenticular",
    "moire_interactive",
    "near_field_carpet",
    "far_field_hologram",
    "stylized_amplitude",
}


def test_manifest_carries_render_recipe_field(client: TestClient) -> None:
    r = client.post("/patterns/generate", json={"slug": "wayuu-kanasu-moire", "params": {}})
    assert r.status_code == 200
    m = r.json()
    assert "render_recipe" in m, "manifest missing render_recipe"
    assert m["render_recipe"] in _VALID_RECIPES
    assert "recipe_data" in m and isinstance(m["recipe_data"], dict)


def test_descriptor_advertises_render_recipe_for_every_pattern(client: TestClient) -> None:
    # /patterns returns the descriptors, which should pre-advertise the
    # recipe so the frontend can decide layout before fetching a manifest.
    r = client.get("/patterns")
    assert r.status_code == 200
    descriptors = r.json()
    assert descriptors, "no descriptors returned"
    for d in descriptors:
        assert d.get("render_recipe") in _VALID_RECIPES, (
            f"{d['slug']} advertises unknown recipe {d.get('render_recipe')!r}"
        )


def test_recipe_data_for_phase_b_c_patterns(client: TestClient) -> None:
    # iridescent_grating patterns must supply period_um so the shader can
    # compute dispersion; stereo_lenticular patterns must supply view_a/view_b
    # PNG urls + slit axis. Catches generators that forget to populate
    # recipe_data after being upgraded to a physics-correct recipe.
    r = client.post("/patterns/generate", json={"slug": "cafetero-iridescence", "params": {}})
    assert r.status_code == 200
    caf = r.json()
    assert caf["render_recipe"] == "iridescent_grating"
    assert "period_um" in caf["recipe_data"], "cafetero missing period_um"

    r = client.post("/patterns/generate", json={"slug": "sombrero-vueltiao-parallax", "params": {}})
    assert r.status_code == 200
    som = r.json()
    assert som["render_recipe"] == "stereo_lenticular"
    assert "view_a_png" in som["recipe_data"]
    assert "view_b_png" in som["recipe_data"]
    assert "slit_axis_deg" in som["recipe_data"]


def test_invalid_param_type_returns_error(client: TestClient) -> None:
    # period_um is a float in wayuu-kanasu-moire. A string should either 422 at pydantic
    # or 400 at the generator. Either is acceptable — just not 200.
    r = client.post(
        "/patterns/generate",
        json={"slug": "wayuu-kanasu-moire", "params": {"period_um": "not-a-number"}},
    )
    assert r.status_code in (400, 422)
