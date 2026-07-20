"""Backend API contract for /patterns — shape, idempotence, error codes."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


# Catalog contract after the 2026-07 parallax-honesty rebuilds:
# - "jp-monogram-phase", "globe-duo-phase", "gear-quill-switch" and
#   "colibri-flap-phase" are parallax barriers (both images interlaced in
#   BACK, pure slit comb in FRONT, stereo_lenticular) — the two-image
#   front/back phase split they once shipped was structurally incapable of
#   switching under honest parallax.
# - "colibri-globe-phase" REMOVED — after its barrier rebuild it was
#   architecturally identical to colibri-globe-lenticular (only a slower
#   default period), so the redundant twin was dropped.
# - "monogram-carrier-reveal" — the honest T5 carrier reveal (single image
#   halftoned onto a carrier in FRONT, exact anti-phase carrier in BACK,
#   moire_interactive).
# - "food-pair-chirp", "jamon-tray", "inscription-line", "monogram-jp" are
#   single-layer front-only shimmers (moire_interactive; empty back).
EXPECTED_SLUGS = {
    # Original catalog
    "wayuu-kanasu-moire",
    "emerald-facet-moire",
    "colibri-globe-lenticular",
    "colibri-globe-moire",
    # Taxonomy rebuild additions (barrier switches, carrier reveal, bitmap)
    "globe-rotation-stereo",
    "orchid-shimmer-moire",
    "jp-monogram-phase",
    "monogram-carrier-reveal",
    "bitmap-halftone",
    # Engagement-box showpieces (six-face plan + experiments)
    "capybara-scanimation",
    "colibri-flap-phase",
    "food-pair-chirp",
    "gear-quill-switch",
    "globe-duo-phase",
    "inscription-line",
    "jamon-tray",
    "monogram-jp",
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
# render_recipe and the recipe must be one of the names the shader actively
# switches on for PATTERN manifests (ids 0 stereo_lenticular and
# 1 moire_interactive; id 3 foliage_moire is bound only by composed PLATE
# manifests, never by a pattern class).
# ---------------------------------------------------------------------------

# NOTE: phase_shift_overlay (recipe 2) is RETIRED with zero users — its
# "switch" was a shader view-sign bias, not physics. The barrier rebuilds
# (jp-monogram-phase, globe-duo-phase, gear-quill-switch, colibri-flap-phase)
# and the single-layer retags (food-pair-chirp, jamon-tray, inscription-line,
# monogram-jp → moire_interactive) removed every user; the name is deleted
# from base.RECIPE_NAMES and the frontend deleted shader id 2. Numeric ids
# 0/1/3 are stable with a permanent hole at 2.
_VALID_RECIPES = {
    "stereo_lenticular",
    "moire_interactive",
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


def test_monogram_barrier_manifest_ships_view_urls(client: TestClient) -> None:
    """jp-monogram-phase was rebuilt from the broken two-image phase overlay
    into a parallax barrier (both images interlaced in the BACK layer, slit
    mask in FRONT — the only construction that switches under honest
    parallax). It must advertise stereo_lenticular and ship the interlaced
    view PNGs + slit period like the other barrier patterns."""
    r = client.post(
        "/patterns/generate",
        json={"slug": "jp-monogram-phase", "params": {}},
    )
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["render_recipe"] == "stereo_lenticular"
    rd = m["recipe_data"]
    assert "view_a_png" in rd and "view_b_png" in rd
    assert "slit_period_um" in rd


def test_carrier_reveal_manifest_ships_carrier_period(client: TestClient) -> None:
    """The honest carrier reveal (image-on-carrier FRONT, uniform image-free
    carrier BACK) renders under plain moire_interactive mask sampling and
    must publish carrier_period_um so the UI/tests can calibrate first-zone
    tilts: the reveal completes at a back shift of p/2 and aliases with
    period p."""
    r = client.post(
        "/patterns/generate",
        json={"slug": "monogram-carrier-reveal", "params": {}},
    )
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["render_recipe"] == "moire_interactive"
    assert "carrier_period_um" in m["recipe_data"]


def test_invalid_param_type_returns_error(client: TestClient) -> None:
    r = client.post(
        "/patterns/generate",
        json={"slug": "wayuu-kanasu-moire", "params": {"period_um": "not-a-number"}},
    )
    assert r.status_code in (400, 422)
