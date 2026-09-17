"""Backend API contract for /patterns — shape, idempotence, error codes."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


# The catalogue IS the box (2026-09-16). Four written production faces, bare
# glass, and ONE hidden exemplar:
# - "monogram-jp" (lid) and "globe-atlantic" (front) are SINGLE-LAYER
#   DIFFRACTION mappings: each region is a fine vertical grating whose PERIOD is
#   its colour. foliage_moire, like the photo faces — a composed box face with
#   no second plane to interact with.
# - "photo-halftone" is the three sides' gold line screen, "solid-gold" the
#   base plate, "blank" the bare-quartz back.
# - "globe-duo-phase" carries no face. It is the PARALLAX BARRIER exemplar,
#   kept registered so the two-image switch construction stays built and
#   measured (tests/test_barrier_registration.py) after the box went single-ply.
EXPECTED_SLUGS = {
    "monogram-jp",
    "globe-atlantic",
    "photo-halftone",
    "solid-gold",
    "blank",
    "globe-duo-phase",
}


def test_list_patterns_returns_all_registered_slugs(client: TestClient) -> None:
    r = client.get("/patterns")
    assert r.status_code == 200
    got = {entry["slug"] for entry in r.json()}
    assert got == EXPECTED_SLUGS


# The generate-shape tests below assert MANIFEST STRUCTURE, never the
# monogram's default geometry, so they run on the `cheap_pattern` fixture
# (smallest legal extent). On a default variant each of them would pay its own
# cold generate, because `client` isolates DATA_ROOT per test and nothing is
# shared between them.
def test_get_manifest_roundtrips_after_generate(
    client: TestClient, cheap_pattern: tuple[str, dict[str, float]]
) -> None:
    slug, params = cheap_pattern
    r = client.post("/patterns/generate", json={"slug": slug, "params": params})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["slug"] == slug
    assert m["pixel_pitch_um"] > 0
    assert len(m["extent_um"]) == 2
    assert "params" in m and isinstance(m["params"], dict)
    assert "substrate" in m and m["substrate"]["thickness_um"] > 0


def test_generate_is_idempotent_on_same_params(
    client: TestClient, cheap_pattern: tuple[str, dict[str, float]]
) -> None:
    slug, params = cheap_pattern
    body = {"slug": slug, "params": params}
    r1 = client.post("/patterns/generate", json=body)
    r2 = client.post("/patterns/generate", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["variant"] == r2.json()["variant"]


def test_generate_writes_pngs_and_thumbnail_with_lazy_svg(
    client: TestClient,
    isolated_data_root: Path,
    cheap_pattern: tuple[str, dict[str, float]],
) -> None:
    slug, params = cheap_pattern
    r = client.post("/patterns/generate", json={"slug": slug, "params": params})
    assert r.status_code == 200, r.text
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
    r = client.get("/patterns/monogram-jp/no-such-variant/svg")
    assert r.status_code == 404


def test_unknown_slug_returns_404(client: TestClient) -> None:
    r = client.get("/patterns/does-not-exist")
    assert r.status_code == 404


def test_thumbnail_endpoint_is_cache_only_and_keyed_on_the_default_variant(
    client: TestClient,
    isolated_data_root: Path,
    cheap_pattern: tuple[str, dict[str, float]],
) -> None:
    """The picker sweep's route: serve the DEFAULT variant's cached PNG or 404.

    Two contracts, both load-bearing for the picker (store.ts::loadThumbnails
    probes this once per catalog slug on every open):
      1. it never generates — a cold slug 404s and leaves no cache dir, which
         is what keeps opening the picker off the heavy-compute path;
      2. it is keyed on the DEFAULT variant — a materialized NON-default
         variant must not satisfy it, or the tile would show a preview of
         parameters nobody picked.
    """
    from app.patterns.base import registry
    from app.service import pattern_dir, variant_key

    slug, params = cheap_pattern
    r = client.get(f"/patterns/{slug}/thumbnail")
    assert r.status_code == 404, r.text
    assert not (isolated_data_root / slug).exists(), "cache-only route generated"

    # A non-default variant on disk must not be served as the default preview.
    m = client.post("/patterns/generate", json={"slug": slug, "params": params}).json()
    default_variant = variant_key(registry[slug].defaults())
    assert m["variant"] != default_variant, "cheap params collided with the defaults"
    assert client.get(f"/patterns/{slug}/thumbnail").status_code == 404

    # With the default slot warm, the PNG bytes come back verbatim.
    payload = b"\x89PNG\r\n\x1a\nstub"
    d = pattern_dir(slug, default_variant)
    d.mkdir(parents=True, exist_ok=True)
    (d / "thumbnail.png").write_bytes(payload)
    r2 = client.get(f"/patterns/{slug}/thumbnail")
    assert r2.status_code == 200, r2.text
    assert r2.headers["content-type"] == "image/png"
    assert r2.content == payload


def test_thumbnail_endpoint_404s_for_unknown_slug(client: TestClient) -> None:
    r = client.get("/patterns/does-not-exist/thumbnail")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Render-recipe manifest round-trip — every manifest must stamp its class
# render_recipe and the recipe must be one of the names the shader actively
# switches on for PATTERN manifests (ids 0 stereo_lenticular and
# 1 moire_interactive; id 3 foliage_moire is bound only by composed PLATE
# manifests, never by a pattern class).
# ---------------------------------------------------------------------------

# NOTE: phase_shift_overlay (recipe 2) is RETIRED with zero users — its
# "switch" was a shader view-sign bias, not physics. The name is gone from
# base.RECIPE_NAMES and the frontend deleted shader id 2. Numeric ids 0/1/3 are
# stable with a permanent hole at 2. moire_interactive (1) has no pattern user
# left either since the catalogue became the box, but it stays a legal name:
# the exemplar's sibling constructions are one commit away, and a recipe the
# shader still switches on is not a recipe to delete from the contract.
_VALID_RECIPES = {
    "stereo_lenticular",
    "moire_interactive",
    # the composed-plate recipe; the photo and blank faces bind it directly
    # because they exist to be composed into a box (CLAUDE.md renderer honesty)
    "foliage_moire",
}


def test_manifest_carries_render_recipe_field(
    client: TestClient, cheap_pattern: tuple[str, dict[str, float]]
) -> None:
    slug, params = cheap_pattern
    r = client.post("/patterns/generate", json={"slug": slug, "params": params})
    assert r.status_code == 200, r.text
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


def test_the_barrier_exemplar_ships_view_a_view_b_urls(client: TestClient) -> None:
    """``globe-duo-phase`` is the one stereo_lenticular pattern left. A barrier
    publishes view_a / view_b PNG URLs in recipe_data (the two interlaced back
    channels) plus the slit period, so a consumer can draw the same lattice
    instead of guessing one."""
    r = client.post(
        "/patterns/generate",
        json={"slug": "globe-duo-phase", "params": {"extent_um": 600.0}},
    )
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["render_recipe"] == "stereo_lenticular"
    rd = m["recipe_data"]
    assert "view_a_png" in rd and "view_b_png" in rd
    assert "slit_period_um" in rd


def test_invalid_param_type_returns_error(client: TestClient) -> None:
    r = client.post(
        "/patterns/generate",
        json={"slug": "monogram-jp", "params": {"overlap": "not-a-number"}},
    )
    assert r.status_code in (400, 422)
