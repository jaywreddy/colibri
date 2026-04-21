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


def test_carpet_stripe_layout_has_cols_gt_1(
    client: TestClient, seeded_tairona_talbot
) -> None:
    """Stripe layout is the canonical Talbot (x, z) carpet diagram: each row is
    a 1D centerline x-cut at one z, so the atlas itself is a 2D (x, z) image.
    rows == n_slices (z axis), cols == W (x axis, > 1), tile == [1, W].
    """
    slug, variant = seeded_tairona_talbot
    r = client.post(
        "/sim/carpet",
        json={
            "slug": slug,
            "variant": variant,
            "wavelength_um": 0.55,
            "z_min_um": 0.0,
            "z_max_um": 2000.0,
            "n_slices": 16,
            "downsample": 4,
            "layout": "stripe",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["layout"] == "stripe"
    assert body["rows"] == 16
    # x-width is set by downsampled grid; must be at least a handful of
    # pixels across (>> 1) so the shader can actually sample an x-line.
    assert body["cols"] > 8, body
    assert body["tile"] == [1, body["cols"]]


def test_carpet_stripe_and_tiles_differ(
    client: TestClient, seeded_tairona_talbot
) -> None:
    """Same slug/variant at stripe vs tiles must produce different atlases —
    the hash key includes layout so they don't collide in cache.
    """
    slug, variant = seeded_tairona_talbot
    base = {
        "slug": slug,
        "variant": variant,
        "wavelength_um": 0.55,
        "z_min_um": 0.0,
        "z_max_um": 2000.0,
        "n_slices": 8,
        "downsample": 4,
    }
    r_stripe = client.post("/sim/carpet", json={**base, "layout": "stripe"})
    r_tiles = client.post("/sim/carpet", json={**base, "layout": "tiles"})
    assert r_stripe.status_code == 200 and r_tiles.status_code == 200
    assert r_stripe.json()["atlas_png"] != r_tiles.json()["atlas_png"]
    assert r_stripe.json()["layout"] == "stripe"
    assert r_tiles.json()["layout"] == "tiles"


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


# ---------------------------------------------------------------------------
# Phase E — /sim/farfield contract. Used by the far_field_hologram recipe for
# colibri-hologram and meridian-speckle. Single RGB PNG whose channels are
# per-wavelength Fraunhofer reconstructions.
# ---------------------------------------------------------------------------
@pytest.fixture
def seeded_colibri_hologram(client: TestClient) -> tuple[str, str]:
    r = client.post(
        "/patterns/generate", json={"slug": "colibri-hologram", "params": {}}
    )
    assert r.status_code == 200, r.text
    m = r.json()
    return m["slug"], m["variant"]


def test_farfield_returns_rgb_reconstruction(
    client: TestClient, seeded_colibri_hologram
) -> None:
    slug, variant = seeded_colibri_hologram
    r = client.post(
        "/sim/farfield",
        json={
            "slug": slug,
            "variant": variant,
            "wavelengths_um": [0.65, 0.55, 0.45],
            "n_angles": 128,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["farfield_png"].endswith(".png")
    assert body["farfield_png"].startswith(f"/data/{slug}/{variant}/")
    assert isinstance(body["shape"], list) and len(body["shape"]) == 2
    # Second call must be cached.
    r2 = client.post(
        "/sim/farfield",
        json={
            "slug": slug,
            "variant": variant,
            "wavelengths_um": [0.65, 0.55, 0.45],
            "n_angles": 128,
        },
    )
    assert r2.json()["cached"] is True

    # Fetch the actual PNG bytes and check it's a 3-channel RGB with real signal
    # (not a uniform field). We read through the TestClient's /data mount.
    png_url = body["farfield_png"]
    png_r = client.get(png_url)
    assert png_r.status_code == 200
    from io import BytesIO

    from PIL import Image
    import numpy as np

    img = Image.open(BytesIO(png_r.content))
    assert img.mode == "RGB"
    arr = np.asarray(img)
    assert arr.shape[2] == 3
    # The colibri CGH is not uniform — its reconstruction has structure.
    assert float(arr.std()) > 2.0, "farfield PNG looks suspiciously flat"


def test_farfield_carrier_shifts_peak_off_center(
    client: TestClient, seeded_colibri_hologram
) -> None:
    """Phase G.2: the colibri CGH uses an off-axis carrier (carrier_cells=4)
    so the reconstruction's first-order replica lands cleanly in the
    upper-right quadrant, separated from the DC spike at center and the
    Hermitian conjugate in the lower-left. The farfield kernel responds
    by cropping the upper-right quadrant rather than the center — so the
    brightest pixel of the output must NOT be near (H/2, W/2).
    """
    slug, variant = seeded_colibri_hologram
    r = client.post(
        "/sim/farfield",
        json={
            "slug": slug,
            "variant": variant,
            "wavelengths_um": [0.65, 0.55, 0.45],
            "n_angles": 128,
            "carrier_cells": 4,
        },
    )
    assert r.status_code == 200, r.text
    png_url = r.json()["farfield_png"]
    png_r = client.get(png_url)
    assert png_r.status_code == 200

    from io import BytesIO

    import numpy as np
    from PIL import Image

    img = Image.open(BytesIO(png_r.content))
    arr = np.asarray(img).astype(np.float32)
    # Sum RGB for an overall-brightness map; the carrier shift drops the
    # target off-axis, so the non-edge peak should not be at the image center.
    lum = arr.sum(axis=2)
    h, w = lum.shape
    # Clear the DC spike (center pixels tend to dominate due to zero-order
    # leakage in binary amplitude CGHs). We mask out the central N/16 square
    # and look for the brightest remaining pixel.
    mask = np.ones_like(lum, dtype=bool)
    cy, cx = h // 2, w // 2
    dh, dw = max(1, h // 16), max(1, w // 16)
    mask[cy - dh : cy + dh, cx - dw : cx + dw] = False
    lum_masked = np.where(mask, lum, 0.0)
    py, px = np.unravel_index(int(np.argmax(lum_masked)), lum.shape)
    # The carrier-shifted replica must land in a different quadrant from the
    # exact center — at minimum, the peak should be > 10% of the image
    # dimension away from center in at least one axis.
    assert abs(py - cy) > h * 0.1 or abs(px - cx) > w * 0.1, (
        f"peak at ({py}, {px}) is too close to center ({cy}, {cx}) — "
        "carrier shift did not translate to an off-center crop"
    )


def test_farfield_carrier_cells_cache_key(
    client: TestClient, seeded_colibri_hologram
) -> None:
    """Different carrier_cells → different cached atlas. Same carrier_cells
    → second call is cached.
    """
    slug, variant = seeded_colibri_hologram
    base = {
        "slug": slug,
        "variant": variant,
        "wavelengths_um": [0.65, 0.55, 0.45],
        "n_angles": 128,
    }
    r0 = client.post("/sim/farfield", json={**base, "carrier_cells": 0})
    r0b = client.post("/sim/farfield", json={**base, "carrier_cells": 0})
    r4 = client.post("/sim/farfield", json={**base, "carrier_cells": 4})
    assert r0.status_code == 200 and r0b.status_code == 200 and r4.status_code == 200
    assert r0.json()["farfield_png"] == r0b.json()["farfield_png"]
    assert r0b.json()["cached"] is True
    assert r0.json()["farfield_png"] != r4.json()["farfield_png"]


def test_farfield_rejects_wrong_channel_count(
    client: TestClient, seeded_colibri_hologram
) -> None:
    slug, variant = seeded_colibri_hologram
    r = client.post(
        "/sim/farfield",
        json={
            "slug": slug,
            "variant": variant,
            "wavelengths_um": [0.55],  # single channel not allowed
        },
    )
    assert r.status_code == 400


def test_farfield_unknown_variant_404(client: TestClient) -> None:
    r = client.post(
        "/sim/farfield",
        json={"slug": "colibri-hologram", "variant": "nope0000ff"},
    )
    assert r.status_code == 404
