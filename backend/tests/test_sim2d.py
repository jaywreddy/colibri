"""2D parallax lab (app.sim2d) — synthetic masks only, milliseconds-fast.

Every mask here is built directly with numpy/PIL (tiny 64x64 gratings and
dots); NO pattern generation or materialization happens in this file. The
endpoint tests write their own front/back/manifest into the isolated data
root so `_load_parallax_pair` finds them without touching the registry.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.sim2d import (
    DEFAULT_LASER,
    GOLD,
    composite_parallax,
    contrast_curve,
    parallax_shift_um,
    transmission_contrast,
)

SIZE = 64


def _grating(period_px: int, size: int = SIZE) -> Image.Image:
    """Vertical stripe grating: gold where (x mod period) < period/2."""
    xs = np.arange(size)
    row = ((xs % period_px) < period_px / 2).astype(np.uint8) * 255
    return Image.fromarray(np.tile(row, (size, 1)), mode="L")


def _flat(value: int, size: int = SIZE) -> Image.Image:
    return Image.fromarray(np.full((size, size), value, dtype=np.uint8), mode="L")


# ---------------------------------------------------------------------------
# parallax_shift_um — the shared formula (parallax.glsl)
# ---------------------------------------------------------------------------


def test_parallax_shift_zero_tilt_is_zero() -> None:
    assert parallax_shift_um(0.0, 0.0, 500.0, 1.46) == (0.0, 0.0)


def test_parallax_shift_matches_hand_computed() -> None:
    # 20 deg, t=500 um, n=1.46:
    #   sinSub = sin(20)/1.46, shift = 500 * sinSub / sqrt(1 - sinSub^2)
    sin_sub = math.sin(math.radians(20.0)) / 1.46
    expected = 500.0 * sin_sub / math.sqrt(1.0 - sin_sub**2)
    dx, dy = parallax_shift_um(20.0, 0.0, 500.0, 1.46)
    assert dx == pytest.approx(expected, rel=1e-9)
    assert dx == pytest.approx(120.48, abs=0.01)  # sanity anchor
    assert dy == 0.0


def test_parallax_shift_scales_linearly_with_thickness() -> None:
    dx_500, _ = parallax_shift_um(15.0, 0.0, 500.0, 1.46)
    dx_1000, _ = parallax_shift_um(15.0, 0.0, 1000.0, 1.46)
    assert dx_1000 == pytest.approx(2.0 * dx_500, rel=1e-12)


def test_parallax_shift_negative_tilt_flips_sign() -> None:
    dx_pos, _ = parallax_shift_um(20.0, 0.0, 500.0, 1.46)
    dx_neg, _ = parallax_shift_um(-20.0, 0.0, 500.0, 1.46)
    assert dx_neg == pytest.approx(-dx_pos, rel=1e-12)


# ---------------------------------------------------------------------------
# compositing rules (plate.frag simplification)
# ---------------------------------------------------------------------------


def test_all_gold_front_blocks_all_transmission() -> None:
    t = transmission_contrast(_flat(255), _grating(8), 3.0, 0.0, 1.0)
    assert t == 0.0


def test_blank_pair_backlight_is_white() -> None:
    img = composite_parallax(_flat(0), _flat(0), 0.0, 0.0, 1.0, illum="backlight")
    arr = np.asarray(img)
    assert arr.shape == (SIZE, SIZE, 3)
    assert (arr == 255).all()


def test_laser_tints_transmission_with_laser_color() -> None:
    img = composite_parallax(_flat(0), _flat(0), 0.0, 0.0, 1.0, illum="laser")
    arr = np.asarray(img)
    expected = tuple(round(c * 255) for c in DEFAULT_LASER)  # (68, 255, 136)
    assert tuple(arr[SIZE // 2, SIZE // 2]) == expected
    # green channel dominates — it's a green laser
    assert arr[..., 1].min() > arr[..., 0].max()


def test_ambient_overlap_darkening_is_the_back_layer_dependence() -> None:
    """front=1 ambient pixels must still track the back layer (plate.frag).

    Paired pin with frontend/tests/unit/composite2d.test.ts ("ambient: f=1
    pixels still track the back layer"). Hand-computed from plate.frag's
    ambient formula at front = 1:
        reflected = 1, transmission = 0, overlap = back
        rgb = GOLD * 0.85 * (1 - 0.35 * back)
        back=0 -> (195.5, 159.7, 68.1)     back=1 -> (127.1, 103.8, 44.2)
    Without the overlap term both states collapse to the same constant and
    every ambient metric over a front-gold figure becomes tilt-blind.
    """
    clear = np.asarray(
        composite_parallax(_flat(255), _flat(0), 0.0, 0.0, 1.0, illum="ambient")
    )
    covered = np.asarray(
        composite_parallax(_flat(255), _flat(255), 0.0, 0.0, 1.0, illum="ambient")
    )
    mid = SIZE // 2
    for channel, gold in enumerate(GOLD):
        assert clear[mid, mid, channel] == pytest.approx(gold * 0.85 * 255, abs=1)
        assert covered[mid, mid, channel] == pytest.approx(
            gold * 0.85 * 0.65 * 255, abs=1
        )
    # The 35% swing itself, not just the endpoints.
    assert covered[mid, mid, 0] / clear[mid, mid, 0] == pytest.approx(0.65, abs=0.01)


def test_ambient_clear_pair_floor_is_the_shader_transmission_term() -> None:
    """Blank pair: reflected=0, transmission=1, overlap=0 -> 0.04 * 255 = 10.2.

    0.04 is plate.frag's constant; both 2D paths used to carry 0.06 (= 15.3).
    Pinned identically in frontend/tests/unit/composite2d.test.ts.
    """
    arr = np.asarray(
        composite_parallax(_flat(0), _flat(0), 0.0, 0.0, 1.0, illum="ambient")
    )
    assert tuple(arr[SIZE // 2, SIZE // 2]) == (10, 10, 10)


def test_unknown_illum_raises() -> None:
    with pytest.raises(ValueError):
        composite_parallax(_flat(0), _flat(0), 0.0, 0.0, 1.0, illum="candlelight")


# ---------------------------------------------------------------------------
# physics: moiré beat vs. degenerate pair
# ---------------------------------------------------------------------------


def test_beat_gratings_modulate_transmission_with_tilt() -> None:
    """p1=8 px vs p2=9 px gratings beat at 72 px — larger than the 64 px frame,
    so sliding the back layer sweeps the bright fringe through the window and
    the mean transmission visibly modulates."""
    rows = contrast_curve(
        _grating(8),
        _grating(9),
        tilts_deg=list(np.linspace(-30.0, 30.0, 21)),
        thickness_um=60.0,
        n=1.46,
        pixel_pitch_um=1.0,
        axis="x",
    )
    values = [r["transmission"] for r in rows]
    assert max(values) - min(values) > 0.05
    # rows carry the swept-axis shift, monotone in tilt
    dxs = [r["dx_um"] for r in rows]
    assert dxs == sorted(dxs)


def test_identical_gratings_whole_period_shifts_are_flat() -> None:
    """Identical p=8 gratings shifted by whole periods realign exactly; the
    zero-filled entry strip also averages 0.5, so the mean is constant."""
    front = _grating(8)
    back = _grating(8)
    values = [
        transmission_contrast(front, back, dx_um, 0.0, 1.0)
        for dx_um in (0.0, 8.0, 16.0, 24.0)
    ]
    assert max(values) - min(values) < 1e-6


# ---------------------------------------------------------------------------
# shift convention — must match plate.frag's `texture2D(uBack, vUv - shift)`
# ---------------------------------------------------------------------------


def test_back_dot_moves_plus_x_under_positive_x_tilt() -> None:
    """Sampling the back at (uv - shift) moves its content BY +shift: a gold
    dot at the center must appear dx_px to the right for +x tilt."""
    back = np.zeros((SIZE, SIZE), dtype=np.uint8)
    back[32, 32] = 255
    dx_um, dy_um = parallax_shift_um(20.0, 0.0, 500.0, 1.46)  # ~120.48 um
    pitch = 10.0
    dx_px = int(round(dx_um / pitch))  # 12 px
    assert dx_px == 12

    img = composite_parallax(
        _flat(0), Image.fromarray(back, mode="L"), dx_um, dy_um, pitch,
        illum="backlight",
    )
    arr = np.asarray(img)
    # dot lands at (32, 32 + 12): transmission blocked -> far from white
    assert arr[32, 32 + dx_px, 0] < 100
    # original location is now clear glass -> pure backlight white
    assert tuple(arr[32, 32]) == (255, 255, 255)


# ---------------------------------------------------------------------------
# endpoints — synthetic variant written straight into the isolated data root
# ---------------------------------------------------------------------------


def _write_variant(data_root: Path, slug: str, variant: str) -> None:
    d = data_root / slug / variant
    d.mkdir(parents=True)
    _grating(8).save(d / "front.png")
    _grating(9).save(d / "back.png")
    manifest = {
        "slug": slug,
        "variant": variant,
        "pixel_pitch_um": 1.0,
        "extent_um": [64.0, 64.0],
        "substrate": {"thickness_um": 60.0, "material": "quartz", "n": 1.46},
    }
    (d / "manifest.json").write_text(json.dumps(manifest))


def test_parallax2d_endpoint_returns_png(client, isolated_data_root: Path) -> None:
    _write_variant(isolated_data_root, "synthetic-beat", "v1")
    r = client.get(
        "/sim/parallax2d/synthetic-beat/v1",
        params={"tilt_x_deg": 10.0, "illum": "backlight"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_parallax2d_curve_endpoint_shape(client, isolated_data_root: Path) -> None:
    _write_variant(isolated_data_root, "synthetic-beat", "v1")
    r = client.get(
        "/sim/parallax2d/synthetic-beat/v1/curve", params={"axis": "x", "points": 11}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slug"] == "synthetic-beat"
    assert body["variant"] == "v1"
    assert body["pixel_pitch_um"] == 1.0
    assert body["thickness_um"] == 60.0
    curve = body["curve"]
    assert len(curve) == 11
    assert curve[0]["tilt_deg"] == -30.0
    assert curve[-1]["tilt_deg"] == 30.0
    assert set(curve[0]) == {"tilt_deg", "dx_um", "transmission"}
    # the beat pair from _write_variant must show tilt modulation end-to-end
    values = [row["transmission"] for row in curve]
    assert max(values) - min(values) > 0.05


def test_parallax2d_unknown_variant_404s_without_generating(
    client, isolated_data_root: Path
) -> None:
    # Non-'default' variants must never trigger materialization.
    r = client.get("/sim/parallax2d/wayuu-kanasu-moire/deadbeef00")
    assert r.status_code == 404
    assert not (isolated_data_root / "wayuu-kanasu-moire").exists()


def test_parallax2d_bad_illum_400s(client, isolated_data_root: Path) -> None:
    _write_variant(isolated_data_root, "synthetic-beat", "v1")
    r = client.get(
        "/sim/parallax2d/synthetic-beat/v1", params={"illum": "candlelight"}
    )
    assert r.status_code == 400
