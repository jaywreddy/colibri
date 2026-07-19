"""Bitmap halftone pattern: registration, demo assets, tonal fidelity (darker
image regions get strictly more gold), the four back modes, the lattice-budget
guard, and tone inversion.

All generation tests use a tiny 32x32 synthetic black/gray/white card in
tmp_path (ASSETS_DIR monkeypatched) so they never depend on the demo files'
exact content — but the demo files' existence/loadability IS asserted, since
the descriptor's image choices and the pattern defaults rely on them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import app.patterns  # noqa: F401  -- populate registry (pulls in the bitmap pkg)
from app.patterns.base import registry
from app.patterns.bitmap import halftone
from app.rasterize import rasterize
from app.sim2d import switch_metrics

# Small extent keeps the working grid at 200x200 cells for the default
# 20 um period — milliseconds per generate, safely inside the memory rules.
EXTENT = 500.0

DEMO_FILES = ["andes-dawn.png", "capybara.png"]


@pytest.fixture
def halfcard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """32x32 test card written to tmp_path, which is monkeypatched in as the
    bitmap assets dir. Left third black, middle third mid-gray, right third
    white: the dark half (left of center) carries far more tone than the
    light half, and the mid-gray band keeps a half-duty region alive through
    autocontrast (a pure two-tone card would saturate to 0/255 and hide the
    phase_reveal shift, since fully-dark stripes are gold at EVERY phase)."""
    arr = np.full((32, 32), 255, dtype=np.uint8)
    arr[:, :11] = 0
    arr[:, 11:21] = 128
    name = "halfcard.png"
    Image.fromarray(arr, mode="L").save(tmp_path / name)
    monkeypatch.setattr(halftone, "ASSETS_DIR", tmp_path)
    return name


def _half_areas(mp) -> tuple[float, float]:
    """(left, right) gold area split at x=0 by rectangle centroid. Halftone
    runs stop at the card's tone boundary, so rects sit fully in one half."""
    left = sum(g.area for g in mp.geoms if g.centroid.x < 0)
    right = sum(g.area for g in mp.geoms if g.centroid.x >= 0)
    return left, right


def test_registered_with_image_choices():
    assert "bitmap-halftone" in registry
    d = registry["bitmap-halftone"].descriptor()
    assert d["render_recipe"] == "moire_interactive"
    assert d["theme"] == "Global Travel"
    image_param = next(p for p in d["params"] if p["name"] == "image")
    assert image_param["type"] == "choice"
    assert image_param["choices"], "image choices empty — demo bitmaps missing?"
    assert image_param["default"] in image_param["choices"]


def test_demo_bitmaps_exist_and_load():
    for name in DEMO_FILES:
        path = halftone.ASSETS_DIR / name
        assert path.is_file(), f"missing demo bitmap {path}"
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"))
        assert arr.shape == (512, 512)
        assert arr.std() > 10, f"{name} is tonally flat — bad demo render"


def test_darker_half_gets_strictly_more_gold(halfcard: str):
    gp = registry["bitmap-halftone"].generate(
        image=halfcard, extent_um=EXTENT, back_mode="none"
    )
    left, right = _half_areas(gp.front)
    assert left > right, f"dark half {left:.0f} not > light half {right:.0f}"
    assert gp.min_feature_um >= 2.0
    assert 0.0 < gp.extra["coverage_front"] < 1.0


def test_invert_flips_the_area_relationship(halfcard: str):
    gp = registry["bitmap-halftone"].generate(
        image=halfcard, extent_um=EXTENT, back_mode="none", invert=True
    )
    left, right = _half_areas(gp.front)
    assert right > left, f"inverted: light half {right:.0f} not > dark half {left:.0f}"


def test_back_mode_none_yields_empty_back(halfcard: str):
    gp = registry["bitmap-halftone"].generate(
        image=halfcard, extent_um=EXTENT, back_mode="none"
    )
    assert gp.back.is_empty
    assert gp.extra["coverage_back"] == 0.0


def test_back_mode_carrier_is_uniform_grating(halfcard: str):
    gp = registry["bitmap-halftone"].generate(
        image=halfcard, extent_um=EXTENT, back_mode="carrier"
    )
    assert not gp.back.is_empty
    # 50% duty carrier: coverage ~0.5, and every band spans the full width
    # (image content must NOT leak into the carrier).
    assert 0.45 <= gp.extra["coverage_back"] <= 0.55
    for g in gp.back.geoms:
        x0, _, x1, _ = g.bounds
        assert (x1 - x0) > 0.9 * EXTENT, "carrier band does not span the extent"


def test_back_mode_complement_anticorrelates_with_front(halfcard: str):
    gp = registry["bitmap-halftone"].generate(
        image=halfcard, extent_um=EXTENT, back_mode="complement"
    )
    f_left, f_right = _half_areas(gp.front)
    b_left, b_right = _half_areas(gp.back)
    assert f_left > f_right  # front: gold follows the dark (left) half
    assert b_right > b_left  # back: gold follows the light (right) half


def test_back_mode_phase_reveal_ships_a_shifted_copy(halfcard: str):
    gp = registry["bitmap-halftone"].generate(
        image=halfcard, extent_um=EXTENT, back_mode="phase_reveal"
    )
    assert not gp.back.is_empty
    # Same halftone content at half-period phase: near-identical total area...
    assert abs(gp.back.area - gp.front.area) < 0.2 * gp.front.area
    # ...but NOT the same geometry (the phase shift moved the bands).
    fb = {g.bounds for g in gp.front.geoms}
    bb = {g.bounds for g in gp.back.geoms}
    assert fb != bb


def test_budget_guard_refuses_abusive_params(halfcard: str):
    """Max extent at min period is 625 lines x 1400 grid columns = 875k
    halftone cells (x1 layer) — over the 400k cap, so the sizing math is
    DESIGNED to raise here, before any image is loaded or polygon built."""
    with pytest.raises(ValueError, match="lattice cells"):
        registry["bitmap-halftone"].generate(
            image=halfcard, extent_um=5000.0, line_period_um=8.0, back_mode="none"
        )
    # And the halftoned-back modes double the cell count: 215 lines x 1400
    # cols is 301k for a front-only build (would pass) but 602k with a
    # complement back — the guard must count both layers.
    with pytest.raises(ValueError, match="lattice cells"):
        registry["bitmap-halftone"].generate(
            image=halfcard, extent_um=3000.0, line_period_um=14.0, back_mode="complement"
        )


def test_defaults_stay_inside_budget():
    """Default params (2000 um extent, 20 um period, carrier back) must fit:
    100 lines x 800 columns x 1 halftoned layer = 80k cells."""
    cls = registry["bitmap-halftone"]
    d = cls.defaults()
    n_grid = int(round(d["extent_um"] / (d["line_period_um"] / halftone.CELLS_PER_LINE)))
    n_lines = int(np.ceil(d["extent_um"] / d["line_period_um"]))
    assert n_grid <= halftone.MAX_GRID
    assert n_lines * n_grid < 400_000
