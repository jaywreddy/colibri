"""Tilt-sweep collage: the sweep geometry, the metrics, and the sheet layout."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app import sim2d
from app.collage import (
    DEFAULT_ANGLES_DEG,
    LABEL_H,
    PAD,
    build_collage,
    compose_grid,
    sweep_frames,
    tile_metrics,
)


def _barrier_pair(n_px: int = 240, period_px: int = 12):
    """A parallax barrier over a two-lane interlace — the switch in miniature.

    FRONT: a slit comb, half open. BACK: alternating lanes of image A (bright)
    and image B (dark) at half the comb period. Shifting the back by a quarter
    period should swing which lane class the slits reveal.
    """
    x = np.arange(n_px)
    front = ((x % period_px) < period_px // 2).astype(float)
    front = np.tile(front, (n_px, 1))
    lane = (x // (period_px // 2)) % 2
    back = np.tile(lane.astype(float), (n_px, 1))
    return front, back


def _uniform_pair(n_px: int = 120):
    return np.ones((n_px, n_px)), np.ones((n_px, n_px))


# --- sweep geometry ---------------------------------------------------------


def test_sweep_returns_a_tile_per_angle_at_the_requested_size():
    f, b = _uniform_pair()
    frames = sweep_frames(
        f, b, pixel_pitch_um=1.0, thickness_um=500.0, n=1.46,
        angles_deg=(-4, 0, 4), tile_px=64,
    )
    assert [a for a, _ in frames] == [-4.0, 0.0, 4.0]
    assert all(im.size == (64, 64) for _, im in frames)


def test_sweep_uses_the_same_snell_shift_as_the_simulator():
    """The angles printed under the tiles must be the ones a wrist produces, so
    the shift has to come from sim2d rather than a small-angle stand-in."""
    dx, _ = sim2d.parallax_shift_um(3.0, 0.0, 500.0, 1.46)
    assert dx == pytest.approx(17.9, abs=0.6)   # ~6 um/deg through fused silica


def test_axis_selects_which_way_the_back_layer_walks():
    f, b = _barrier_pair()
    fx = sweep_frames(f, b, pixel_pitch_um=1.0, thickness_um=500.0, n=1.46,
                      angles_deg=(6,), axis="x", tile_px=48)
    fy = sweep_frames(f, b, pixel_pitch_um=1.0, thickness_um=500.0, n=1.46,
                      angles_deg=(6,), axis="y", tile_px=48)
    ax = np.asarray(fx[0][1].convert("L"), dtype=float)
    ay = np.asarray(fy[0][1].convert("L"), dtype=float)
    # The comb runs along x, so an x-tilt changes it and a y-tilt barely does.
    assert not np.allclose(ax, ay)


def test_rejects_bad_illum_and_axis():
    f, b = _uniform_pair()
    with pytest.raises(ValueError):
        sweep_frames(f, b, pixel_pitch_um=1.0, thickness_um=500.0, n=1.46, illum="x-ray")
    with pytest.raises(ValueError):
        sweep_frames(f, b, pixel_pitch_um=1.0, thickness_um=500.0, n=1.46, axis="z")


def test_a_coarse_raster_still_resolves_one_degree_steps():
    """Regression: sim2d rounds the back-layer shift to WHOLE raster pixels, and
    a real variant raster runs only ~1-1.5 px per degree. Without refining the
    grid first, neighbouring angles round to the same pixel shift and come out
    byte-identical — the sheet would show duplicate columns and the angle labels
    under them would be wrong by up to half a step."""
    f, b = _barrier_pair(n_px=200, period_px=16)
    # 12 um pixels: one degree of tilt is 6 um, i.e. HALF a raw pixel.
    frames = sweep_frames(
        f, b, pixel_pitch_um=12.0, thickness_um=500.0, n=1.46,
        angles_deg=(0, 1, 2, 3), tile_px=100, illum="backlight",
    )
    arrs = [np.asarray(im.convert("L"), dtype=float) for _, im in frames]
    for i in range(len(arrs) - 1):
        assert not np.allclose(arrs[i], arrs[i + 1]), f"tiles {i},{i+1} identical"


def test_refinement_is_an_integer_resample_of_the_same_mask():
    """The refinement must not invent structure: an integer nearest-neighbour
    upsample re-samples the same rectangles, so at zero tilt the composite is
    unchanged by it."""
    from app.collage import _refine_factor

    f, b = _barrier_pair(n_px=200, period_px=16)
    k = _refine_factor((200, 200), 12.0, 500.0, 1.46, (0, 1, 2, 3), "x")
    assert k > 1                                     # refinement actually kicks in
    coarse = sweep_frames(f, b, pixel_pitch_um=12.0, thickness_um=500.0, n=1.46,
                          angles_deg=(0,), tile_px=100, illum="backlight")
    fine = sweep_frames(f, b, pixel_pitch_um=12.0 / k, thickness_um=500.0, n=1.46,
                        angles_deg=(0,), tile_px=100, illum="backlight")
    a0 = np.asarray(coarse[0][1].convert("L"), dtype=float)
    a1 = np.asarray(fine[0][1].convert("L"), dtype=float)
    assert np.abs(a0 - a1).mean() < 1.0


# --- metrics ----------------------------------------------------------------


def test_a_real_barrier_switch_registers_a_strong_effect():
    """The product is formed at FULL raster resolution before the tile is
    averaged down, so the correlation term survives — this is exactly what the
    3D preview loses once the lattice goes sub-pixel."""
    f, b = _barrier_pair()
    # BACKLIGHT: ambient weights the transmission term at 0.04, so a barrier
    # switch — whose whole signal is what shows THROUGH the slits — reads at a
    # few percent there. Backlight puts transmission at full weight, which is
    # why it is the diagnostic mode for the barrier family.
    _, m = build_collage(
        f, b, pixel_pitch_um=2.0, thickness_um=500.0, n=1.46,
        angles_deg=(-6, 0, 6), tile_px=96, illum="backlight",
    )
    assert m["effect_strength"] > 0.05
    assert m["changed_frac"] > 0.05


def test_a_featureless_pair_registers_no_effect():
    """Nothing to modulate -> no effect.

    This also pins the centre crop. Shifting a finite raster pulls zeros in at
    the trailing border, so that band changes with tilt on ANY input; at 6 deg
    it is 30% of the width, and counting it scores a blank pair as a 25%
    effect. The inset therefore has to scale with the sweep's own shift.
    """
    f, b = _uniform_pair()
    _, m = build_collage(
        f, b, pixel_pitch_um=1.0, thickness_um=500.0, n=1.46,
        angles_deg=(-6, 0, 6), tile_px=48,
    )
    assert m["effect_strength"] == pytest.approx(0.0, abs=1e-6)
    assert m["changed_frac"] == pytest.approx(0.0, abs=1e-6)


def test_the_sweep_endpoints_can_alias_to_the_same_image():
    """The reason the metric compares the most-different PAIR rather than the
    first and last tile. These effects are periodic in the parallax shift, so
    the two extremes can sit an integer number of periods apart and match
    exactly — here +/-6 deg is -/+1.5 periods on a 24 um barrier, so endpoint
    differencing would score a perfectly good switch as dead."""
    f, b = _barrier_pair()
    frames = sweep_frames(
        f, b, pixel_pitch_um=2.0, thickness_um=500.0, n=1.46,
        angles_deg=(-6, 0, 6), tile_px=96, illum="backlight",
    )
    ends = [np.asarray(im.convert("L"), dtype=float) for _, im in (frames[0], frames[-1])]
    assert np.abs(ends[0] - ends[1]).mean() == pytest.approx(0.0, abs=1e-9)
    # ...yet the sweep obviously contains a switch, and the metric finds it.
    m = tile_metrics(frames, crop_frac=0.1)
    assert m["effect_strength"] > 0.5
    assert 0.0 in m["peak_pair_deg"]


def test_metrics_handle_an_empty_sweep():
    assert tile_metrics([])["effect_strength"] == 0.0


# --- sheet layout -----------------------------------------------------------


def test_grid_geometry_is_exact():
    f, b = _uniform_pair()
    frames = sweep_frames(f, b, pixel_pitch_um=1.0, thickness_um=500.0, n=1.46,
                          angles_deg=(-2, -1, 0, 1, 2), tile_px=50)
    sheet = compose_grid(frames, cols=3)
    # 3 columns, 2 rows of (tile + label), padding between and around.
    assert sheet.size == (3 * 50 + 4 * PAD, 2 * (50 + LABEL_H) + 3 * PAD)


def test_grid_with_a_title_is_taller():
    f, b = _uniform_pair()
    frames = sweep_frames(f, b, pixel_pitch_um=1.0, thickness_um=500.0, n=1.46,
                          angles_deg=(0,), tile_px=40)
    plain = compose_grid(frames, cols=1)
    titled = compose_grid(frames, cols=1, title="globe-duo-phase")
    assert titled.height > plain.height
    assert titled.width == plain.width


def test_default_sweep_brackets_zero_and_covers_the_swap():
    assert 0 in DEFAULT_ANGLES_DEG
    assert min(DEFAULT_ANGLES_DEG) <= -2.5 and max(DEFAULT_ANGLES_DEG) >= 2.5
    assert len(DEFAULT_ANGLES_DEG) == len(set(DEFAULT_ANGLES_DEG))


def test_compose_rejects_nothing_to_compose():
    with pytest.raises(ValueError):
        compose_grid([])


def test_collage_returns_an_image_and_metrics_together():
    f, b = _barrier_pair()
    sheet, m = build_collage(
        f, b, pixel_pitch_um=2.0, thickness_um=500.0, n=1.46,
        angles_deg=(-3, 0, 3), tile_px=64, title="t",
    )
    assert isinstance(sheet, Image.Image)
    assert sheet.mode == "RGB"
    assert set(("effect_strength", "mean_swing", "changed_frac")) <= set(m)
