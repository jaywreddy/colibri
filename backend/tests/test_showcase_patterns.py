"""Sanity checks for the showcase patterns: the parallax-barrier image
switches rebuilt from the retired phase overlays (J+P monogram ↔ heart,
CA ↔ Colombia duo-globe, gear ↔ quill, colibrí wing beat), the spinning-globe
lenticular, the Cattleya shimmer, the J+P carrier reveal, and the globe-motif
rotation support they build on. Extents are kept small so each generate()
stays in the sub-second range on the constrained dev host."""
from __future__ import annotations

import math

import numpy as np
import pytest

import app.patterns.artistic  # noqa: F401  -- populate registry
from app.patterns.base import RECIPE_NAMES, Substrate, registry
from app.patterns.motifs import globe, monogram
from app.rasterize import rasterize

SHOWCASE_SLUGS = [
    "globe-rotation-stereo",
    "orchid-shimmer-moire",
    "jp-monogram-phase",
    "globe-duo-phase",
    "gear-quill-switch",
    "colibri-flap-phase",
    "monogram-carrier-reveal",
]

# The slugs rebuilt from phase overlays into parallax barriers: both images
# interlaced in the BACK layer as half-period column channels, pure FULL-FIELD
# slit mask in FRONT (never clipped to the silhouettes — a union-gated comb is
# itself a static front image; the front-coverage test below is the tripwire
# for that defect class). colibri-globe-phase, once first in this list, was
# removed as a redundant twin of colibri-globe-lenticular.
BARRIER_SLUGS = [
    "jp-monogram-phase",
    "globe-duo-phase",
    "gear-quill-switch",
    "colibri-flap-phase",
]

# Small extent keeps raster grids and lattice cell counts tiny.
SMALL_EXTENT = 800.0


def _assert_layer_inside_extent(mp, extent_um: float, name: str) -> None:
    assert not mp.is_empty, f"{name} is empty"
    assert mp.area > 0, f"{name} has zero area"
    half = extent_um / 2 + 1e-6
    minx, miny, maxx, maxy = mp.bounds
    assert minx >= -half and maxx <= half, f"{name} exceeds extent in x: {mp.bounds}"
    assert miny >= -half and maxy <= half, f"{name} exceeds extent in y: {mp.bounds}"


def test_showcase_slugs_registered_with_valid_recipe_and_descriptor():
    for slug in SHOWCASE_SLUGS:
        assert slug in registry, f"{slug} missing from registry"
        cls = registry[slug]
        assert cls.render_recipe in RECIPE_NAMES
        d = cls.descriptor()
        assert d["slug"] == slug
        assert d["name"]
        assert d["render_recipe"] == cls.render_recipe
        assert d["params"], f"{slug} exposes no params"


@pytest.mark.parametrize("slug", SHOWCASE_SLUGS)
def test_generate_small_extent_nonempty_layers_inside_bounds(slug):
    cls = registry[slug]
    kwargs = cls.defaults() | {"extent_um": SMALL_EXTENT}
    gp = cls.generate(**kwargs)
    assert gp.extent_um == (SMALL_EXTENT, SMALL_EXTENT)
    _assert_layer_inside_extent(gp.front, SMALL_EXTENT, f"{slug} front")
    _assert_layer_inside_extent(gp.back, SMALL_EXTENT, f"{slug} back")


def test_stereo_views_present_and_differ():
    cls = registry["globe-rotation-stereo"]
    gp = cls.generate(**(cls.defaults() | {"extent_um": SMALL_EXTENT}))
    view_a = gp.extra_layers.get("view_a")
    view_b = gp.extra_layers.get("view_b")
    assert view_a is not None and view_b is not None
    _assert_layer_inside_extent(view_a, SMALL_EXTENT, "view_a")
    _assert_layer_inside_extent(view_b, SMALL_EXTENT, "view_b")
    # The two spin states occupy complementary interlace columns AND depict
    # different rotations, so at least one of area / bounds must differ.
    # (No symmetric_difference here — the raster-strip MultiPolygons are
    # concatenations whose members share edges, i.e. not GEOS-boolean-safe.)
    assert (
        abs(view_a.area - view_b.area) > 1e-9 or view_a.bounds != view_b.bounds
    ), "view_a and view_b are indistinguishable"
    assert gp.recipe_data["slit_axis_deg"] == 0.0
    assert gp.recipe_data["slit_period_um"] == cls.defaults()["slit_period_um"]


# ---------------------------------------------------------------------------
# Rebuilt barrier switches (formerly phase overlays)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", BARRIER_SLUGS)
def test_barrier_rebuild_uses_stereo_recipe_and_slit_keys(slug):
    cls = registry[slug]
    assert cls.render_recipe == "stereo_lenticular"
    gp = cls.generate(**(cls.defaults() | {"extent_um": SMALL_EXTENT}))
    assert gp.recipe_data["slit_axis_deg"] == 0.0
    assert gp.recipe_data["slit_period_um"] == cls.defaults()["slit_period_um"]
    # The retired phase-overlay keys must be gone — the shader binding
    # switched from phase_shift_overlay to stereo_lenticular.
    assert "switch_axis_deg" not in gp.recipe_data
    assert "carrier_period_um" not in gp.recipe_data


@pytest.mark.parametrize("slug", BARRIER_SLUGS)
def test_barrier_back_holds_both_interlace_channels(slug):
    cls = registry[slug]
    gp = cls.generate(**(cls.defaults() | {"extent_um": SMALL_EXTENT}))
    view_a = gp.extra_layers.get("view_a")
    view_b = gp.extra_layers.get("view_b")
    assert view_a is not None and view_b is not None
    _assert_layer_inside_extent(view_a, SMALL_EXTENT, f"{slug} view_a")
    _assert_layer_inside_extent(view_b, SMALL_EXTENT, f"{slug} view_b")
    # Both images depict different motifs on complementary column sets.
    # (Area/bounds comparison only — the raster-strip MultiPolygons are
    # concatenations whose members share edges, i.e. not GEOS-boolean-safe.)
    assert (
        abs(view_a.area - view_b.area) > 1e-9 or view_a.bounds != view_b.bounds
    ), f"{slug} view_a and view_b are indistinguishable"
    # The back layer is the concatenation of BOTH channels (channels never
    # overlap, so member areas sum exactly) — i.e. all image content lives
    # behind the slit mask, and each channel contributes non-trivially.
    # (No fixed headroom factor: image pairs can be very asymmetric — the
    # sparse colibri holds ~16% of the back vs the solid globe's ~84%.)
    assert gp.back.area == pytest.approx(view_a.area + view_b.area, rel=1e-9)
    assert min(view_a.area, view_b.area) > 0.01 * gp.back.area, (
        f"{slug}: one interlace channel is nearly empty"
    )


@pytest.mark.parametrize("slug", BARRIER_SLUGS)
def test_barrier_front_is_slit_mask_with_expected_coverage(slug):
    cls = registry[slug]
    defaults = cls.defaults()
    gp = cls.generate(**(defaults | {"extent_um": SMALL_EXTENT}))
    duty = defaults["slit_duty"]
    coverage = gp.front.area / (SMALL_EXTENT * SMALL_EXTENT)
    # Front is a pure barrier: gold fraction = 1 - slit_duty, up to the
    # partial stripes cut at the crop window edges (≤ one stripe width).
    assert coverage == pytest.approx(1 - duty, abs=0.05), (
        f"{slug} front coverage {coverage:.3f} != 1 - duty {1 - duty:.3f}"
    )


@pytest.mark.parametrize("slug", BARRIER_SLUGS)
def test_barrier_switch_metadata_includes_snell_and_quarter_period(slug):
    """The switch completes at s = p/4 (the slit straddles the channel
    boundary) and the exterior angle includes the substrate index:
    θ(s) = asin(n · sin(atan(s / t))). The legacy metadata used atan(p/(2t))
    — wrong shift and no Snell factor."""
    cls = registry[slug]
    p = cls.defaults()["slit_period_um"]
    gp = cls.generate(**(cls.defaults() | {"extent_um": SMALL_EXTENT}))
    sub = Substrate()
    expected_switch = math.degrees(
        math.asin(sub.n * math.sin(math.atan(p / 4 / sub.thickness_um)))
    )
    expected_zone = math.degrees(
        math.asin(sub.n * math.sin(math.atan(p / 2 / sub.thickness_um)))
    )
    assert gp.extra["switch_half_angle_deg"] == pytest.approx(expected_switch, rel=1e-6)
    assert gp.extra["zone_half_angle_deg"] == pytest.approx(expected_zone, rel=1e-6)


@pytest.mark.parametrize("slug", BARRIER_SLUGS)
def test_barrier_budget_guard_rejects_abusive_params(slug):
    """Min period at max extent would emit >1M interlace rectangles — the
    lattice budget guard must refuse before any raster or GEOS work."""
    cls = registry[slug]
    with pytest.raises(ValueError, match="lattice cells"):
        cls.generate(slit_period_um=8.0, extent_um=5000.0)


# ---------------------------------------------------------------------------
# Carrier reveal (honest single-image anti-phase effect)
# ---------------------------------------------------------------------------


def test_carrier_reveal_recipe_and_metadata():
    cls = registry["monogram-carrier-reveal"]
    assert cls.render_recipe == "moire_interactive"
    gp = cls.generate(**(cls.defaults() | {"extent_um": SMALL_EXTENT}))
    p = cls.defaults()["period_um"]
    sub = Substrate()
    expected_vanish = math.degrees(
        math.asin(sub.n * math.sin(math.atan(p / 2 / sub.thickness_um)))
    )
    assert gp.extra["carrier_period_um"] == p
    assert gp.extra["vanish_angle_deg"] == pytest.approx(expected_vanish, rel=1e-6)


def _eroded(mask: np.ndarray, steps: int = 2) -> np.ndarray:
    """Cheap 4-neighbourhood erosion via rolls — keeps only interior pixels."""
    m = mask
    for _ in range(steps):
        m = (
            m
            & np.roll(m, 1, axis=0)
            & np.roll(m, -1, axis=0)
            & np.roll(m, 1, axis=1)
            & np.roll(m, -1, axis=1)
        )
    return m


def test_carrier_reveal_carriers_are_antiphase():
    """Head-on (zero shift) the front and back carriers interlock inside the
    monogram — near-zero open area — while a half-period back shift aligns
    the two carriers and opens the figure up. This is the polygon-level
    signature of an exact anti-phase carrier pair; a phase error would leak
    transmission at zero shift and kill the reveal contrast."""
    cls = registry["monogram-carrier-reveal"]
    period = cls.defaults()["period_um"]  # 40 um
    gp = cls.generate(**(cls.defaults() | {"extent_um": SMALL_EXTENT}))

    # Rasterize both layers at a pitch that divides the half-period exactly
    # so the half-period shift is an integer pixel roll (5 um -> 4 px).
    pitch = period / 8.0
    n_px = int(round(SMALL_EXTENT / pitch))
    fr = np.asarray(rasterize(gp.front, gp.extent_um, pitch)) > 127
    bk = np.asarray(rasterize(gp.back, gp.extent_um, pitch)) > 127
    assert fr.shape == bk.shape == (n_px, n_px)

    # Interior of the monogram (eroded so silhouette-edge pixels from the
    # two different sampling grids don't pollute the statistics).
    fig = _eroded(monogram.jp_monogram_silhouette(gp.extent_um, n_grid=n_px))
    assert fig.sum() > 200, "eroded monogram interior unexpectedly small"

    open_zero = float((~fr & ~bk)[fig].mean())
    half_px = int(round(period / 2.0 / pitch))
    bk_half = np.roll(bk, half_px, axis=1)
    open_half = float((~fr & ~bk_half)[fig].mean())

    # Zero shift: carriers are complementary -> the figure is (near-)opaque.
    assert open_zero < 0.1, f"anti-phase interlock leaks: open@0 = {open_zero:.3f}"
    # Half period: back stripes hide behind the front's -> the figure opens
    # to roughly the carrier duty (0.5), i.e. it dissolves into the ground.
    assert open_half > 0.25, f"reveal never opens: open@p/2 = {open_half:.3f}"
    assert open_half > open_zero + 0.2


def test_carrier_reveal_budget_guard_rejects_abusive_params():
    """Max extent at min period would emit ~1M back-carrier rectangles — the
    lattice budget guard must refuse before any raster or GEOS work."""
    cls = registry["monogram-carrier-reveal"]
    with pytest.raises(ValueError, match="lattice cells"):
        cls.generate(period_um=6.0, extent_um=5000.0)


# ---------------------------------------------------------------------------
# Globe motif rotation + orchid budget guard (unchanged support checks)
# ---------------------------------------------------------------------------


def test_globe_rotation_zero_matches_legacy_output():
    """rotation_deg=0 must reproduce the pre-parameter output byte-for-byte —
    cached variants of every existing globe pattern depend on it."""
    legacy = globe.globe_silhouette((1000.0, 1000.0), n_grid=128)
    explicit_zero = globe.globe_silhouette((1000.0, 1000.0), n_grid=128, rotation_deg=0.0)
    assert np.array_equal(legacy, explicit_zero)


def test_globe_rotation_changes_silhouette():
    a = globe.globe_silhouette((1000.0, 1000.0), n_grid=128, rotation_deg=0.0)
    b = globe.globe_silhouette((1000.0, 1000.0), n_grid=128, rotation_deg=40.0)
    assert not np.array_equal(a, b)
    # Both spins still draw a globe of comparable coverage.
    assert 0.01 < float(b.mean()) < 0.95


def test_orchid_budget_guard_rejects_abusive_params():
    """Max extent at min period would emit ~1M front rectangles — the lattice
    budget guard must refuse before any raster or GEOS work happens."""
    cls = registry["orchid-shimmer-moire"]
    with pytest.raises(ValueError, match="lattice cells"):
        cls.generate(period_um=6.0, extent_um=5000.0)
    # Defaults stay comfortably inside the budget (generate() succeeds).
    gp = cls.generate(**(cls.defaults() | {"extent_um": SMALL_EXTENT}))
    assert not gp.front.is_empty
