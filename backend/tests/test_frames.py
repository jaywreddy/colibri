"""Phase I acceptance tests for the frame engine.

Mirrors the spec's harness:
  1. Every motif drawer runs without throwing (call each in isolation).
  2. For every theme × algorithm: a non-empty scene with no non-finite coords.
  3. Segment midpoints stay within the decoration band (proves the "follows
     the frame, doesn't spiral inward" requirement).
  4. The reveal animation reaches maxT (growth completes).

Plus a determinism check: same seed → identical scene.
"""
from __future__ import annotations

import json
import math

import pytest

from app.patterns.frames import (
    RectFrame,
    Scene,
    generate_frame,
    scene_to_multipolygon,
)
from app.patterns.frames.api import FrameParams
from app.patterns.frames.motifs import FLOWERS, LEAVES
from app.patterns.frames.shapely_pen import ShapelyPen
from app.patterns.frames.themes import THEMES


# ---- (1) every motif draws without throwing --------------------------------

@pytest.mark.parametrize("name,drawer", list(FLOWERS.items()))
def test_flower_drawer_runs(name: str, drawer) -> None:
    pen = ShapelyPen()
    drawer(pen, size=200.0, seed=1234)
    mp = pen.finish()
    assert not mp.is_empty, f"flower '{name}' produced empty geometry"


@pytest.mark.parametrize("name,drawer", list(LEAVES.items()))
def test_leaf_drawer_runs(name: str, drawer) -> None:
    pen = ShapelyPen()
    drawer(pen, size=200.0, seed=4321)
    mp = pen.finish()
    assert not mp.is_empty, f"leaf '{name}' produced empty geometry"


# ---- (2) scene has no non-finite coords ------------------------------------

@pytest.mark.parametrize("theme_slug", list(THEMES.keys()))
@pytest.mark.parametrize("algorithm", ["colonize"])
def test_scene_finite_coords(theme_slug: str, algorithm: str) -> None:
    rect = RectFrame(width_um=2000.0, height_um=1500.0)
    params = FrameParams(algorithm=algorithm, theme_slug=theme_slug, seed=7)
    scene = generate_frame(rect, params)
    assert len(scene.segments) > 0, "expected at least one segment"
    for s in scene.segments:
        for v in (s.x1, s.y1, s.x2, s.y2, s.w, s.t):
            assert math.isfinite(v), f"segment has non-finite coord: {s}"
    for f in scene.flowers:
        for v in (f.x, f.y, f.size, f.rot, f.t):
            assert math.isfinite(v), f"flower has non-finite coord: {f}"
    for l in scene.leaves:
        for v in (l.x, l.y, l.angle, l.size, l.t):
            assert math.isfinite(v), f"leaf has non-finite coord: {l}"


# ---- (3) segment midpoints stay within the band ---------------------------

def test_segments_hug_frame() -> None:
    """Median midpoint distance to the frame must stay inside the band.

    Catches the "free-integrated flow field that spirals inward" failure
    mode the spec warns about. We're generous on the cap (3× the band) so
    minor wiggle excursions don't false-positive.
    """
    w, h = 2000.0, 1500.0
    rect = RectFrame(width_um=w, height_um=h)
    band = 0.12 * min(w, h)
    params = FrameParams(
        algorithm="colonize",
        theme_slug="esmeralda",
        seed=42,
        frame_width_um=band,
    )
    scene = generate_frame(rect, params)
    assert scene.segments, "no segments generated"
    midpoint_dists = [
        rect.dist_frame(0.5 * (s.x1 + s.x2), 0.5 * (s.y1 + s.y2))
        for s in scene.segments
    ]
    midpoint_dists.sort()
    median = midpoint_dists[len(midpoint_dists) // 2]
    assert median <= band * 1.2, (
        f"median midpoint distance {median:.1f}μm exceeds band {band:.1f}μm "
        "— algorithm is drifting toward the center"
    )
    # No midpoint should be more than 3× the band — that's clearly outside
    # the decoration zone even with wiggle.
    assert max(midpoint_dists) <= band * 3.0, (
        f"max midpoint distance {max(midpoint_dists):.1f}μm exceeds 3× band"
    )


# ---- (4) reveal reaches maxT ----------------------------------------------

def test_reveal_completes() -> None:
    rect = RectFrame(width_um=1800.0, height_um=1800.0)
    params = FrameParams(seed=99)
    scene = generate_frame(rect, params)
    assert scene.max_t > 0, "expected non-zero growth time"
    # Every segment's t should be <= max_t, and at least one should equal it.
    ts = [s.t for s in scene.segments]
    assert max(ts) == scene.max_t, "max segment t must equal scene.max_t"
    assert min(ts) >= 0


# ---- (5) determinism -------------------------------------------------------

def test_same_seed_same_scene() -> None:
    rect = RectFrame(width_um=2000.0, height_um=1500.0)
    p = FrameParams(seed=2024, density=1.0, bloom=0.7, foliage=0.5)
    a = generate_frame(rect, p)
    b = generate_frame(rect, p)
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(
        b.to_dict(), sort_keys=True
    ), "same seed should produce identical scenes"


def test_different_seed_different_scene() -> None:
    rect = RectFrame(width_um=2000.0, height_um=1500.0)
    a = generate_frame(rect, FrameParams(seed=1))
    b = generate_frame(rect, FrameParams(seed=2))
    assert json.dumps(a.to_dict(), sort_keys=True) != json.dumps(
        b.to_dict(), sort_keys=True
    ), "different seeds should produce different scenes"


# ---- (6) end-to-end: scene → mask ------------------------------------------

def test_scene_to_multipolygon_produces_mask() -> None:
    rect = RectFrame(width_um=2000.0, height_um=1500.0)
    params = FrameParams(seed=11)
    scene = generate_frame(rect, params)
    mask = scene_to_multipolygon(scene, rect, params)
    assert not mask.is_empty, "expected non-empty mask"
    # Mask must lie inside the rect (with a small epsilon for buffer width).
    bx0, by0, bx1, by1 = mask.bounds
    eps = 5.0
    assert bx0 >= rect.left - eps
    assert by0 >= rect.bottom - eps
    assert bx1 <= rect.right + eps
    assert by1 <= rect.top + eps


def test_scene_json_roundtrip() -> None:
    rect = RectFrame(width_um=1600.0, height_um=900.0)
    scene = generate_frame(rect, FrameParams(seed=3))
    payload = json.dumps(scene.to_dict())
    restored = Scene.from_dict(json.loads(payload))
    assert len(restored.segments) == len(scene.segments)
    assert len(restored.flowers) == len(scene.flowers)
    assert len(restored.leaves) == len(scene.leaves)
    assert restored.max_t == scene.max_t


# ---- (7) motif_scale: smaller motifs, denser rank, same band ----------------

def _leaf_stats(scene: Scene) -> tuple[int, float]:
    sizes = [leaf.size for leaf in scene.leaves]
    assert sizes, "expected leaves in the wreath scene"
    return len(sizes), sum(sizes) / len(sizes)


def test_motif_scale_shrinks_motifs_and_fills_in() -> None:
    """0.6 gives ~0.6x-long leaves, more of them, and the SAME vine gauge.

    The point of the knob is a finer-grained foliage read without narrowing
    the band: motif sizes and their station spacing scale together, so the
    rank packs proportionally more (smaller) leaves into the same ribbon.
    """
    rect = RectFrame(width_um=3000.0, height_um=2400.0)
    base = generate_frame(rect, FrameParams(seed=5))
    small = generate_frame(rect, FrameParams(seed=5, motif_scale=0.6))

    n_base, mean_base = _leaf_stats(base)
    n_small, mean_small = _leaf_stats(small)

    ratio = mean_small / mean_base
    assert 0.54 <= ratio <= 0.66, f"mean leaf length ratio {ratio:.3f} != ~0.6"
    # Spacing shrank with the motifs, so the band fills rather than thinning.
    assert n_small > n_base * 1.3, f"expected a denser rank, got {n_small} vs {n_base}"
    # The vine is NOT a motif: its gauge must not move with the dial.
    assert max(s.w for s in small.segments) == pytest.approx(
        max(s.w for s in base.segments)
    ), "motif_scale must leave the vine stroke width alone"


def test_motif_scale_one_is_the_unscaled_wreath() -> None:
    """The default dial is a no-op — pins the knob's neutral position."""
    rect = RectFrame(width_um=2000.0, height_um=1500.0)
    a = generate_frame(rect, FrameParams(seed=17))
    b = generate_frame(rect, FrameParams(seed=17, motif_scale=1.0))
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(
        b.to_dict(), sort_keys=True
    ), "motif_scale=1.0 must be identical to the unscaled generator"


def test_frame_spec_forwards_motif_scale() -> None:
    """FrameSpec -> FrameParams carries the dial (the real production path)."""
    from dataclasses import replace

    from app.plates import FrameSpec

    assert FrameSpec().to_frame_params().motif_scale == 1.0
    fp = replace(FrameSpec(), motif_scale=0.55).to_frame_params()
    assert fp.motif_scale == 0.55
