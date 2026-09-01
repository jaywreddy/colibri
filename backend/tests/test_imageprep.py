"""Halftone image prep — each step is undoing a specific defect of the medium."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.patterns.bitmap.imageprep import (
    PrepSpec,
    load_gray,
    local_contrast,
    linear_to_srgb,
    prep_darkness,
    prepare_asset,
    printable_window,
    screen_period_for,
    srgb_to_linear,
)


def _photo(n: int = 256) -> np.ndarray:
    """A stand-in with the features that break a naive pipeline: a smooth
    gradient (bands), a soft blob (needs local contrast), and true black/white
    corners (get clipped)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    a = 0.25 + 0.5 * yy
    a += 0.18 * np.exp(-(((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02))
    a[: n // 16, : n // 16] = 0.0
    a[-n // 16 :, -n // 16 :] = 1.0
    return np.clip(a, 0.0, 1.0)


# --- the window -------------------------------------------------------------


def test_output_never_leaves_the_printable_window():
    """The medium cannot make 0% or 100% duty: the finest gold band and the
    finest gap are both bounded by the process. Anything outside the window is
    a tone the plate cannot carry."""
    for steps in (8, 22, 30):
        lo, hi = printable_window(steps)
        out = prep_darkness(_photo(), PrepSpec(tone_steps=steps))
        assert out.min() >= lo - 1e-6, f"{steps} steps undershot the window"
        assert out.max() <= hi + 1e-6, f"{steps} steps overshot the window"


def test_the_dither_does_not_escape_the_window():
    """Regression: the dither runs LAST, so clipping it to [0,1] instead of the
    window silently undid the compression on exactly the highlights and shadows
    that step existed to protect."""
    lo, hi = printable_window(22)
    out = prep_darkness(_photo(), PrepSpec(tone_steps=22, dither=True))
    assert out.min() >= lo - 1e-6 and out.max() <= hi + 1e-6
    # ...and it is actually doing something.
    flat = prep_darkness(_photo(), PrepSpec(tone_steps=22, dither=False))
    assert not np.allclose(out, flat)


def test_a_coarser_screen_opens_the_window():
    assert printable_window(8) == (pytest.approx(0.125), pytest.approx(0.875))
    assert printable_window(22)[0] < printable_window(8)[0]


# --- the individual steps ---------------------------------------------------


def test_unsharp_raises_local_contrast():
    """The one step with a directly measurable payoff."""
    src = _photo()
    # dither off: its noise dominates a local-contrast measure on a smooth test
    # image and would mask the very thing this asserts.
    off = prep_darkness(src, PrepSpec(unsharp_amount=0.0, dither=False))
    on = prep_darkness(src, PrepSpec(unsharp_amount=0.75, dither=False))
    assert local_contrast(on) > local_contrast(off) * 1.10


def test_coverage_must_be_linear_light_to_reproduce_the_source():
    """The chain is: coverage -> reflected light (linear in coverage) -> the eye
    re-encodes. So the printed appearance is encode(coverage), and only
    coverage = linearize(source) comes back to the source.

    Feeding sRGB straight in as coverage is the classic error — it renders a
    0.25 midtone as 0.54 — and mapping through L* is nearly a no-op, since L*
    and the sRGB curve agree to within 0.02."""
    g = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(g)), g, atol=1e-4)
    # the two wrong answers, pinned so nobody reintroduces them
    assert linear_to_srgb(g)[1] == pytest.approx(0.537, abs=0.01)   # sRGB as coverage
    assert srgb_to_linear(np.float32(0.5)) == pytest.approx(0.214, abs=0.005)


def test_linearize_darkens_the_midtones_as_it_must():
    mid = np.full((64, 64), 0.5, dtype=np.float32)
    assert srgb_to_linear(mid).mean() < 0.25
    x = np.linspace(0, 1, 64, dtype=np.float32)
    y = srgb_to_linear(x)
    assert np.all(np.diff(y) >= -1e-6)
    assert y[0] == pytest.approx(0.0, abs=1e-4)
    assert y[-1] == pytest.approx(1.0, abs=1e-4)


def test_falloff_darkens_the_edge_and_spares_the_centre():
    flat = np.full((200, 200), 0.6, dtype=np.float32)
    out = prep_darkness(flat, PrepSpec(falloff=0.3, unsharp_amount=0.0,
                                       linearize=False, dither=False))
    assert out[100, 104] > out[5, 5]
    assert out[100, 104] > out[-5, -5]


def test_each_step_can_be_switched_off():
    src = _photo()
    plain = prep_darkness(src, PrepSpec(unsharp_amount=0.0, falloff=0.0,
                                        linearize=False, dither=False))
    assert np.isfinite(plain).all()
    lo, hi = printable_window(22)
    assert plain.min() >= lo - 1e-6 and plain.max() <= hi + 1e-6


# --- dot gain ---------------------------------------------------------------


def test_gain_pulls_tone_back_for_a_process_that_prints_dark():
    """The witness plate's duty ladder measures the bias; this inverts it."""
    src = _photo()
    none = prep_darkness(src, PrepSpec(gain=0.0, dither=False))
    comp = prep_darkness(src, PrepSpec(gain=0.05, dither=False))
    assert comp.mean() < none.mean()


def test_an_absurd_gain_is_refused():
    with pytest.raises(ValueError):
        PrepSpec(gain=0.9)


# --- guards -----------------------------------------------------------------


def test_bad_spec_is_refused():
    with pytest.raises(ValueError):
        PrepSpec(tone_steps=1)
    with pytest.raises(ValueError):
        PrepSpec(clip_percentiles=(90.0, 10.0))


def test_a_colour_image_is_rejected_rather_than_guessed_at():
    with pytest.raises(ValueError):
        prep_darkness(np.zeros((8, 8, 3), dtype=np.float32))


def test_prep_is_deterministic():
    src = _photo()
    assert np.array_equal(prep_darkness(src), prep_darkness(src))


# --- the screen relation ----------------------------------------------------


def test_screen_period_matches_the_depth_cap():
    """steps <= period / 2um, so the coarsest useful period is steps * 2."""
    assert screen_period_for(22) == pytest.approx(44.0)
    assert screen_period_for(8) == pytest.approx(16.0)


# --- round trip -------------------------------------------------------------


def test_prepare_asset_writes_a_readable_eight_bit_asset(tmp_path):
    src = tmp_path / "src.png"
    Image.fromarray((_photo(300) * 255).astype(np.uint8), "L").save(src)
    dst = tmp_path / "out" / "prepped.png"
    stats = prepare_asset(src, dst, size=256, spec=PrepSpec(tone_steps=22))
    assert dst.exists()
    got = Image.open(dst)
    assert got.size == (256, 256) and got.mode == "L"
    lo, hi = printable_window(22)
    assert stats["min"] >= lo - 1e-6 and stats["max"] <= hi + 1e-6


def test_crop_is_expressed_in_source_fractions(tmp_path):
    """So a crop chosen on one export survives a re-export at another size."""
    src = tmp_path / "wide.png"
    a = np.zeros((400, 800), dtype=np.uint8)
    a[:, 400:] = 255
    Image.fromarray(a, "L").save(src)
    left = load_gray(src, crop=(0.0, 0.0, 0.25), size=32)
    right = load_gray(src, crop=(0.55, 0.0, 0.25), size=32)
    assert left.mean() < 0.1
    assert right.mean() > 0.9
