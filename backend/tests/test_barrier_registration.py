"""Barrier REGISTRATION regression — switch_metrics on the barrier exemplar's
REAL generated geometry.

``globe-duo-phase`` is the one parallax-barrier construction the repo keeps
(2026-09-16): no production face carries a switch, but the construction is kept
built and measured so it cannot rot. The metrics themselves used to be pinned
separately against synthetic masks that were perfectly registered by
construction, which could only prove the METRICS work; it never looked at a
shipped pattern. That gap is how the
2026-07 audit found every p=60 barrier misregistered: the back interlace period
was derived from the extent (60.15 µm at the defaults) instead of from the slit
lattice (60.0 µm), and the open slit sat ~p/8 off the channel boundary, so the
±p/4 extinction the manifests advertise never happened and the "hidden" image
stayed ~19 % visible at every tilt. The 0.819/0.000 separation quoted in those
modules had been measured once on the p=40 family, whose lattice came out
accidentally exact, then copy-pasted. A second defect rode along: with a fixed
p/2 barrier phase the registration CLASS flipped with ``extent_um`` — half the
100 µm slider steps turned the left↔right switch into a sign-symmetric blink at
twice the stamped angle, while ``switch_half_angle_deg`` kept reporting p/4.

This file closes both holes by measuring the real thing: ``generate()`` at
DEFAULT params, rasterize the emitted front/back MultiPolygons, and require the
straddle signature at the CLAUDE.md barrier contract shift —

  * strong channel separation at a ±p/4 back shift,
  * the dominant channel SWAPPING with the tilt sign,
  * 50/50 mud at ±p/2 (the aliasing zone; a slit-CENTRED barrier is the exact
    opposite — sharp at p/2, mud at p/4 — which is how the class is pinned).

Cost: two ``generate()`` calls at legal extents (never beyond — the host
bugchecks under memory load) and ~1200x1200 px rasters. No file I/O, no
registry-wide sweep.
"""
from __future__ import annotations

import numpy as np
import pytest

import app.patterns.artistic  # noqa: F401  -- populate registry
from app.patterns._helpers import exterior_tilt_deg
from app.patterns.base import registry
from app.rasterize import rasterize
from app.sim2d import switch_metrics

# The parallax-barrier exemplar: BOTH images interlaced in the BACK layer as
# half-period column channels, pure full-field slit mask in FRONT.
BARRIER_SLUGS = ["globe-duo-phase"]

# One non-default extent, chosen as a legal 100 µm slider step that was the
# WRONG registration class before the phase became extent-aware: it exercises
# the other of the two barrier-phase solutions (mod p/2) for the p=60 family.
EXTENT_CASES = [("globe-duo-phase", 2100.0)]

# Raster pitch = slit_period_um / 24, so every lattice the measurement depends
# on lands on whole pixels: the p/2 channel is 12 px, the p/4 shift is 6 px, and
# the p/8 interlace cell is 3 px. Finer buys nothing and costs memory.
PX_PER_PERIOD = 24

# Columns of open slit trimmed from each aperture edge before measuring.
APERTURE_ERODE_PX = 2


def _erode_aperture(front: np.ndarray, steps: int) -> np.ndarray:
    """Dilate the front GOLD by `steps` columns — i.e. shrink the open slit.

    The comb's bar edges land on fractional pixels (the solved barrier phase is
    not generally a multiple of the raster pitch), so PIL's polygon fill can
    place them a pixel either side of the true edge, and at slit_duty = 0.5 the
    slit exactly covers one channel at ±p/4 — zero geometric margin. Shrinking
    the aperture absorbs that ambiguity. It cannot manufacture a pass: it only
    ever LOWERS a measured visibility, it is applied identically at both tilt
    signs, and a misregistered barrier leaks over ~p/8 (6 px here), an order of
    magnitude more than this trim.
    """
    g = np.array(front, dtype=bool)
    for _ in range(steps):
        d = g.copy()
        d[:, 1:] |= g[:, :-1]
        d[:, :-1] |= g[:, 1:]
        g = d
    return g


def _channel_aligned_slice(chan_a: np.ndarray, period_px: int) -> slice:
    """Column slice starting ON a channel-A boundary, an integer number of slit
    periods wide.

    ``switch_metrics`` labels the two interlace channels by ABSOLUTE column
    phase from the left edge of the frame ((x mod p) < p/2), while
    ``raster_to_polygons`` centres the interlace grid on the extent and the
    generators FLOOR the row count — so the grid is inset from the extent by up
    to one cell (2.5 µm at the p=60 defaults). Left uncorrected, the metric's
    labels mix the two real channels and understate even a perfectly registered
    barrier. The offset is MEASURED from the view_a layer rather than recomputed
    from the generator's arithmetic, so this stays a black-box check; a barrier
    whose interlace period does not equal the slit period has no single
    alignment and still fails the separation floor below.
    """
    col = np.asarray(chan_a).sum(axis=0, dtype=np.int64)
    idx = np.arange(col.size)
    half = period_px // 2
    k = max(
        range(period_px),
        key=lambda o: int(col[((idx - o) % period_px) < half].sum()),
    )
    n_periods = (col.size - k) // period_px
    assert n_periods >= 1, "raster narrower than one slit period"
    return slice(k, k + n_periods * period_px)


def _measure(slug: str, **overrides) -> tuple[dict, dict, dict]:
    """Generate `slug`, rasterize its real layers, and return
    (gp.extra, metrics at ±p/4, metrics at ±p/2)."""
    cls = registry[slug]
    params = cls.defaults() | overrides
    p = float(params["slit_period_um"])
    gp = cls.generate(**params)

    pitch = p / PX_PER_PERIOD
    front = np.asarray(rasterize(gp.front, gp.extent_um, pitch)) > 127
    back = np.asarray(rasterize(gp.back, gp.extent_um, pitch)) > 127
    chan_a = np.asarray(
        rasterize(gp.extra_layers["view_a"], gp.extent_um, pitch)
    ) > 127

    sl = _channel_aligned_slice(chan_a, PX_PER_PERIOD)
    f = _erode_aperture(front, APERTURE_ERODE_PX)[:, sl]
    b = back[:, sl]
    return (
        gp.extra,
        switch_metrics(f, b, pitch, p, shift_um=p / 4.0),
        switch_metrics(f, b, pitch, p, shift_um=p / 2.0),
    )


def _assert_straddle_barrier(slug: str, extra: dict, quarter: dict, half: dict) -> None:
    p = float(registry[slug].defaults()["slit_period_um"])

    # Registration witnesses the generators publish. interlace_period_um is the
    # raster-from-slit-lattice fix: it must equal the front comb's period
    # EXACTLY, not the 60.15 µm the extent-derived raster used to produce.
    assert extra["interlace_period_um"] == pytest.approx(p, rel=1e-12), (
        f"{slug}: back interlace period {extra['interlace_period_um']} "
        f"!= front comb period {p}"
    )
    # The stamped switch angle is the p/4 angle — the separation floor below is
    # what makes that stamp honest rather than aspirational.
    assert extra["switch_half_angle_deg"] == pytest.approx(
        exterior_tilt_deg(p / 4.0), rel=1e-9
    )

    dom_plus = max(quarter["vis_a_plus"], quarter["vis_b_plus"])
    sup_plus = min(quarter["vis_a_plus"], quarter["vis_b_plus"])
    dom_minus = max(quarter["vis_a_minus"], quarter["vis_b_minus"])
    sup_minus = min(quarter["vis_a_minus"], quarter["vis_b_minus"])

    # The headline assertion. The misregistered build measured ~4.3 here (the
    # hidden image ~19 % visible at every tilt); a registered straddle barrier
    # extinguishes the off-channel image outright.
    assert quarter["separation"] > 50.0, (
        f"{slug}: ±p/4 channel separation {quarter['separation']:.1f} — the "
        f"hidden image never extinguishes "
        f"(vis a/b at +: {quarter['vis_a_plus']:.3f}/{quarter['vis_b_plus']:.3f}, "
        f"at -: {quarter['vis_a_minus']:.3f}/{quarter['vis_b_minus']:.3f})"
    )
    assert dom_plus > 0.2 and dom_minus > 0.2, f"{slug}: shown channel too dark"
    assert sup_plus < 0.02 and sup_minus < 0.02, f"{slug}: hidden channel leaks"

    # A left↔right switch, not a blink: the two tilt signs must show DIFFERENT
    # channels.
    assert (quarter["vis_a_plus"] > quarter["vis_b_plus"]) != (
        quarter["vis_a_minus"] > quarter["vis_b_minus"]
    ), f"{slug}: the same channel dominates at both tilt signs"

    # Class pin. Straddle: sharp at p/4, 50/50 mud at p/2 (the first aliasing
    # zone). Slit-CENTRED registration is the mirror image — mud at p/4, sharp
    # at p/2 — and reports the same switch_half_angle_deg while switching at
    # twice the angle, which is exactly the extent-dependent flip this pins.
    assert half["separation"] < 5.0, (
        f"{slug}: ±p/2 separation {half['separation']:.1f} — this is the "
        f"slit-centred class, not the straddle class the metadata claims"
    )


@pytest.mark.parametrize("slug", BARRIER_SLUGS)
def test_barrier_switches_at_quarter_period_on_real_geometry(slug):
    """Every barrier slug, at its DEFAULT params, must actually extinguish the
    off-channel image at the ±p/4 back shift its manifest advertises."""
    extra, quarter, half = _measure(slug)
    _assert_straddle_barrier(slug, extra, quarter, half)


@pytest.mark.parametrize(("slug", "extent_um"), EXTENT_CASES)
def test_barrier_registration_class_survives_extent(slug, extent_um):
    """The registration class must not depend on the extent slider.

    Before the barrier phase became extent-aware, ``(extent_um/2) mod (p/2)``
    silently chose between straddle and slit-centred, so alternate 100 µm steps
    degenerated to a sign-symmetric blink at twice the stamped switch angle.
    Same assertions as the default-extent case: passing them at a second extent
    is the class-stability proof."""
    extra, quarter, half = _measure(slug, extent_um=extent_um)
    _assert_straddle_barrier(slug, extra, quarter, half)
