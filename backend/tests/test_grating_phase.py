"""One phase convention for every grating emitter: gold line k's LEFT edge sits
at ``(k - phase) * period``.

Three independent emitters bake the same gratings — the coarse raster
(``plates._grating_grid``, which feeds the preview PNG and the plate SVG), the
fine axis-aligned vector path (``export_fine._line_rects_local`` via
``_clip_axis_grating``) and the fine angled vector path
(``export_fine._angled_grating_local_rects``). They once disagreed on the SIGN of
``phase``: the axis path placed the line at ``(k + phase) * period``, which for
the interlace front comb (``phase = -0.25``, period 60 µm) put the fabricated
barrier BAR exactly p/2 = 30 µm off, on top of the designed open SLOT — so the
fabricated switch showed the opposite image from the preview. These tests pin the
shared convention so it cannot fork again.

Comparisons are restricted to gold lines lying FULLY inside the test window: the
axis path clips a straddling line's x to the zone bbox while the angled path
emits the whole line, and that boundary difference is not what is under test.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.export_fine import _angled_grating_local_rects, _line_rects_local
from app.plates import _grating_grid

PERIOD = 60.0          # centerpiece switch / barrier period (exact, µm)
DUTY = 0.5
PITCH = 1.0            # boundary raster, 1 µm/cell — 60 cells per period
W_PX = 240             # x from -120 µm to +120 µm → exactly 4 periods
H_PX = 6
X0, X1 = -120.0, 120.0
TOL = 1e-9


def _fully_inside(edges, line_w: float) -> list[float]:
    """Keep only left edges whose whole gold line sits strictly inside the window."""
    return sorted(e for e in edges if e > X0 + TOL and e + line_w < X1 - TOL)


def _expected_left_edges(phase: float, duty: float = DUTY) -> list[float]:
    """Left edges of the ONE shared convention: line k starts at (k − phase)·p."""
    return _fully_inside(((k - phase) * PERIOD for k in range(-6, 7)), duty * PERIOD)


def _raster_left_edges(phase: float) -> list[float]:
    """Left edge (µm) of every gold run in a ``_grating_grid`` raster row."""
    grid = _grating_grid(W_PX, H_PX, PITCH, PERIOD, DUTY, 0.0, phase=phase)
    row = grid[H_PX // 2]
    xs = (np.arange(W_PX) - (W_PX - 1) / 2.0) * PITCH
    starts = np.flatnonzero(row & ~np.concatenate([[False], row[:-1]]))
    return _fully_inside((float(xs[c] - PITCH / 2.0) for c in starts), DUTY * PERIOD)


def _axis_left_edges(phase: float, duty: float = DUTY) -> list[float]:
    rects = _line_rects_local((X0, -3.0, X1, 3.0), PERIOD, duty, phase)
    return _fully_inside((float(v) for v in rects[:, 0]), duty * PERIOD)


def _angled_left_edges(phase: float, duty: float = DUTY) -> list[float]:
    """The angled emitter at angle 0 — same geometry, different code path."""
    zone = np.ones((H_PX, W_PX), dtype=bool)
    rects = _angled_grating_local_rects(
        zone, PITCH, (W_PX * PITCH, H_PX * PITCH), PERIOD, duty, 0.0, phase=phase
    )
    return _fully_inside({round(float(v), 6) for v in rects[:, 0]}, duty * PERIOD)


def test_interlace_comb_phase_agrees_across_all_three_emitters() -> None:
    """phase = -0.25 is the load-bearing case: the front barrier comb."""
    expected = _expected_left_edges(-0.25)
    assert expected == [-105.0, -45.0, 15.0, 75.0]
    assert _raster_left_edges(-0.25) == pytest.approx(expected)
    assert _axis_left_edges(-0.25) == pytest.approx(expected)
    assert _angled_left_edges(-0.25) == pytest.approx(expected)


def test_axis_and_angled_emitters_agree_for_every_phase_in_use() -> None:
    # 0.0 (carriers, A lane), 0.5 (B lane), -0.25 (front comb) are the phases the
    # fine writer actually emits; ±0.5/±0.25 guard the sign in both directions.
    for phase in (0.0, 0.5, -0.25, -0.5, 0.25):
        expected = _expected_left_edges(phase)
        assert expected, phase
        assert _axis_left_edges(phase) == pytest.approx(expected), phase
        assert _angled_left_edges(phase) == pytest.approx(expected), phase
        assert _raster_left_edges(phase) == pytest.approx(expected), phase


def test_negative_phase_shifts_gold_in_the_positive_x_direction() -> None:
    """Regression: the banned ``(k + phase)`` sign would put the bar at -15 µm."""
    edges = _axis_left_edges(-0.25)
    assert 15.0 in edges          # (k − phase)·p at k = 0
    assert -15.0 not in edges     # (k + phase)·p — the inverted comb


def test_half_period_phase_is_period_equivalent() -> None:
    # phase 0.5 and -0.5 describe the SAME comb (the B interlace lane); a sign
    # error there is invisible, which is why -0.25 is the one that bit us.
    assert _axis_left_edges(0.5) == pytest.approx(_axis_left_edges(-0.5))


def test_no_edge_line_is_dropped_at_the_zone_boundary() -> None:
    # The k-enumeration must bracket the bbox for any phase/duty: a line whose
    # gold only partially overlaps the bbox is CLIPPED, never skipped.
    for phase in (-0.45, -0.25, 0.0, 0.3, 0.5):
        for duty in (0.25, 0.5, 0.75):
            rects = _line_rects_local((X0, -3.0, X1, 3.0), PERIOD, duty, phase)
            line_w = duty * PERIOD
            want = sorted(
                max((k - phase) * PERIOD, X0)
                for k in range(-6, 7)
                if (k - phase) * PERIOD + line_w > X0 + TOL
                and (k - phase) * PERIOD < X1 - TOL
            )
            got = sorted(float(v) for v in rects[:, 0])
            assert got == pytest.approx(want), (phase, duty)
