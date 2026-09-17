"""Sanity checks on the surviving motif modules — non-empty geometry,
determinism, and that binary silhouettes convert to non-empty MultiPolygons
via raster_to_polygons.

Two modules are left (2026-09-16): ``globe`` (the front face's orthographic
world, and the exemplar switch's two hemispheres) and ``monogram`` (the lid's
J+P). The shapely lattice motifs — wayuu's kanasü weave, emerald's hex facets —
went with the patterns that used them, and with them the direct pin on the
lattice-budget guard; that guard is still live and still pinned, through
``_helpers.check_lattice_budget`` below.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.patterns._helpers import (
    MAX_LATTICE_CELLS,
    check_lattice_budget,
    raster_to_polygons,
)
from app.patterns.motifs import globe, monogram

EXTENT = (1000.0, 1000.0)


def test_lattice_budget_guard_rejects_machine_killing_params():
    """A 5000 um extent at a 4 um period is inside the UI slider ranges but
    would build ~15M GEOS polygons (~tens of GB commit) — it froze and
    bugchecked the dev machine on 2026-06-10. The guard must refuse with an
    actionable ValueError instead of building the lattice."""
    cells = (5000.0 / 4.0) ** 2
    assert cells > MAX_LATTICE_CELLS
    with pytest.raises(ValueError, match="lattice cells"):
        check_lattice_budget(int(cells), "kanasu diamonds", pitch_um=4.0)
    # and a real motif's own grid stays comfortably inside it
    check_lattice_budget(128 * 128, "globe silhouette", pitch_um=EXTENT[0] / 128)


def test_binary_silhouettes_are_non_trivial():
    """Silhouettes should cover a meaningful fraction of the grid — not all 0,
    not all 1."""
    for name, fn in [
        ("globe", globe.globe_silhouette),
        ("monogram", monogram.monogram_silhouette),
    ]:
        grid = fn(EXTENT, n_grid=128)
        frac = float(grid.mean())
        assert 0.01 < frac < 0.95, f"{name} silhouette fill={frac:.3f} out of expected band"


def test_pillow_silhouettes_are_deterministic():
    """Same args -> identical bool array (no hidden randomness in the draw
    pipeline). The monogram in particular goes through a font raster."""
    for fn in (globe.globe_silhouette, monogram.monogram_silhouette):
        a = fn(EXTENT, n_grid=128)
        b = fn(EXTENT, n_grid=128)
        assert np.array_equal(a, b), fn.__qualname__


def test_raster_to_polygons_on_silhouette_produces_polygons():
    grid = globe.globe_silhouette(EXTENT, n_grid=128)
    cell = EXTENT[0] / grid.shape[0]
    mp = raster_to_polygons(grid.astype(np.uint8), cell, EXTENT)
    assert not mp.is_empty
    assert mp.area > 0
