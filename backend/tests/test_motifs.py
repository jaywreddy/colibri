"""Sanity checks on every motif module — non-empty geometry, determinism, and
that binary silhouettes convert to non-empty MultiPolygons via raster_to_polygons."""
from __future__ import annotations

import numpy as np

from app.patterns._helpers import raster_to_polygons
from app.patterns.motifs import (
    cafetero,
    cana_flecha,
    caravel,
    colibri,
    compass,
    emerald,
    meridian,
    muzo,
    tairona,
    wayuu,
)

EXTENT = (1000.0, 1000.0)


def test_shapely_motifs_produce_nonempty_geometry():
    checks = [
        ("wayuu.kanasu_diamonds", wayuu.kanasu_diamonds(EXTENT, 20.0)),
        ("emerald.hex_facets", emerald.hex_facets(EXTENT, 30.0)),
        ("cana_flecha.concentric_bands", cana_flecha.concentric_bands(EXTENT, 50.0)),
        ("cana_flecha.pinta_triangles", cana_flecha.pinta_triangles(EXTENT, 200.0, 40.0)),
        ("compass.compass_rose", compass.compass_rose(EXTENT, 200.0)),
        ("compass.spiral_arms", compass.spiral_arms(EXTENT, 6, 20.0)),
        ("meridian.meridian_grid", meridian.meridian_grid(EXTENT)),
        ("tairona.goldwork_spiral", tairona.goldwork_spiral(EXTENT, 10.0, 30.0, n_turns=8)),
        ("tairona.concentric_goldwork", tairona.concentric_goldwork(EXTENT, 30.0)),
        ("cafetero.terrace_envelope", cafetero.terrace_envelope(EXTENT, 80.0)),
    ]
    for name, mp in checks:
        assert not mp.is_empty, f"{name} produced empty geometry"
        assert mp.area > 0, f"{name} produced zero-area geometry"


def test_binary_silhouettes_are_non_trivial():
    """Silhouettes should cover a meaningful fraction of the grid — not all 0, not all 1."""
    for name, fn in [
        ("colibri", colibri.colibri_silhouette),
        ("caravel", caravel.caravel_silhouette),
        ("muzo", muzo.hex_crystal),
    ]:
        grid = fn(EXTENT, n_grid=128)
        frac = float(grid.mean())
        assert 0.01 < frac < 0.7, f"{name} silhouette fill={frac:.3f} out of expected band"


def test_pillow_silhouettes_are_deterministic():
    """Same args → identical bool array (no hidden randomness in draw pipeline)."""
    a = colibri.colibri_silhouette(EXTENT, n_grid=128)
    b = colibri.colibri_silhouette(EXTENT, n_grid=128)
    assert np.array_equal(a, b)


def test_raster_to_polygons_on_silhouette_produces_polygons():
    grid = caravel.caravel_silhouette(EXTENT, n_grid=128)
    cell = EXTENT[0] / grid.shape[0]
    mp = raster_to_polygons(grid.astype(np.uint8), cell, EXTENT)
    assert not mp.is_empty
    assert mp.area > 0


def test_kanasu_rotation_changes_geometry():
    """A rotated lattice should not be bit-identical to the unrotated one."""
    a = wayuu.kanasu_diamonds(EXTENT, 20.0, rotation_deg=0.0)
    b = wayuu.kanasu_diamonds(EXTENT, 20.0, rotation_deg=5.0)
    # Bounding-box extent is preserved (we crop), but symmetric_difference has area.
    sym = a.symmetric_difference(b)
    assert sym.area > 0


def test_hex_facets_period_scales_feature_density():
    """Halving the period should roughly quadruple the number of hexes."""
    a = emerald.hex_facets(EXTENT, period_um=40.0)
    b = emerald.hex_facets(EXTENT, period_um=20.0)
    assert len(list(b.geoms)) > len(list(a.geoms)) * 2
