"""Tiled DRC checks == direct whole-region checks, and hierarchical GDS
emission flattens to exactly the flat geometry.

The tiled path (drc._tiled_check_stats) exists because klayout's width/space
checks degrade superlinearly with contiguous edge length (a fused full-plate
carrier measured ~27 min/check). Tiling must not change WHAT is measured —
these cases put genuine sub-floor features ON tile boundaries (tile_um shrunk
to 50 µm so a ~200 µm scene spans many tiles) and pin the tiled result to a
direct single-region reference computed with the identical check options.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.patterns.effects import drc as D

kdb = pytest.importorskip("klayout.db")

FLOOR = 2.0
DBU = 0.001


def _direct_stats(polys) -> tuple[int, float, int, float]:
    """Reference: single whole-region checks with the identical options,
    counting merged violation SITES (marker areas) exactly as the tiled path
    defines them — pair granularity is representation-dependent, sites are not."""
    reg, k = D._region_from_polys(polys, DBU)
    floor_dbu = int(round(FLOOR / DBU))
    euc = k.Region.Euclidian
    zdm = k.Region.NeverIncludeZeroDistance

    def tally(pairs):
        markers, mn = k.Region(), float("inf")
        for ep in pairs.each():
            d = abs(ep.distance())
            if d <= 0:
                continue
            markers.insert(ep.polygon(0))
            mn = min(mn, d * DBU)
        markers.merge()
        return int(markers.count()), mn

    nw, mnw = tally(reg.width_check(floor_dbu, metrics=euc, ignore_angle=80, shielded=False, zero_distance_mode=zdm))
    ns, mns = tally(reg.space_check(floor_dbu, metrics=euc, ignore_angle=80, shielded=False, zero_distance_mode=zdm))
    return nw, mnw, ns, mns


def _tiled_stats(polys, tile_um: float = 50.0) -> tuple[int, float, int, float]:
    floor_dbu = int(round(FLOOR / DBU))
    # Drop the trailing violation-marker Regions (they ride along for the
    # heal's surgical weld) — these tests compare the count/min stats.
    return D._tiled_check_stats(
        polys, kdb, width_dbu=floor_dbu, gap_dbu=floor_dbu, dbu_um=DBU, tile_um=tile_um
    )[:4]


def _grating(x0: float, n: int, period: float, width: float, y0: float, y1: float) -> np.ndarray:
    xs = x0 + period * np.arange(n)
    return np.stack([xs, xs + width, np.full(n, y0), np.full(n, y1)], axis=1)


def test_tiled_matches_direct_on_clean_grating():
    # 5 µm-wide lines, 15 µm gaps, 200 µm tall — spans 4+ tiles at tile 50.
    polys = [_grating(-100.0, 10, 20.0, 5.0, -100.0, 100.0)]
    assert _tiled_stats(polys) == _direct_stats(polys)
    nw, _, ns, _ = _tiled_stats(polys)
    assert nw == 0 and ns == 0


def test_tiled_matches_direct_with_boundary_straddling_violations():
    rects = [
        _grating(-100.0, 8, 25.0, 6.0, -80.0, 80.0),
        # Sub-floor SPACE: a 1.4 µm gap pair crossing the y=0 tile boundary.
        np.array([[104.0, 110.0, -30.0, 30.0], [111.4, 118.0, -30.0, 30.0]]),
        # Sub-floor WIDTH: a 1.2 µm sliver straddling the x=150 tile boundary,
        # clear of all other metal so the merge cannot legalize it.
        np.array([[149.4, 150.6, -60.0, 60.0]]),
    ]
    direct = _direct_stats(rects)
    tiled = _tiled_stats(rects)
    assert tiled[0] == direct[0]                      # width-violation count
    assert tiled[2] == direct[2]                      # space-violation count
    assert tiled[1] == pytest.approx(direct[1], abs=2 * DBU)
    assert tiled[3] == pytest.approx(direct[3], abs=2 * DBU)
    assert direct[0] > 0 and direct[2] > 0            # the scenario is real


def test_tiled_matches_direct_on_big_ring_decomposition():
    # One ring spanning >2 tiles (triggers the trapezoid-decompose path) with a
    # genuine 1.5 µm notch gap against a neighboring bar.
    ring = np.array([
        [-120.0, -20.0], [120.0, -20.0], [120.0, 20.0], [-120.0, 20.0]
    ])
    neighbor = np.array([[-120.0, 21.5, 120.0, 25.0]]).reshape(1, 4)
    # neighbor is (N,4) [x0,x1,y0,y1]
    neighbor = np.array([[-120.0, 120.0, 21.5, 25.0]])
    polys = [ring, neighbor]
    direct = _direct_stats(polys)
    tiled = _tiled_stats(polys)
    assert tiled[2] == direct[2] and direct[2] > 0
    assert tiled[3] == pytest.approx(direct[3], abs=2 * DBU)


def test_deduped_gds_flattens_to_flat_geometry():
    from app.export_fine import DBU_UM, insert_polys_deduped

    rng = np.random.default_rng(7)
    line = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 40.0], [0.0, 40.0]])
    polys = [line + np.array([x, 0.0]) for x in np.arange(0.0, 200.0, 10.0)]
    polys += [np.array([[0, 0], [7, 0], [7, 5], [3, 5], [3, 2], [0, 2]], dtype=float) + rng.uniform(220, 260, 2)]

    def build(dedup: bool):
        ly = kdb.Layout()
        ly.dbu = DBU_UM
        top = ly.create_cell("t")
        layer = ly.layer(10, 0)
        if dedup:
            st = insert_polys_deduped(top, layer, polys, dbu_um=DBU_UM)
        else:
            st = None
            for v in polys:
                top.shapes(layer).insert(
                    kdb.Polygon([kdb.Point(int(round(x / DBU_UM)), int(round(y / DBU_UM))) for x, y in v])
                )
        flat = kdb.Region(top.begin_shapes_rec(layer))
        flat.merge()
        return ly, top, flat, st

    _, _, flat_ref, _ = build(dedup=False)
    ly, top, flat_dedup, st = build(dedup=True)
    assert st["cells"] == 1 and st["refs"] == 20 and st["flat"] == 1
    # XOR of the two flattened regions must be empty: identical printed metal.
    assert (flat_ref ^ flat_dedup).is_empty()
