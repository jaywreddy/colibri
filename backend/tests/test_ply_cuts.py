"""Per-ply cut rects and bench marks (``app.ply_cuts``) — pure math, no
geometry build. Everything here runs in milliseconds.

These are the numbers the witness plate's dies are cut and identified by:
``witness_dies.die_dims`` takes its face dimensions from :func:`pair_rects`,
``bench_marks`` its tick code from :func:`id_tick_rects`, and ``dice_ticks``
its street marks from :func:`dice_tick_rects`.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.assembly import FoilSpec, bonded_overlap_um
from app.patterns.base import LITHO_FLOOR_UM
from app.ply_cuts import (
    Placement,
    dice_tick_rects,
    PRODUCTION_ID_TICK_OFFSET_UM,
    fold_band_offset_um,
    id_tick_rects,
    mirror_rects,
    pair_rects,
    subplate_id,
)
from app.witness_geom import PLY_UM as PLY

FOLD = bonded_overlap_um(FoilSpec(tape_width_um=9_525.0), PLY)  # 3/8" tape


# --- cut rects ----------------------------------------------------------------


def test_pair_rects_nested_shell_dims():
    rects = pair_rects(30_000.0, 30_000.0, 33_000.0, PLY)
    assert len(rects) == 12
    by_id = {r.face: r for r in rects}
    for fid in ("front", "back", "top", "bottom", "left", "right"):
        f = by_id[subplate_id(fid, "F")]
        b = by_id[subplate_id(fid, "B")]
        # The inner ply is the nested shell: exactly one ply smaller per edge.
        assert b.width_um == pytest.approx(f.width_um - 2 * PLY)
        assert b.height_um == pytest.approx(f.height_um - 2 * PLY)


def test_pair_rects_spares_duplicate_a_pair_exactly():
    base = {r.face: (r.width_um, r.height_um)
            for r in pair_rects(30_000.0, 30_000.0, 33_000.0, PLY)}
    rects = pair_rects(30_000.0, 30_000.0, 33_000.0, PLY, spare_faces=("front",))
    assert len(rects) == 14
    by_id = {r.face: (r.width_um, r.height_um) for r in rects}
    # A spare is the SAME plate again — interchangeable at the bench.
    assert by_id["front:F:spare1"] == base["front:F"]
    assert by_id["front:B:spare1"] == base["front:B"]
    with pytest.raises(ValueError):
        pair_rects(30_000.0, 30_000.0, 33_000.0, PLY, spare_faces=("lid",))


# --- write transforms ---------------------------------------------------------


def test_mirror_rects_is_an_involution_and_flips_x():
    rects = np.array([[1.0, 3.0, -2.0, 5.0], [-4.0, -1.0, 0.0, 1.0]])
    m = mirror_rects(rects)
    assert np.allclose(m[0], [-3.0, -1.0, -2.0, 5.0])
    assert np.allclose(mirror_rects(m), rects)


# --- bench marks --------------------------------------------------------------


def test_id_ticks_encode_face_and_layer():
    w, h = 30_000.0, 33_000.0
    off = fold_band_offset_um(PLY, FOLD)
    assert off == pytest.approx(PLY + FOLD / 2.0)
    for idx in range(6):
        f = id_tick_rects(idx, False, w, h, PLY, FOLD)
        b = id_tick_rects(idx, True, w, h, PLY, FOLD)
        assert f.shape[0] == idx + 1
        assert b.shape[0] == idx + 2  # + underline bar
        for rects in (f, b):
            widths = np.minimum(rects[:, 1] - rects[:, 0], rects[:, 3] - rects[:, 2])
            assert widths.min() >= LITHO_FLOOR_UM
            # Inside the inner ply footprint, in the interior fold band.
            assert (np.abs(rects[:, 2:]) <= h / 2 - PLY + 1e-6).all()


def test_the_production_tick_band_offset_is_pinned():
    """The ID ticks on the written plate sit 2.94375 mm in from the die edge —
    ``fold_band_offset_um(2250, 1387.5)``, the centre of the interior foil-fold
    band of the BONDED build. The box is six single plies with 1/4" tape now, so
    the derived band would be somewhere else entirely; the offset is PINNED so
    the plate does not change."""
    from app.witness_dies import bench_marks

    assert PRODUCTION_ID_TICK_OFFSET_UM == pytest.approx(fold_band_offset_um(PLY, FOLD))
    assert PRODUCTION_ID_TICK_OFFSET_UM == 2943.75

    h = 32_000.0
    ticks = id_tick_rects(0, False, h, h, PLY, 0.0,
                          band_offset_um=PRODUCTION_ID_TICK_OFFSET_UM)
    cy = (ticks[0][2] + ticks[0][3]) / 2.0
    assert h / 2.0 + cy == pytest.approx(PRODUCTION_ID_TICK_OFFSET_UM)
    # ...and that is what the plate writer actually emits.
    plate = bench_marks("front", "F", h, h)
    assert np.allclose(plate[: len(ticks)], id_tick_rects(
        0, False, h, h, PLY, 0.0, band_offset_um=PRODUCTION_ID_TICK_OFFSET_UM))
    # And where that leaves them on the NEW build: the 1/4" fold reaches
    # 2.05 mm, so the tick is no longer under the copper — it sits in the blank
    # ring between the fold and the 3.6375 mm art rim. Outside the garland,
    # inside the rim, and visible unless the bead covers it (see the constant's
    # docstring). Pinned here so that fact cannot change silently.
    assert 2050.0 < PRODUCTION_ID_TICK_OFFSET_UM < 3637.5


def test_dice_ticks_stay_in_the_street():
    street = 1_000.0
    p = Placement("front:F", 1_000.0, 2_000.0, 20_000.0, 16_000.0, False)
    ticks = dice_tick_rects(p)
    assert ticks.shape[0] == 8  # 4 corners x 2 legs of the L
    for x0, x1, y0, y1 in ticks:
        outside = (
            x1 <= p.x0 or x0 >= p.x0 + p.width_um
            or y1 <= p.y0 or y0 >= p.y0 + p.height_um
        )
        assert outside
        assert x0 >= p.x0 - street and x1 <= p.x0 + p.width_um + street
        assert y0 >= p.y0 - street and y1 <= p.y0 + p.height_um + street


# --- the bonded flags still survive the API bodies -----------------------------


def test_api_body_models_keep_bonded_and_carrier_mode():
    """``bonded`` and ``carrier_scale_mode`` must survive the request bodies —
    a regression once shipped a bonded UI box to the backend as bonded=False."""
    from app.api.boxes import BoxSpecBody
    from app.api.plates import PlateSpecBody

    assert BoxSpecBody(bonded=True).to_spec().bonded is True
    assert BoxSpecBody().to_spec().bonded is False
    p = PlateSpecBody(pattern_slug="monogram-jp", carrier_scale_mode="fixed").to_spec()
    assert p.carrier_scale_mode == "fixed"
    assert PlateSpecBody(pattern_slug="monogram-jp").to_spec().carrier_scale_mode == "gap"
