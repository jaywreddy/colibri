"""Per-ply cut rects and bench marks (``app.ply_cuts``) — pure math, no
geometry build. Everything here runs in milliseconds.

These are the numbers the witness plate's dies are cut and identified by:
``witness_dies.die_dims`` takes its face dimensions from :func:`face_rects`,
``bench_marks`` its tick code from :func:`id_tick_rects`, and ``dice_ticks``
its street marks from :func:`dice_tick_rects`.

``face_rects`` replaced ``pair_rects`` on 2026-09-17 with the rest of the
bonded mechanics. The six written dims did not move — that is pinned in
``test_plates_and_boxes.py::test_the_single_ply_cut_dims_are_the_dies_that_were_written``.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.patterns.base import LITHO_FLOOR_UM
from app.ply_cuts import (
    Placement,
    dice_tick_rects,
    ID_TICK_OFFSET_UM,
    face_rects,
    fold_band_offset_um,
    id_tick_rects,
    mirror_rects,
)
from app.production import PLY_UM as PLY

FOLD = 1387.5
"""The BONDED build's per-face foil fold: 3/8 inch tape (9525 um) wrapping a
stepped edge of three 2.25 mm plies, ``(9525 - 3 * 2250) / 2``. It was
``assembly.bonded_overlap_um(FoilSpec(tape_width_um=9525), PLY)`` until the
bonded math was deleted on 2026-09-17; it is written out here because the
PRODUCTION tick offset below is pinned to what it produced, and a pin with no
derivation beside it is just a number."""


# --- cut rects ----------------------------------------------------------------


def test_face_rects_are_the_stained_glass_cut_list():
    """One rect per face, straight off ``assembly.face_cut_dims``: walls sit ON
    the bottom plate and the left/right walls fit BETWEEN front and back."""
    from app.assembly import FACE_IDS

    W, D, H = 30_000.0, 30_000.0, 33_000.0
    rects = face_rects(W, D, H, PLY)
    assert len(rects) == 6
    by_id = {r.face: r for r in rects}
    assert set(by_id) == set(FACE_IDS)
    for fid in ("top", "bottom"):
        assert (by_id[fid].width_um, by_id[fid].height_um) == (W, D)
    for fid in ("front", "back"):
        assert by_id[fid].width_um == pytest.approx(W)
        assert by_id[fid].height_um == pytest.approx(H - 2 * PLY)
    for fid in ("left", "right"):
        assert by_id[fid].width_um == pytest.approx(D - 2 * PLY)
        assert by_id[fid].height_um == pytest.approx(H - 2 * PLY)


def test_face_rects_spares_duplicate_a_face_exactly():
    base = {r.face: (r.width_um, r.height_um)
            for r in face_rects(30_000.0, 30_000.0, 33_000.0, PLY)}
    rects = face_rects(30_000.0, 30_000.0, 33_000.0, PLY, spare_faces=("front",))
    assert len(rects) == 7
    by_id = {r.face: (r.width_um, r.height_um) for r in rects}
    # A spare is the SAME plate again — interchangeable at the bench.
    assert by_id["front:spare1"] == base["front"]
    with pytest.raises(ValueError):
        face_rects(30_000.0, 30_000.0, 33_000.0, PLY, spare_faces=("lid",))


# --- write transforms ---------------------------------------------------------


def test_mirror_rects_is_an_involution_and_flips_x():
    rects = np.array([[1.0, 3.0, -2.0, 5.0], [-4.0, -1.0, 0.0, 1.0]])
    m = mirror_rects(rects)
    assert np.allclose(m[0], [-3.0, -1.0, -2.0, 5.0])
    assert np.allclose(mirror_rects(m), rects)


# --- bench marks --------------------------------------------------------------


def test_id_ticks_encode_the_face():
    """One bar per face index, in the fold band. (The bonded build underlined
    the bars on a B ply; there are no B plies, and the underline went with
    them on 2026-09-17.)"""
    w, h = 30_000.0, 33_000.0
    off = fold_band_offset_um(PLY, FOLD)
    assert off == pytest.approx(PLY + FOLD / 2.0)
    for idx in range(6):
        rects = id_tick_rects(idx, w, h, PLY, FOLD)
        assert rects.shape[0] == idx + 1
        widths = np.minimum(rects[:, 1] - rects[:, 0], rects[:, 3] - rects[:, 2])
        assert widths.min() >= LITHO_FLOOR_UM
        # Inside the foil-fold band, a ply in from the edge.
        assert (np.abs(rects[:, 2:]) <= h / 2 - PLY + 1e-6).all()


def test_the_production_tick_band_offset_is_pinned():
    """The ID ticks on the written plate sit 2.94375 mm in from the die edge —
    ``fold_band_offset_um(2250, 1387.5)``, the centre of the interior foil-fold
    band of the BONDED build. The box is six single plies with 1/4" tape now, so
    the derived band would be somewhere else entirely; the offset is PINNED so
    the plate does not change."""
    from app.witness_dies import bench_marks

    assert ID_TICK_OFFSET_UM == pytest.approx(fold_band_offset_um(PLY, FOLD))
    assert ID_TICK_OFFSET_UM == 2943.75

    h = 32_000.0
    ticks = id_tick_rects(0, h, h, PLY, 0.0,
                          band_offset_um=ID_TICK_OFFSET_UM)
    cy = (ticks[0][2] + ticks[0][3]) / 2.0
    assert h / 2.0 + cy == pytest.approx(ID_TICK_OFFSET_UM)
    # ...and that is what the plate writer actually emits.
    plate = bench_marks("front", h, h)
    assert np.allclose(plate[: len(ticks)], id_tick_rects(
        0, h, h, PLY, 0.0, band_offset_um=ID_TICK_OFFSET_UM))
    # And where that leaves them on the NEW build: the 1/4" fold reaches
    # 2.05 mm, so the tick is no longer under the copper — it sits in the blank
    # ring between the fold and the 3.6375 mm art rim. Outside the garland,
    # inside the rim, and visible unless the bead covers it (see the constant's
    # docstring). Pinned here so that fact cannot change silently.
    assert 2050.0 < ID_TICK_OFFSET_UM < 3637.5


def test_dice_ticks_stay_in_the_street():
    street = 1_000.0
    p = Placement("front", 1_000.0, 2_000.0, 20_000.0, 16_000.0, False)
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


# --- the request bodies ---------------------------------------------------------


def test_api_body_models_keep_carrier_mode_and_ignore_bonded():
    """``carrier_scale_mode`` must survive the request body — a regression once
    shipped a UI choice to the backend as the default.

    ``bonded`` must be IGNORED rather than fatal: the flag and its two-ply math
    went on 2026-09-16, but box manifests cached before then still carry it and
    reading one back must not raise."""
    from app.api.boxes import BoxSpecBody
    from app.api.plates import PlateSpecBody
    from app.boxes import BoxSpec

    assert not hasattr(BoxSpecBody(), "bonded")
    assert not hasattr(BoxSpec.from_dict({"bonded": True}), "bonded")
    p = PlateSpecBody(pattern_slug="monogram-jp", carrier_scale_mode="fixed").to_spec()
    assert p.carrier_scale_mode == "fixed"
    assert PlateSpecBody(pattern_slug="monogram-jp").to_spec().carrier_scale_mode == "gap"
