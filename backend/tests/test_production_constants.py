"""app/production.py is the ONE holder of the process / stock / box constants.

Two things are pinned here:

  * the generated ``frontend/src/production.ts`` is CURRENT — rendered into a
    temp path and compared byte for byte, the same trick
    ``test_assembly.py::test_golden_fixture_matches_backend`` plays on the
    assembly golden fixture. A backend constant change that forgets to
    regenerate fails the backend suite instead of leaving the preview quietly
    disagreeing with the plate;
  * no module re-defines a constant it should import. The duplicates this
    module replaced were real drift risk: the litho floor lived under three
    names, the DBU under three, the layer map under four, and
    ``witness_dies.blank_plan`` carried a runtime assert whose only job was to
    catch the glass copies diverging.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from app import boxes, export_fine, export_witness, ply_cuts, production, witness_dies, witness_geom
from app.patterns import base as pattern_base

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "tools" / "dev" / "gen_production_constants.py"
TS = REPO / "frontend" / "src" / "production.ts"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_production_constants", GEN)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_generated_production_ts_is_current(tmp_path: Path) -> None:
    gen = _load_generator()
    out = tmp_path / "production.ts"
    gen.write(out)
    assert TS.exists(), f"{TS} is missing — run {GEN.name}"
    # Text mode on both sides: the repo checks out CRLF on Windows, and the
    # line ending is not part of the contract. Everything else is.
    fresh = out.read_text(encoding="utf-8")
    checked_in = TS.read_text(encoding="utf-8")
    assert checked_in == fresh, (
        "frontend/src/production.ts is STALE. Regenerate it in the same commit:\n"
        "  uv run --directory backend python ../tools/dev/gen_production_constants.py"
    )


def test_one_definition_per_constant() -> None:
    """Every module that used to own a copy now reads production's object.

    ``is`` rather than ``==``: a re-typed 2.0 would compare equal and is
    exactly the drift this module exists to prevent (floats this small are
    interned by identity only when they are literally the same object)."""
    assert export_fine.LITHO_FLOOR_UM is production.LITHO_FLOOR_UM
    assert pattern_base.LITHO_FLOOR_UM is production.LITHO_FLOOR_UM
    assert witness_dies.LITHO_FLOOR_UM is production.LITHO_FLOOR_UM
    assert export_fine.DBU_UM is production.DBU_UM
    assert export_witness.DBU_UM is production.DBU_UM
    assert witness_geom.PLY_UM is production.PLY_UM
    assert witness_geom.GLASS_N is production.GLASS_N
    assert witness_dies.PLY_UM is production.PLY_UM
    assert ply_cuts.ID_TICK_OFFSET_UM is production.ID_TICK_OFFSET_UM
    # The layer map, once four tuples in three modules.
    assert witness_geom.LAYER_GOLD is production.LAYER_GOLD
    assert export_witness.LAYER_GOLD is production.LAYER_GOLD
    assert export_witness.LAYER_DICE is production.LAYER_DICE
    assert export_witness.LAYER_OUTLINE is production.LAYER_OUTLINE


def test_default_box_is_built_from_production_constants() -> None:
    spec = boxes.default_box_spec()
    assert spec.width_um == production.WIDTH_UM
    assert spec.depth_um == production.DEPTH_UM
    assert spec.height_um == production.HEIGHT_UM
    assert spec.glass.thickness_um == production.PLY_UM
    assert spec.glass.n == production.GLASS_N
    assert spec.glass.material == production.GLASS_MATERIAL
    assert spec.foil.tape_width_um == production.TAPE_UM
    assert spec.art_rim_um == production.ART_RIM_UM
    for fid, face in spec.faces.items():
        assert face.frame.motif_scale == production.MOTIF_SCALE, fid
        assert face.frame.band_um == production.BAND_UM, fid


def test_pinned_values_are_the_written_plate() -> None:
    """The constants the 2026-09-15 mask was WRITTEN with. A change to any of
    these is a mask change; the witness rebuild is the gate, not this test."""
    assert production.ART_RIM_UM == 3637.5
    assert production.ID_TICK_OFFSET_UM == 2943.75
    assert production.MOTIF_SCALE == 0.68
    assert production.BAND_UM == 2400.0
    assert (production.WIDTH_UM, production.DEPTH_UM, production.HEIGHT_UM) == (
        32000.0, 32000.0, 35000.0)
    assert production.LITHO_FLOOR_UM == 2.0
    assert production.FINISH_RADIUS_UM == 1.0
    assert production.DBU_UM == 0.001
    assert production.POLARITY == "clear"
    # The die finish is an OPEN of radius r, which deletes every line under 2r,
    # so it can never be coarser than half the floor without eating legal gold.
    assert production.FINISH_RADIUS_UM == pytest.approx(production.LITHO_FLOOR_UM / 2.0)
    # 1/4" tape on a single ply: a 2.05 mm fold, well inside the art rim.
    fold = (production.TAPE_UM - production.PLY_UM) / 2.0
    assert fold == pytest.approx(2050.0)
    assert fold < production.ART_RIM_UM
    # The pinned ID tick sits OUTSIDE that fold and inside the art rim — the
    # open question in project-cleanup-resume, pinned here so it stays visible.
    assert fold < production.ID_TICK_OFFSET_UM < production.ART_RIM_UM
