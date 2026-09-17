"""Smoke + geometric round-trip tests: every registered pattern must generate,
produce non-empty polygons, and rasterize to a valid non-blank image at the
advertised pitch."""
from __future__ import annotations

import numpy as np

from app.patterns import base  # noqa: F401  -- ensure registry is populated
from app.patterns.base import registry
from app.rasterize import rasterize

# Importing the patterns package triggers every @register. __init__.py does this,
# but be defensive in case the import graph changes.
import app.patterns.artistic  # noqa: F401


def test_all_patterns_registered():
    """The catalogue IS the box (2026-09-16), plus one hidden exemplar.

    Four written production faces — the lid's monogram and the front's globe as
    single-layer diffraction mappings, the photo screen the three sides share,
    and the solid gold base — plus bare glass, plus ``globe-duo-phase``, the
    parallax-barrier switch kept so the two-ply construction stays built and
    tested. Nothing else registers; a slug appearing here that no face uses is
    how the catalogue grew to twenty-one in the first place."""
    expected = {
        # written production faces
        "monogram-jp",
        "globe-atlantic",
        "photo-halftone",
        "solid-gold",
        # bare glass (the back and, before 2026-09-15, the bottom)
        "blank",
        # hidden two-ply exemplar: the parallax barrier switch
        "globe-duo-phase",
    }
    got = set(registry.keys())
    assert got == expected, f"registry mismatch: got {got}, expected {expected}"


def test_every_pattern_generates_and_rasterizes():
    for slug, cls in registry.items():
        gp = cls.generate(**cls.defaults())
        if slug == "blank":
            # bare glass by definition: both layers empty is the contract
            assert gp.front.is_empty and gp.back.is_empty
            continue
        assert not (gp.front.is_empty and gp.back.is_empty), f"{slug}: both layers empty"
        img = rasterize(
            gp.front if not gp.front.is_empty else gp.back,
            gp.extent_um,
            gp.pixel_pitch_um,
        )
        arr = np.asarray(img)
        assert arr.size > 0
        assert arr.max() > 0, f"{slug}: rasterized image is entirely blank"


def test_the_exemplar_switch_is_a_barrier():
    """``globe-duo-phase`` is the one image-switch construction left, and it must
    be a PARALLAX BARRIER: both views interlaced in the BACK layer under a slit
    comb in FRONT (CLAUDE.md's image-switch rule — a front-layer image cannot
    vanish under parallax, because the front mask does not move with tilt). So
    it ships the slit metadata and both interlaced view layers.

    That it actually EXTINGUISHES the off-channel image on the real generated
    geometry is measured in tests/test_barrier_registration.py."""
    cls = registry["globe-duo-phase"]
    assert cls.render_recipe == "stereo_lenticular"
    gp = cls.generate(**cls.defaults())
    assert "slit_period_um" in gp.recipe_data
    assert float(gp.recipe_data["slit_period_um"]) > 0
    assert "view_a" in gp.extra_layers and not gp.extra_layers["view_a"].is_empty
    assert "view_b" in gp.extra_layers and not gp.extra_layers["view_b"].is_empty
