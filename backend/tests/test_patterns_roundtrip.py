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
    expected = {
        "wayuu-kanasu-moire",
        "emerald-facet-moire",
        "colibri-globe-lenticular",
        "colibri-globe-moire",
        "colibri-globe-phase",
    }
    got = set(registry.keys())
    assert got == expected, f"registry mismatch: got {got}, expected {expected}"


def test_every_pattern_generates_and_rasterizes():
    for slug, cls in registry.items():
        gp = cls.generate(**cls.defaults())
        assert not (gp.front.is_empty and gp.back.is_empty), f"{slug}: both layers empty"
        img = rasterize(
            gp.front if not gp.front.is_empty else gp.back,
            gp.extent_um,
            gp.pixel_pitch_um,
        )
        arr = np.asarray(img)
        assert arr.size > 0
        assert arr.max() > 0, f"{slug}: rasterized image is entirely blank"


def test_lenticular_pattern_has_view_a_view_b_layers():
    """colibri-globe-lenticular must ship extra_layers for view_a and view_b
    so the shader's stereo_lenticular path has scenes to interlace."""
    cls = registry["colibri-globe-lenticular"]
    gp = cls.generate(**cls.defaults())
    assert "view_a" in gp.extra_layers and not gp.extra_layers["view_a"].is_empty
    assert "view_b" in gp.extra_layers and not gp.extra_layers["view_b"].is_empty


def test_phase_pattern_reports_carrier_in_recipe_data():
    """colibri-globe-phase carries a carrier_period_um so the shader knows
    how to scale parallax-driven phase flips."""
    cls = registry["colibri-globe-phase"]
    gp = cls.generate(**cls.defaults())
    assert "carrier_period_um" in gp.recipe_data
    assert float(gp.recipe_data["carrier_period_um"]) > 0
