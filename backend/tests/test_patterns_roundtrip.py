"""Smoke + geometric round-trip tests: every registered pattern must generate,
produce non-empty polygons, and rasterize to a valid non-blank image at the
advertised pitch."""
from __future__ import annotations

import numpy as np
from PIL import Image

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
        "sombrero-vueltiao-parallax",
        "cafetero-iridescence",
        "colibri-hologram",
        "muzo-emerald-zone",
        "tairona-talbot",
        "caravel-latent",
        "compass-rose-spiral",
        "meridian-speckle",
    }
    got = set(registry.keys())
    missing = expected - got
    assert not missing, f"Missing patterns: {missing}"


def test_every_pattern_generates_and_rasterizes():
    for slug, cls in registry.items():
        gp = cls.generate(**cls.defaults())
        # Allow exactly one side to be empty (single-layer patterns like
        # colibri-hologram / muzo-emerald-zone), but not both.
        assert not (gp.front.is_empty and gp.back.is_empty), f"{slug}: both layers empty"
        img = rasterize(gp.front if not gp.front.is_empty else gp.back, gp.extent_um, gp.pixel_pitch_um)
        arr = np.asarray(img)
        assert arr.size > 0
        # Should have some non-zero pixels (gold present)
        assert arr.max() > 0, f"{slug}: rasterized image is entirely blank"
