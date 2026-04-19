"""Every registered pattern must declare a valid tier and theme.

This guard prevents the 'forgot to set theme on a new pattern' class of bug —
the Gallery UI groups by theme, so a missing theme would silently hide cards.
"""
from __future__ import annotations

import app.patterns.artistic  # noqa: F401  -- populate registry
from app.patterns.base import registry


VALID_TIERS = {1, 2, 3}
VALID_THEMES = {"Colombia", "Global Travel"}


def test_every_pattern_has_valid_tier():
    for slug, cls in registry.items():
        assert getattr(cls, "tier", None) in VALID_TIERS, (
            f"{slug}: tier={cls.tier!r} not in {VALID_TIERS}"
        )


def test_every_pattern_has_valid_theme():
    for slug, cls in registry.items():
        assert getattr(cls, "theme", None) in VALID_THEMES, (
            f"{slug}: theme={cls.theme!r} not in {VALID_THEMES}"
        )


def test_colombia_section_has_at_least_seven_patterns():
    colombia = [s for s, c in registry.items() if c.theme == "Colombia"]
    assert len(colombia) >= 7, f"expected ≥7 Colombia patterns, got {colombia}"


def test_descriptor_emits_tier_and_theme():
    cls = next(iter(registry.values()))
    d = cls.descriptor()
    assert d["tier"] in VALID_TIERS
    assert d["theme"] in VALID_THEMES
