"""The registered centrepiece patterns.

THREE slugs (2026-09-16). ``monogram-jp`` and ``globe-atlantic`` are the two
written centrepieces of the production box — single-layer diffraction mappings,
colour by region. ``globe-duo-phase`` is the hidden two-ply exemplar: a parallax
barrier switch, kept so the construction (and its registration) stays built and
tested even though no face carries one.

The rest of the catalogue — the lenticulars, the carrier reveal, the scanimation,
the shimmer moires and the lab motifs behind them — went with the two-ply design.
They are in git history at 22d1634.
"""
from __future__ import annotations

from . import globe_atlantic, globe_duo_phase, monogram_jp

__all__ = ["globe_atlantic", "globe_duo_phase", "monogram_jp"]
