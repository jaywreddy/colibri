"""Fine-pitch grating & diffraction toolkit.

Exploits the 2 um lithography floor (min line 2 um, min gap 2 um -> min grating
period 4 um) to build designable optical effects. Since the box became six
single written plies (2026-09-16) that means ONE effect: spectral colour from a
fine grating's first order, mapped by period — the region_art centrepieces, the
photo colour zones and the garland's per-family leaf gratings.

Three submodules:
  * ``gratings`` — parameterized grating GENERATORS. Each returns a numpy bool
    mask (1 = gold) AND a rect-array (row-span) baker for the fab/SVG path.
    All are budget-aware: given an extent and period they compute the element
    count and refuse (or coarsen) below the 400k-lattice cap.
  * ``moire``    — the closed forms (beat period, diffraction onset) and the
    litho floor every other module imports rather than re-declares.
  * ``drc``      — the design-rule check and the metal heal the mask writer runs.
"""
from __future__ import annotations

from .gratings import (
    GratingResult,
    bool_to_row_spans,
    checker_grating,
    chirped_grating,
    grating_line_budget,
    linear_grating_mask,
    radial_grating_mask,
    row_spans_to_verts,
)
from .moire import (
    DIFFRACTION_PERIOD_UM,
    MIN_PERIOD_UM,
    beat_period_parallel,
    beat_period_rotated,
    diffraction_onset,
)

__all__ = [
    # gratings
    "GratingResult",
    "linear_grating_mask",
    "radial_grating_mask",
    "chirped_grating",
    "checker_grating",
    "bool_to_row_spans",
    "row_spans_to_verts",
    "grating_line_budget",
    # physics
    "beat_period_parallel",
    "beat_period_rotated",
    "diffraction_onset",
    "DIFFRACTION_PERIOD_UM",
    "MIN_PERIOD_UM",
]
