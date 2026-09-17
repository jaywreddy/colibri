"""Fine-pitch grating & diffraction toolkit.

Exploits the 2 um lithography floor (min line 2 um, min gap 2 um -> min grating
period 4 um) to build designable optical effects. Since the box became six
single written plies (2026-09-16) that means ONE effect: spectral colour from a
fine grating's first order, mapped by period — the region_art centrepieces, the
photo colour zones and the garland's per-family leaf gratings.

Three submodules:
  * ``gratings`` — the few grating helpers the emitters still share: the
    row-span baker (``bool_to_row_spans``), the interleave band selector, the
    beat-delta closed form and the two-ply shimmer pair used by the moiré
    exemplar. The generator family (linear/radial/chirped/checker masks, the
    diffraction accent) went with the catalogue on 2026-09-16.
  * ``moire``    — the closed forms (beat period, diffraction onset) and the
    litho floor every other module imports rather than re-declares.
  * ``drc``      — the design-rule check and the metal heal the mask writer runs.
"""
from __future__ import annotations

from .gratings import band_select, beat_delta_um, bool_to_row_spans, shimmer_moire_layers
from .moire import (
    DIFFRACTION_PERIOD_UM,
    MIN_PERIOD_UM,
    beat_period_parallel,
    beat_period_rotated,
    diffraction_onset,
)

__all__ = [
    # gratings
    "band_select",
    "beat_delta_um",
    "bool_to_row_spans",
    "shimmer_moire_layers",
    # physics
    "beat_period_parallel",
    "beat_period_rotated",
    "diffraction_onset",
    "DIFFRACTION_PERIOD_UM",
    "MIN_PERIOD_UM",
]
