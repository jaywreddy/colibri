"""Fine-pitch grating & moiré effects toolkit.

Exploits the 2 µm lithography floor (min line 2 µm, min gap 2 µm → min grating
period 4 µm) on a 500 µm fused-silica substrate (n=1.46) to build designable
optical effects: moiré shimmer, phase-switch tilt art, moiré magnification,
scanimation (barrier-grid animation), and controlled diffraction-flash accents.

Three submodules:
  * ``gratings`` — parameterized grating GENERATORS. Each returns a numpy bool
    mask (1 = gold) AND a rect-array (row-span) baker for the fab/SVG path.
    All are budget-aware: given an extent and period they compute the element
    count and refuse (or coarsen) below the 400k-lattice cap.
  * ``moire``   — the PHYSICS calculators (beat period, magnification factor,
    phase-switch tilt angle, scanimation motion, diffraction onset). Pure,
    documented, unit-consistent (µm everywhere).
  * ``demos``   — side-by-side demo renderers used by the throwaway runner to
    validate the effects visually across carrier periods 20/12/8/6/4 µm.

Nothing here touches ``plates.py`` or the shaders — the integrator wires it.
The API the integrator should call is re-exported below.
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
    PARALLAX_UM_PER_DEG,
    SUBSTRATE_N,
    SUBSTRATE_T_UM,
    beat_period_parallel,
    beat_period_rotated,
    diffraction_onset,
    moire_magnification,
    parallax_shift_um,
    phase_switch_tilt_deg,
    scanimation_plan,
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
    # moire physics
    "beat_period_parallel",
    "beat_period_rotated",
    "moire_magnification",
    "phase_switch_tilt_deg",
    "scanimation_plan",
    "diffraction_onset",
    "parallax_shift_um",
    "PARALLAX_UM_PER_DEG",
    "SUBSTRATE_T_UM",
    "SUBSTRATE_N",
]
