"""Wave-optics sim kernels (Tier 2 Fraunhofer, Tier 3 angular spectrum).

Deliberately import-light: the numpy/FFT kernels live in the submodules, which
``api/sim.py`` imports lazily. Only the memory budget guard lives here.
"""

from __future__ import annotations

# A complex FFT costs roughly 48 B per grid cell at peak: the complex64 field
# plus its ifftshift copy (8 B each), the complex128 array numpy's fft2 always
# returns regardless of input dtype, and a full fftshift copy of that (16 B
# each). 4M cells is therefore already ~200 MB per FFT — and this host
# bugchecks under memory pressure long before swap saves it.
MAX_FFT_CELLS = 4_000_000
# Composed RGB atlases are float32 x 3 channels plus a uint8 copy (~15 B/px).
MAX_ATLAS_CELLS = 8_000_000


def check_fft_budget(
    n_cells: int, what: str, cap: int = MAX_FFT_CELLS, **dials: float
) -> None:
    """Refuse sim grids whose FFTs would not fit in memory.

    Same contract as ``patterns._helpers.check_lattice_budget``: raise
    ValueError with an actionable message naming the dials that drove the
    size, which the /sim routes surface as an HTTP 400. Unlike the pattern
    lattice this cannot be bounded by inspecting the request alone — the
    variant raster feeds the grid too — so both endpoints carry a
    ``downsample`` dial that scales it down.
    """
    if n_cells <= cap:
        return
    knobs = ", ".join(f"{k}={v:g}" for k, v in dials.items())
    raise ValueError(
        f"{what} would allocate {n_cells:,} cells ({knobs}); the cap is "
        f"{cap:,}. Raise downsample, lower n_angles, or ask for fewer "
        "wavelengths / view angles."
    )
