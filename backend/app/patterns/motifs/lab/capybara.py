from __future__ import annotations

"""Resting capybara — side profile, serene, facing left.

The capybara is the couple's beloved animal, so the read has to be unmistakable.
Rather than hand-drawing the outline (which came out cartoony), this motif is
**traced from a CC0 PhyloPic silhouette** (Skye M, Hydrochoerus hydrochaeris) via
``motifs/trace.py`` — real capybara anatomy: the blocky squared MUZZLE, the tiny
rounded EAR set high and well back, the deep barrel BODY, short stubby LEGS, and
no tail. See ``lab/refs/LICENSE_NOTE.md`` for the source + license.

Composition: the animal is dropped LOW in the art box (belly ≈ y 0.82, feet ≈
0.89) so the ``capybara-scanimation`` waterline at y≈0.66 half-submerges the
lower barrel and legs while the head and back stay dry — a serene "wading" read.
A single calm eye is punched as engraved negative space so it reads as an animal,
not a blob. Authored in the normalized 0..1 art box (Pillow y-down), rasterized
to a bool grid (True == gold), scale-free (the caller controls ``cell_um``).
"""

from pathlib import Path

import numpy as np

from ..trace import TraceParams, trace_silhouette

# CC0 reference silhouette (already faces LEFT). See lab/refs/LICENSE_NOTE.md.
_REF = Path(__file__).parent / "refs" / "capybara_skye_source.png"

# Deterministic trace recipe — tuned by vision iteration (r1/r2) so the animal
# reads naturalistic, edges are clean, no background bleed, and the thinnest leg
# clears the litho min-feature floor.
_PARAMS = TraceParams(
    use_alpha=True,        # PhyloPic PNG carries a clean alpha silhouette
    flood_bg=True,
    fill_holes=True,
    min_speck_frac=0.002,
    min_feature_px=3,      # legs must clear the litho floor
    max_thicken_px=6,      # blob guard
    fill=0.94,             # subject fills 94% of the larger art-box dim
    anchor_x=0.5,
    anchor_y=0.62,         # drop low -> belly ≈ 0.82, feet ≈ 0.89
    flip_h=False,          # reference already faces left
    holes=(
        # calm eye: small dot high on the head, set back from the muzzle.
        (0.205, 0.375, 0.013, 0.015),
    ),
)


def capybara_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — resting/wading capybara facing left.

    Scale-free: ``extent_um`` is accepted for signature parity with the other
    centerpiece motifs and ignored (the caller picks ``cell_um`` to hit the
    physical extent). The mask is traced deterministically from the vendored CC0
    reference, so it bakes identically every run.
    """
    del extent_um  # scale-free; caller controls cell_um to hit the extent
    return trace_silhouette(_REF, n_grid=n_grid, params=_PARAMS)
