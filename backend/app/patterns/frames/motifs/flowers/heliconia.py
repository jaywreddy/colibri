"""Heliconia (lobster-claw): a bold pendent zigzag of sharp keeled bracts.

*Heliconia rostrata* is a dramatic hanging inflorescence: a nearly vertical
rachis from which large, sharply beaked, boat-shaped bracts jut alternately
left and right in a descending zigzag, each a clean triangular claw. We build it
along +x (anchor/petiole at the origin) so the colonize placement — which
orients a motif along its vine-tip direction — hangs the claw off the branch.

The read we're after: a straight-ish stalk with 5-6 crisp, sharply pointed
claw bracts stepping down it, alternating sides, clearly separated — not a
cluster of rounded leaves.
"""
from __future__ import annotations

import math

from ...geometry import Mulberry32
from ...pen import Pen


def _bract(pen: Pen, length: float, depth: float) -> None:
    """One sharp keeled claw bract along +x, beak at +length, keel on -y.

    Triangular boat: a nearly straight top edge to a sharp beak, then a deep
    convex keel belly sweeping back to the base — the lobster-claw segment.
    """
    pen.move_to(0.0, 0.0)
    # Top edge — FLAT ridge to the sharp beak (the claw's straight upper line).
    pen.bezier_to(length * 0.5, depth * 0.02, length * 0.82, depth * 0.02, length, 0.0)
    # Keel belly — deep convex curve back to base (the boat/claw underside).
    pen.bezier_to(length * 0.62, -depth * 0.98, length * 0.22, -depth * 1.02, 0.0, 0.0)
    pen.close_path()
    pen.fill_path()


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)
    stroke = size * 0.026

    n_bracts = 6
    # Rachis: mostly vertical (down +? ) but expressed along +x with a gentle
    # dip so the whole claw hangs. Anchor at origin, tip descending.
    rachis_len = size * 1.10
    pen.move_to(0.0, 0.0)
    pen.quadratic_to(rachis_len * 0.5, -size * 0.06, rachis_len, -size * 0.16)
    pen.stroke_path(stroke * 1.4)

    for i in range(n_bracts):
        frac = i / (n_bracts - 1) if n_bracts > 1 else 0.0
        rx = rachis_len * (0.05 + 0.92 * frac)
        ry = -size * (0.06 + 0.12 * frac * frac)
        side = -1 if (i % 2 == 0) else 1
        # Bracts taper as the claw descends.
        blen = size * (0.72 - 0.34 * frac) * (0.94 + 0.10 * rng.next_float())
        bdepth = blen * 0.46
        # Steep hang: bracts jut nearly perpendicular to the stalk, angled a
        # little forward toward the descending tip.
        base_deg = 232.0 if side < 0 else 128.0
        fwd = 16.0 * frac * side
        pen.save()
        pen.translate(rx, ry)
        pen.rotate(math.radians(base_deg + fwd))
        _bract(pen, blen, bdepth)
        pen.restore()
