"""Cattleya orchid — Colombia's national flower.

Redrawn for a clearer species read: a strongly BILATERAL cattleya rather than a
radial daisy. Structure top-to-bottom:
  * 3 slender SEPALS (one erect, two lower-lateral) — the narrow points.
  * 2 broad, ruffled PETALS flanking the throat — the wide "wings".
  * 1 large frilled LABELLUM (lip) hanging below, the dominant lobe with a
    trumpet throat — the unmistakable cattleya signature.

Anchor at the flower center; the labellum hangs toward -y so, placed on a vine
tip, the bloom faces outward with its lip drooping.
"""
from __future__ import annotations

import math

from ...geometry import Mulberry32
from ...pen import Pen


def _blade(pen: Pen, length: float, width: float, tip_taper: float = 0.5) -> None:
    """A pointed petal/sepal blade along +x, tip at +length."""
    pen.move_to(0.0, 0.0)
    pen.bezier_to(length * 0.30, width * 0.62, length * 0.75, width * tip_taper, length, 0.0)
    pen.bezier_to(length * 0.75, -width * tip_taper, length * 0.30, -width * 0.62, 0.0, 0.0)
    pen.close_path()
    pen.fill_path()


def draw(pen: Pen, size: float, seed: int) -> None:
    rng = Mulberry32(seed)

    # --- 3 slender sepals: erect + two lower-lateral (narrow, pointed) -------
    sepal_len = size * 0.62
    sepal_w = size * 0.14
    for angle_deg in (90.0, 215.0, 325.0):
        pen.save()
        pen.rotate(math.radians(angle_deg + rng.uniform(-5.0, 5.0)))
        _blade(pen, sepal_len, sepal_w, tip_taper=0.35)
        pen.restore()

    # --- 2 broad ruffled petals (the wide wings) flanking the throat ---------
    petal_len = size * 0.48
    petal_w = size * 0.40
    for angle_deg in (150.0, 30.0):
        pen.save()
        pen.rotate(math.radians(angle_deg))
        # Broad, slightly ruffled wing: fuller belly than a sepal.
        pen.move_to(0.0, 0.0)
        pen.bezier_to(petal_len * 0.22, petal_w * 0.75, petal_len * 0.66, petal_w * 0.80, petal_len, petal_w * 0.18)
        pen.bezier_to(petal_len * 0.92, petal_w * 0.02, petal_len * 0.92, -petal_w * 0.02, petal_len, -petal_w * 0.18)
        pen.bezier_to(petal_len * 0.66, -petal_w * 0.80, petal_len * 0.22, -petal_w * 0.75, 0.0, 0.0)
        pen.close_path()
        pen.fill_path()
        pen.restore()

    # --- Labellum (lip): the big frilled trumpet hanging below ---------------
    pen.save()
    pen.rotate(math.radians(270.0))
    lip_len = size * 0.66
    lip_w = size * 0.46
    # Frilled outer margin: a wide fan with three soft ruffle lobes, opening
    # from a narrow throat at the base to a broad frilled rim.
    pen.move_to(0.0, 0.0)
    pen.bezier_to(lip_len * 0.12, lip_w * 0.30, lip_len * 0.30, lip_w * 0.42, lip_len * 0.48, lip_w * 0.40)
    pen.bezier_to(lip_len * 0.66, lip_w * 0.62, lip_len * 0.86, lip_w * 0.44, lip_len, lip_w * 0.16)  # ruffle
    pen.bezier_to(lip_len * 1.02, lip_w * 0.02, lip_len * 1.02, -lip_w * 0.02, lip_len, -lip_w * 0.16)
    pen.bezier_to(lip_len * 0.86, -lip_w * 0.44, lip_len * 0.66, -lip_w * 0.62, lip_len * 0.48, -lip_w * 0.40)
    pen.bezier_to(lip_len * 0.30, -lip_w * 0.42, lip_len * 0.12, -lip_w * 0.30, 0.0, 0.0)
    pen.close_path()
    pen.fill_path()
    # Throat shadow line down the lip center.
    pen.move_to(lip_len * 0.05, 0.0)
    pen.line_to(lip_len * 0.75, 0.0)
    pen.stroke_path(size * 0.02)
    pen.restore()

    # Column / throat knot at the center.
    pen.circle(0.0, -size * 0.02, size * 0.07, fill=True)
