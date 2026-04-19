from __future__ import annotations

import math

from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from .._helpers import crop, ensure_multipolygon


def terrace_envelope(
    extent_um: tuple[float, float] | float,
    band_height_um: float,
    duty: float = 0.55,
    phase: float = 0.0,
) -> MultiPolygon:
    """Horizontal envelope bands evoking Eje Cafetero mountain terraces.

    Opaque bands of height `band_height_um * duty`, used to modulate a fine
    iridescent grating so the iridescence only shows on the 'terraces'.
    """
    if isinstance(extent_um, (int, float)):
        extent_um = (float(extent_um), float(extent_um))
    w, h = extent_um
    bands: list[Polygon] = []
    n = int(h / band_height_um) + 3
    stripe_h = band_height_um * duty
    for i in range(-n, n + 1):
        cy = (i + phase) * band_height_um
        # Slightly sinusoidally curved bands — evokes terrace contours
        samples = 40
        top = []
        bot = []
        for s in range(samples + 1):
            t = s / samples
            x = -w / 2 + t * w
            wobble = 0.2 * stripe_h * math.sin(2 * math.pi * t * 1.5 + i * 0.4)
            top.append((x, cy + stripe_h / 2 + wobble))
            bot.append((x, cy - stripe_h / 2 + wobble))
        bands.append(Polygon(top + list(reversed(bot))))
    mp = unary_union(bands)
    return crop(ensure_multipolygon(mp), extent_um)
