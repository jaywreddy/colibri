"""Perimeter math + deterministic PRNG + value noise for frame generators.

Coordinate convention: every algorithm here works in the plate's own μm space
with the origin at the plate center. A ``RectFrame`` describes the axis-aligned
rectangle the decoration must hug; ``dist_frame`` and ``perim_info`` are the
two primitives algorithms use to stay anchored to that rectangle.

The PRNG (mulberry32) and value-noise functions mirror the engraving spec so
the Python generator and the Canvas2D preview can in principle agree on
output for the same seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RectFrame:
    """Axis-aligned rectangle the decoration hugs, in μm, centered at origin."""

    width_um: float
    height_um: float

    @property
    def left(self) -> float:
        return -self.width_um / 2

    @property
    def right(self) -> float:
        return self.width_um / 2

    @property
    def bottom(self) -> float:
        return -self.height_um / 2

    @property
    def top(self) -> float:
        return self.height_um / 2

    @property
    def perimeter(self) -> float:
        return 2.0 * (self.width_um + self.height_um)

    def dist_frame(self, x: float, y: float) -> float:
        """Distance from (x, y) to the nearest edge of the rectangle.

        Negative distances are *outside* the rect; positive distances are
        inside. Algorithms scatter attractors where ``0 < dist < band``.
        """
        # Signed distance from each edge interior (positive when inside on
        # that side); the overall distance to the rectangle's boundary
        # equals the minimum.
        d_left = x - self.left
        d_right = self.right - x
        d_bottom = y - self.bottom
        d_top = self.top - y
        return min(d_left, d_right, d_bottom, d_top)

    def perim_info(self, d: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Arc-length parametrization of the perimeter.

        Walks clockwise starting at the bottom-left corner. Returns
        ``((x, y), (tx, ty), (nx, ny))`` — the point on the perimeter at
        arc-length ``d``, the unit tangent in the walk direction, and the
        unit inward normal.

        ``d`` wraps modulo ``perimeter`` so callers don't have to guard.
        """
        p = self.perimeter
        d = d % p if p > 0 else 0.0
        w = self.width_um
        h = self.height_um

        # Edge 0 — bottom, left -> right
        if d <= w:
            x = self.left + d
            y = self.bottom
            return (x, y), (1.0, 0.0), (0.0, 1.0)
        d -= w
        # Edge 1 — right, bottom -> top
        if d <= h:
            x = self.right
            y = self.bottom + d
            return (x, y), (0.0, 1.0), (-1.0, 0.0)
        d -= h
        # Edge 2 — top, right -> left
        if d <= w:
            x = self.right - d
            y = self.top
            return (x, y), (-1.0, 0.0), (0.0, -1.0)
        d -= w
        # Edge 3 — left, top -> bottom
        x = self.left
        y = self.top - d
        return (x, y), (0.0, -1.0), (1.0, 0.0)

    def in_band(self, x: float, y: float, band_um: float) -> bool:
        """True iff the point sits inside the rect and within ``band_um`` of an edge."""
        d = self.dist_frame(x, y)
        return 0.0 <= d <= band_um


# ----- seeded PRNG (mulberry32) ----------------------------------------------


class Mulberry32:
    """32-bit deterministic PRNG matching the spec's reference implementation.

    Kept stateful (instance-per-seed) so independent generator components can
    each carry their own stream without cross-contamination. ``next_float()``
    returns a float in [0, 1).
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        # Mask to 32 bits and shift so 0 produces a non-degenerate stream.
        self._state = (int(seed) & 0xFFFFFFFF) or 0xDEADBEEF

    def next_uint32(self) -> int:
        self._state = (self._state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self._state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF
        return (t ^ (t >> 14)) & 0xFFFFFFFF

    def next_float(self) -> float:
        return self.next_uint32() / 4294967296.0

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.next_float()

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive at both ends."""
        if hi < lo:
            lo, hi = hi, lo
        return lo + int(self.next_float() * (hi - lo + 1))

    def choice(self, seq):
        return seq[int(self.next_float() * len(seq))]


# ----- 2D value noise --------------------------------------------------------

_NOISE_GRID = 256


class ValueNoise2D:
    """Bilinear-interpolated value noise on a 256×256 grid of seeded random values.

    Cheap, smooth, and deterministic from a single seed. Used by the flowing-
    border algorithm and the colonization wiggle term.
    """

    __slots__ = ("_grid",)

    def __init__(self, seed: int) -> None:
        rng = Mulberry32(seed)
        n = _NOISE_GRID
        self._grid = [rng.next_float() for _ in range(n * n)]

    def at(self, x: float, y: float) -> float:
        n = _NOISE_GRID
        xi = math.floor(x)
        yi = math.floor(y)
        fx = x - xi
        fy = y - yi
        x0 = xi % n
        y0 = yi % n
        x1 = (x0 + 1) % n
        y1 = (y0 + 1) % n
        a = self._grid[y0 * n + x0]
        b = self._grid[y0 * n + x1]
        c = self._grid[y1 * n + x0]
        d = self._grid[y1 * n + x1]
        # Smoothstep on the fractional offsets — gives C¹-ish interp.
        ux = fx * fx * (3.0 - 2.0 * fx)
        uy = fy * fy * (3.0 - 2.0 * fy)
        ab = a + (b - a) * ux
        cd = c + (d - c) * ux
        return ab + (cd - ab) * uy

    def fbm(self, x: float, y: float, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5) -> float:
        """Sum of octaves; result roughly in [0, 1]."""
        amp = 1.0
        freq = 1.0
        total = 0.0
        norm = 0.0
        for _ in range(octaves):
            total += amp * self.at(x * freq, y * freq)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / max(norm, 1e-9)
