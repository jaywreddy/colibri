"""Single-layer DIFFRACTION centrepieces: the region-art contract.

The box is ONE written ply per face (2026-09-15: the bonded moiré pairs were
dropped after the first plate — the moirés read badly on glass and the pair
could not be cleaved; every face is now a single-layer diffraction mapping,
like the photograph sides always were). A centrepiece on such a face is not a
silhouette filled with a carrier but a MAP OF REGIONS, each region written as
its own fine 50 %-duty vertical grating whose PERIOD sets the colour it flashes
under a lamp (the physics of the photo colour zones and the garland's leaf
families: see ``patterns/bitmap/colourzone.py`` and ``leaf_fills.py``).

    labels   int32 (n, n) raster over the CENTERPIECE SQUARE (plates.CENTERPIECE_FILL
             of the aperture), y DOWN like every motif raster; 0 = bare glass
    regions  id -> Region: the period each label is written at (0 = solid gold)

Every consumer reads the SAME function through :func:`centerpiece_regions`:
the fine GDS bake (``export_fine.build_plate_fine``), the fab SVG bake and the
composed preview (``plates``). A motif registers its region function under the
face's pattern slug with :func:`register_centerpiece_regions`; the function
takes ``(n_px, params)`` and returns a :class:`RegionArt`.

Design rules the emitter enforces (so a motif author does not have to):

* all gratings are VERTICAL (angle 0). Colour comes from the PERIOD, which is
  view-independent between regions (a period ratio survives every tilt); an
  angle difference between regions does not (colourzone.py, "What actually
  controls the hue"). Vertical also means axis-aligned rectangles: one plate
  rectangle per line per run, no rotated vertices, no decomposition slivers.
* a region's period must clear the litho floor (line = p·duty ≥ 2 µm) AND the
  die finish (``production.FINISH_RADIUS_UM``: an open of radius r erases lines under
  2r) — the same guard the leaf families pass. The colour ladder the leaves use
  (``plates.SINGLE_PLY_LEAF_HUE_PERIODS_UM``, 4.15–6.02 µm) is the safe choice.
* regions of different periods are separated by a one-cell gutter at the zone
  pitch (20 µm of glass) wherever two METAL regions touch, so two gratings
  never meet in a sub-floor sliver; a region's edge against bare glass is not
  cut. Thin features (a graticule line, a star, a hairline serif) should still
  be SOLID (period 0), not gratings: a region narrower than ~2 periods has no
  spectrum to give.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Region:
    """One colour region of a single-layer centrepiece."""

    name: str
    period_um: float
    """Grating period (µm) of the vertical 50 % grating written here. ``0``
    means SOLID gold: no grating, plain specular metal."""
    duty: float = 0.5

    def __post_init__(self) -> None:
        if self.period_um < 0:
            raise ValueError(f"{self.name}: period_um must be >= 0 (0 = solid)")
        if not 0.05 <= self.duty <= 0.95:
            raise ValueError(f"{self.name}: duty must be 0.05..0.95")

    @property
    def solid(self) -> bool:
        return self.period_um <= 0.0


@dataclass
class RegionArt:
    """``labels`` (int32, square, y-down; 0 = glass) + ``regions`` (id -> Region)."""

    labels: np.ndarray
    regions: dict[int, Region] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels)
        if self.labels.ndim != 2 or self.labels.shape[0] != self.labels.shape[1]:
            raise ValueError("labels must be a square 2-D raster")
        self.labels = self.labels.astype(np.int32, copy=False)
        ids = set(int(v) for v in np.unique(self.labels)) - {0}
        missing = ids - set(self.regions)
        if missing:
            raise ValueError(f"labels carry ids with no Region: {sorted(missing)}")
        if 0 in self.regions:
            raise ValueError("region id 0 is reserved for bare glass")

    def resized(self, n_px: int) -> "RegionArt":
        """The same map at ``n_px`` (nearest-neighbour; labels are categorical)."""
        n_px = int(n_px)
        if n_px == self.labels.shape[0]:
            return self
        from PIL import Image

        im = Image.fromarray(self.labels.astype(np.int32), mode="I").resize(
            (n_px, n_px), Image.NEAREST
        )
        return RegionArt(np.asarray(im, dtype=np.int32), dict(self.regions))

    def period_lut(self) -> np.ndarray:
        """``lut[id] = period_um`` (0 for glass and for solid regions)."""
        n = int(self.labels.max()) + 1 if self.labels.size else 1
        lut = np.zeros(max(n, max(self.regions, default=0) + 1), dtype=np.float64)
        for rid, r in self.regions.items():
            lut[rid] = r.period_um
        return lut

    def metal(self) -> np.ndarray:
        """bool: where any gold is written (grating or solid)."""
        return self.labels > 0

    def coloured(self) -> np.ndarray:
        """bool: where a diffractive (non-solid) region is written."""
        return self.period_lut()[self.labels] > 0.0

    def describe(self) -> dict[str, Any]:
        n = float(max(1, self.labels.size))
        return {
            rid: {
                "name": r.name,
                "period_um": float(r.period_um),
                "duty": float(r.duty),
                "area_frac": round(float((self.labels == rid).sum()) / n, 4),
            }
            for rid, r in sorted(self.regions.items())
        }


REGION_ZONE_PITCH_UM = 20.0
"""Raster pitch (µm) the fine bake quantises a region map's BOUNDARIES at
(``export_fine._emit_region_art``): 0.23 arcmin at 300 mm, a quarter of the
eye's limit, so a letter edge or a coastline reads smooth and the one-cell
glass gutter between two regions is invisible. The periods inside are exact."""
REGION_ZONE_MAX_PX = 1400
"""Cap on the region raster's side (1.96 M cells of bool): the art box of the
32 mm lid is 17.4 mm, 870 px at the pitch above."""

RegionFn = Callable[[int, dict[str, Any]], RegionArt]

CENTERPIECE_REGION_FNS: dict[str, RegionFn] = {}
"""slug -> region function. Filled by :func:`register_centerpiece_regions`
from the motif modules (imported through ``app.patterns``)."""


def register_centerpiece_regions(slug: str) -> Callable[[RegionFn], RegionFn]:
    """Decorator: ``fn(n_px, params) -> RegionArt`` is the single-layer
    centrepiece of every face whose ``pattern_slug`` is ``slug``."""

    def deco(fn: RegionFn) -> RegionFn:
        CENTERPIECE_REGION_FNS[slug] = fn
        return fn

    return deco


def _ensure_registered() -> None:
    # The region functions live beside their motifs and register on import;
    # importing the pattern package is what runs those imports.
    if not CENTERPIECE_REGION_FNS:
        from . import patterns  # noqa: F401


def has_centerpiece_regions(slug: str) -> bool:
    _ensure_registered()
    return slug in CENTERPIECE_REGION_FNS


def centerpiece_regions(slug: str, n_px: int, params: dict[str, Any] | None = None) -> RegionArt | None:
    """The region map of ``slug`` at ``n_px``, or None if the slug has none."""
    _ensure_registered()
    fn = CENTERPIECE_REGION_FNS.get(slug)
    if fn is None:
        return None
    art = fn(max(16, int(n_px)), dict(params or {}))
    if art is None:
        return None
    return art.resized(max(16, int(n_px)))


def single_layer_centerpiece(spec: Any) -> bool:
    """True when ``spec`` is a single-ply face whose slug has a region map —
    the one predicate every writer tests before choosing the region path."""
    return bool(getattr(spec, "single_ply", False)) and has_centerpiece_regions(
        getattr(spec, "pattern_slug", "")
    )
