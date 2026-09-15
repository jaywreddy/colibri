from __future__ import annotations

from typing import Any

import numpy as np

from .._helpers import check_lattice_budget, empty_layer
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..bitmap.photo import rects_to_multipolygon
from ..motifs import monogram
from ...region_art import register_centerpiece_regions

# The two rungs of the single-ply colour ladder
# (``plates.SINGLE_PLY_LEAF_HUE_PERIODS_UM`` = 4.15 … 6.02 µm, blue→red) the
# two initials are written at. 4.47 and 6.02 are the ladder's second and last
# rungs: their first orders differ by 1.35×, so at any one tilt the J flashes
# blue-green where the P flashes orange-red — as far apart as the ladder goes
# without standing the J on 4.15 µm, the rung whose 2.075 µm lines have only
# 4 % of margin over the 2 µm litho floor and the die finish. The hero of the
# lid is not the place to spend that margin.
J_PERIOD_UM = 4.47
P_PERIOD_UM = 6.02

# Cell of the region raster the STANDALONE ``generate`` writes on. The composed
# box face rasters at ``region_art.REGION_ZONE_PITCH_UM`` (20 µm — a quarter of
# the eye's limit on a 17.4 mm lid at 300 mm); the standalone part is a 2 mm
# miniature of the same art, so it needs a proportionally finer cell or the
# letterform quantises to nothing. 6.25 µm keeps the vertical run height a
# comfortable 3× over the litho floor.
REGION_CELL_UM = 6.25


def _resolve_grid(extent_um: float) -> tuple[int, float]:
    """``(n_grid, cell_um)`` for the region raster.

    Module-level because ``generate`` and ``pixel_pitch_um`` both need it and
    the plate compositor reads the pitch WITHOUT generating — one expression,
    so they cannot disagree.
    """
    n_grid = max(128, min(512, int(round(extent_um / REGION_CELL_UM))))
    return n_grid, extent_um / n_grid


def _grating_rects(
    zone: np.ndarray,
    cell_um: float,
    extent_um: tuple[float, float],
    period_um: float,
    duty: float,
) -> np.ndarray:
    """One region's VERTICAL 50 %-duty grating, clipped to ``zone``.

    ``(N, 4)`` ``[x0, x1, y0, y1]`` µm rectangles, plate-centred — the same
    construction ``export_fine._emit_region_art`` bakes for the box face, at
    this pattern's own raster cell. Built analytically from the zone's row runs
    rather than by rasterizing the grating: a 4.47 µm period on a 2 mm extent
    would need a 1 µm cell and four million lattice cells to raster, and the
    result would be the same axis-aligned rectangles.

    Only WHOLE lines are kept. A line clipped by a run edge would be a rect
    narrower than ``period·duty``, i.e. a DRC sliver under the 2 µm floor, in
    exchange for a letter edge sharper than the ≤ 6 µm it costs — which is
    already three times finer than the 20 µm the real part quantises to.
    """
    h, w = zone.shape
    hx, hy = w * cell_um / 2.0, h * cell_um / 2.0
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = zone
    d = np.diff(padded, axis=1)
    rows, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)
    if starts.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    x_start = starts * cell_um - hx
    x_end = ends * cell_um - hx
    y1 = hy - rows * cell_um
    y0 = y1 - cell_um

    if period_um <= 0.0:  # solid gold
        return np.stack([x_start, x_end, y0, y1], axis=1)

    line = period_um * duty
    k0 = np.ceil(x_start / period_um).astype(np.int64)
    # Last line start that still fits a whole line inside the run.
    k1 = np.floor((x_end - line) / period_um).astype(np.int64) + 1
    counts = np.maximum(0, k1 - k0)
    total = int(counts.sum())
    if total == 0:
        return np.empty((0, 4), dtype=np.float64)
    idx = np.repeat(np.arange(counts.size), counts)
    offs = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    lx0 = (k0[idx] + offs) * period_um
    return np.stack([lx0, lx0 + line, y0[idx], y1[idx]], axis=1)


@register_centerpiece_regions("monogram-jp")
def monogram_jp_regions(n_px: int, params: dict[str, Any]):
    """The lid's single-layer diffraction map: J at one period, P at another.

    Registered under the face's ``pattern_slug``, so the fine GDS bake, the fab
    SVG bake and the composed preview all read the SAME letters and the SAME
    two periods (``region_art`` module docstring).
    """
    d = MonogramJP.defaults()
    return monogram.monogram_regions(
        n_px,
        overlap=float(params.get("overlap", d["overlap"])),
        first_period_um=float(params.get("j_period_um", d["j_period_um"])),
        second_period_um=float(params.get("p_period_um", d["p_period_um"])),
    )


@register
class MonogramJP(Pattern):
    """Interlocked cursive J+P wedding monogram, as a SINGLE-LAYER DIFFRACTION
    MAPPING: one colour per initial.

    The box is one written quartz ply per face. There is no second plate to
    beat against, so the monogram is not a silhouette carved into a carrier: it
    is a map of two regions, one per letter, each written as its own fine
    vertical 50 %-duty gold grating. The PERIOD is the colour — 4.47 µm for the
    J, 6.02 µm for the P, two rungs of the same ladder the garland's leaf
    families use — so under a lamp the two initials flash two different hues and
    sweep through them together as the lid turns, while the glass between them
    stays dark. Uniform over each letter: one colour per initial, end to end, no
    patches and no strips.

    Where the two letters cross, the J keeps its stroke and the P is cut clear
    of it and re-opened, so the P visibly passes UNDER — an engraver's weave,
    which is also the only honest answer on ONE ply, where two gratings cannot
    occupy the same gold. The intersection does NOT get a third colour of its
    own: Great Vibes sets the P's bowl tangent to the J's arch, so most of that
    intersection is a tapering sliver far under the ~2 periods a region needs
    before it has a spectrum at all (see ``monogram.monogram_regions``).

    The box face gets this map through ``monogram_jp_regions`` above, which the
    fine bake, the fab SVG and the composed preview all dispatch to. This
    standalone ``generate`` bakes the same two gratings directly, so the
    catalog, the tilt collage and the standalone fab path show the same part.
    """

    slug = "monogram-jp"
    name = "J + P monogram (engagement engraving)"
    description = (
        "An interlocked cursive J and P — the couple's initials entwined the "
        "way an engraver would set them on a signet or a wedding invitation "
        "(Great Vibes copperplate swashes). One written ply, no carrier: each "
        "initial is a region of fine vertical gold grating whose period IS its "
        "colour, so the J flashes blue-green and the P orange-red as the lid "
        "turns, and the P passes under the J where they cross."
    )
    tags = ["monogram", "engagement", "cursive", "diffraction", "single-ply", "lid"]
    tier = 1
    theme = "Colombia"
    # A composed box face, like the photograph sides: the plate manifest forces
    # foliage_moire anyway (CLAUDE.md renderer honesty), and declaring it here
    # keeps the standalone Pattern Lab view honest — there is no moiré on one
    # ply, so moire_interactive would be a shader fake of an effect the part
    # does not have.
    render_recipe = "foliage_moire"
    params = [
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
        # 0.76: the P's bowl rides over the J's stem so the pair is one knot,
        # 0.92 of the art box wide and two-thirds as tall.
        ParamSpec("overlap", "Glyph interlock", "float", 0.76, 0.4, 0.85, 0.01),
        # Both ranges are the colour ladder's own span: every value in range
        # writes lines of at least 4.15/2 = 2.075 µm, which clears the 2 µm
        # litho floor and the 1 µm die-finish open (export_fine.check_region_periods).
        ParamSpec("j_period_um", "J colour period", "float", J_PERIOD_UM, 4.15, 6.02, 0.01, "μm"),
        ParamSpec("p_period_um", "P colour period", "float", P_PERIOD_UM, 4.15, 6.02, 0.01, "μm"),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------

    @classmethod
    def pixel_pitch_um(
        cls,
        extent_um: float = 2000.0,
        overlap: float = 0.76,
        j_period_um: float = J_PERIOD_UM,
        p_period_um: float = P_PERIOD_UM,
    ) -> float:
        return _resolve_grid(extent_um)[1]

    @classmethod
    def min_feature_um(
        cls,
        extent_um: float = 2000.0,
        overlap: float = 0.76,
        j_period_um: float = J_PERIOD_UM,
        p_period_um: float = P_PERIOD_UM,
    ) -> float:
        """The finest gold line: half the FINER of the two region periods.

        Both regions are 50 % duty and only whole lines are written, so the
        narrowest rectangle on the plate is exactly ``period · duty``."""
        return min(j_period_um, p_period_um) * 0.5

    @classmethod
    def extra_metadata(
        cls,
        extent_um: float = 2000.0,
        overlap: float = 0.76,
        j_period_um: float = J_PERIOD_UM,
        p_period_um: float = P_PERIOD_UM,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        return (
            {
                "construction": "single_layer_regions",
                "region_periods_um": [float(j_period_um), float(p_period_um)],
                "grating_duty": 0.5,
                "grating_angle_deg": 0.0,
                "overlap": float(overlap),
            },
            # What every region face publishes: the art IS the metal, so the
            # shader fills it as gold instead of laying a procedural carrier
            # over a silhouette (plates._carrier_recipe_data).
            {"art_solid": True},
            (),
        )

    @classmethod
    def generate(
        cls,
        extent_um: float = 2000.0,
        overlap: float = 0.76,
        j_period_um: float = J_PERIOD_UM,
        p_period_um: float = P_PERIOD_UM,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)
        n_grid, cell_um = _resolve_grid(extent_um)

        art = monogram.monogram_regions(
            n_grid,
            overlap=overlap,
            first_period_um=j_period_um,
            second_period_um=p_period_um,
        )

        parts: list[np.ndarray] = []
        areas: dict[str, float] = {}
        for rid, reg in sorted(art.regions.items()):
            zone = art.labels == rid
            if not zone.any():
                continue
            parts.append(_grating_rects(zone, cell_um, extent, reg.period_um, reg.duty))
            areas[reg.name] = round(float(zone.sum()) / zone.size, 4)
        rects = (
            np.concatenate(parts, axis=0) if parts else np.empty((0, 4), dtype=np.float64)
        )
        # The true emitted count, not an upper bound: the rects are numpy
        # (a few MB at the cap) and only the shapely build below is expensive.
        check_lattice_budget(
            int(rects.shape[0]),
            "monogram-jp region gratings",
            extent_um=extent_um,
            j_period_um=j_period_um,
            p_period_um=p_period_um,
        )

        kw = dict(
            extent_um=extent_um,
            overlap=overlap,
            j_period_um=j_period_um,
            p_period_um=p_period_um,
        )
        extra, recipe_data, _layer_names = cls.extra_metadata(**kw)
        extra = {**extra, "region_area_frac": areas, "n_line_rects": int(rects.shape[0])}
        return GeneratedPattern(
            # ONE ply: everything is in FRONT, the back is bare glass. There is
            # no second plate behind this face any more.
            front=rects_to_multipolygon(rects),
            back=empty_layer(),
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=cls.min_feature_um(**kw),
            extra=extra,
            recipe_data=recipe_data,
        )
