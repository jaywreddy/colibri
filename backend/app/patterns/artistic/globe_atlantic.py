"""Atlantic globe — ONE static view, colour by REGION (single-layer diffraction).

This slug replaces ``globe-duo-phase`` on the box FRONT. That pattern was a
parallax-barrier tilt switch between a California-centred globe and a
Colombia-centred one, and a switch needs two planes: an interlaced BACK image
and a FRONT slit mask a substrate thickness in front of it. The box is now ONE
written quartz ply per face (see ``app/region_art.py``), so there is no second
plane, no parallax, and nothing for a barrier to gate — the duo switch cannot
exist here at all.

What a single ply CAN do is colour. Every region of the motif is written as its
own fine vertical 50 %-duty gold grating, and the PERIOD sets the hue it flashes
under a lamp; a period RATIO is view-independent, so the colours hold at every
tilt instead of pretending to move. The couple's geography therefore stops being
two views of one globe and becomes ONE map with five colours:

    ocean                       bare glass (no gold) — the dark sea
    generic land                4.15 µm   (bluest rung)
    United States               4.82 µm   — a gold star on California
    graticule + limb ring       5.19 µm
    Europe (Russia excluded)    5.59 µm
    Colombia                    6.02 µm   (reddest rung) — a gold star on Bogotá
    the two stars               SOLID gold (period 0)

The furniture would rather have been solid gold, and is not, for a hard reason:
the fine writer heals the merged layer with ``drc_clean_region``, which returns
EXTERIOR rings only. A closed curve of solid gold therefore comes back as the
filled disk it bounds — a solid limb ring makes the globe a blank gold coin, and
a solid graticule fills every cell its meridians and parallels enclose. Grating
regions are disjoint vertical lines and enclose nothing, so they are immune; the
stars are solid because a star is simply connected. See
``geo.render.GLOBE_REGION_FURNITURE``.

The projection, the Natural Earth 110m rings and the label raster all live in
``patterns.geo.render`` / ``patterns.motifs.globe`` — the SAME code the composed
plate reads through ``region_art.centerpiece_regions``, so the catalog pattern,
the composed preview and the fine GDS bake are one map drawn once. See
``motifs.globe.globe_regions`` for why the centre is 36 N / 56 W and why the
periods are assigned the way they are.

This standalone ``generate`` bakes the same gratings the plate emitter writes by
calling the emitter's own rect builders (``export_fine._clip_axis_grating`` /
``_zone_solid_rects``, behind the same one-cell inter-region gutter) rather than
re-deriving them — the two must not be allowed to disagree, and the fab path is
the one that is right by construction. The import is function-local because
``export_fine`` imports ``plates``, which imports this package.
"""
from __future__ import annotations

import math

from shapely.geometry import MultiPolygon, box

from ...region_art import RegionArt, register_centerpiece_regions
from .._helpers import check_lattice_budget
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs.globe import (
    ATLANTIC_COLOMBIA_PERIOD_UM,
    ATLANTIC_EUROPE_PERIOD_UM,
    ATLANTIC_FURNITURE_PERIOD_UM,
    ATLANTIC_LAND_PERIOD_UM,
    ATLANTIC_LAT0,
    ATLANTIC_LON0,
    ATLANTIC_USA_PERIOD_UM,
    globe_regions,
)

# Raster floor for the STANDALONE bake. On a plate the region map is rastered at
# region_art.REGION_ZONE_PITCH_UM (20 µm) over the ~16.6 mm art box — 830 cells,
# plenty for a coastline. A catalog variant is a 2 mm tile, where 20 µm would
# give a 100-cell map and a globe made of stair-steps, so the standalone raster
# is floored at 256 cells (and capped by the same REGION_ZONE_MAX_PX). Only the
# region BOUNDARIES quantise to the resulting pitch; the periods inside are exact.
MIN_MAP_PX = 256

# Duty of every region grating. 0.5 is what region_art promises and what
# ``check_region_periods`` sizes the litho/finish floors against.
REGION_DUTY = 0.5


def _map_px(extent_um: float) -> int:
    """Side of the region label raster for a standalone bake at ``extent_um``."""
    from ...region_art import REGION_ZONE_MAX_PX, REGION_ZONE_PITCH_UM

    n = int(round(float(extent_um) / REGION_ZONE_PITCH_UM))
    return max(MIN_MAP_PX, min(n, REGION_ZONE_MAX_PX))


@register
class GlobeAtlantic(Pattern):
    slug = "globe-atlantic"
    name = "Atlantic globe — colour by region"
    description = (
        "One orthographic globe centred over the mid Atlantic at 36 N / 56 W, "
        "so California, Colombia and western Europe all sit inside the limb at "
        "once. It is a SINGLE-LAYER DIFFRACTION mapping, not a tilt switch: the "
        "ocean is bare glass, and every land region is written as its own fine "
        "vertical 50 % gold grating whose period sets the colour it flashes "
        "under a lamp — generic land at 4.15 µm, the United States at 4.82 µm, "
        "the graticule and limb ring at 5.19 µm, Europe at 5.59 µm and Colombia "
        "at 6.02 µm, five rungs of the same hue ladder the garland's leaf "
        "families use. Two stars are plain solid metal: one on California, one "
        "on Bogotá — the couple's two homes. Geography is Natural Earth 110m "
        "vector data, projected and rasterized in app.patterns.geo."
    )
    tags = ["globe", "diffraction", "colour", "Colombia", "Global Travel", "front"]
    tier = 1
    theme = "Colombia"
    # Same recipe as the photo sides and every composed plate: the centrepiece
    # is already the picture (``art_solid``), so the shader must not run a
    # procedural carrier over it.
    render_recipe = "foliage_moire"
    # Period bounds: ``export_fine.check_region_periods`` demands line = p·duty
    # STRICTLY above 2 × the die-finish radius (1 µm → 2 µm) as well as clear of
    # the 2 µm litho floor, so at duty 0.5 the smallest legal period is just over
    # 4.0 µm. 4.1 is the first safe step; the top of the ladder is 6.02 and 8 µm
    # of headroom lets a taste pass reshuffle hues without touching code.
    params = [
        ParamSpec("lat0", "Centre latitude", "float", ATLANTIC_LAT0, 0.0, 60.0, 0.5, "°"),
        ParamSpec("lon0", "Centre longitude", "float", ATLANTIC_LON0, -90.0, -20.0, 0.5, "°"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
        ParamSpec("land_period_um", "Land period", "float",
                  ATLANTIC_LAND_PERIOD_UM, 4.1, 8.0, 0.01, "μm"),
        ParamSpec("usa_period_um", "USA period", "float",
                  ATLANTIC_USA_PERIOD_UM, 4.1, 8.0, 0.01, "μm"),
        ParamSpec("europe_period_um", "Europe period", "float",
                  ATLANTIC_EUROPE_PERIOD_UM, 4.1, 8.0, 0.01, "μm"),
        ParamSpec("colombia_period_um", "Colombia period", "float",
                  ATLANTIC_COLOMBIA_PERIOD_UM, 4.1, 8.0, 0.01, "μm"),
        ParamSpec("furniture_period_um", "Graticule + limb period", "float",
                  ATLANTIC_FURNITURE_PERIOD_UM, 4.1, 8.0, 0.01, "μm"),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------

    @classmethod
    def region_periods_um(
        cls,
        land_period_um: float = ATLANTIC_LAND_PERIOD_UM,
        usa_period_um: float = ATLANTIC_USA_PERIOD_UM,
        europe_period_um: float = ATLANTIC_EUROPE_PERIOD_UM,
        colombia_period_um: float = ATLANTIC_COLOMBIA_PERIOD_UM,
        furniture_period_um: float = ATLANTIC_FURNITURE_PERIOD_UM,
    ) -> dict[str, float]:
        """The four diffractive periods, named. ONE expression, read by
        ``generate``, by the metadata accessors and by the registered region
        function, so a plate can never be written at periods the manifest does
        not advertise."""
        return {
            "land": float(land_period_um),
            "usa": float(usa_period_um),
            "europe": float(europe_period_um),
            "colombia": float(colombia_period_um),
            "graticule-limb": float(furniture_period_um),
        }

    @classmethod
    def pixel_pitch_um(cls, extent_um: float = 2000.0, **_params) -> float:
        return float(extent_um) / _map_px(extent_um)

    @classmethod
    def min_feature_um(
        cls,
        land_period_um: float = ATLANTIC_LAND_PERIOD_UM,
        usa_period_um: float = ATLANTIC_USA_PERIOD_UM,
        europe_period_um: float = ATLANTIC_EUROPE_PERIOD_UM,
        colombia_period_um: float = ATLANTIC_COLOMBIA_PERIOD_UM,
        furniture_period_um: float = ATLANTIC_FURNITURE_PERIOD_UM,
        **_params,
    ) -> float:
        # Every diffractive region is a 50 % grating, so the narrowest printed
        # feature is the finest period's line (== its gap). The solid region
        # (the stars) is whole cells of the region raster — tens of µm — and the
        # emitter's one-cell gutter keeps neighbours from meeting, so no
        # sub-grating sliver exists to be narrower than this.
        periods = cls.region_periods_um(
            land_period_um, usa_period_um, europe_period_um, colombia_period_um,
            furniture_period_um,
        )
        return min(periods.values()) * REGION_DUTY

    @classmethod
    def extra_metadata(
        cls,
        lat0: float = ATLANTIC_LAT0,
        lon0: float = ATLANTIC_LON0,
        extent_um: float = 2000.0,
        land_period_um: float = ATLANTIC_LAND_PERIOD_UM,
        usa_period_um: float = ATLANTIC_USA_PERIOD_UM,
        europe_period_um: float = ATLANTIC_EUROPE_PERIOD_UM,
        colombia_period_um: float = ATLANTIC_COLOMBIA_PERIOD_UM,
        furniture_period_um: float = ATLANTIC_FURNITURE_PERIOD_UM,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        periods = cls.region_periods_um(
            land_period_um, usa_period_um, europe_period_um, colombia_period_um,
            furniture_period_um,
        )
        return (
            {
                "center_latlon": (float(lat0), float(lon0)),
                "region_periods_um": periods,
                "region_duty": REGION_DUTY,
                "grating_axis_deg": 0.0,
                "region_map_px": _map_px(extent_um),
                # Static by construction — there is nothing for a tilt to gate.
                # Stated so a readability/collage gate reads it rather than
                # hunting for a switch angle that does not exist.
                "switch_half_angle_deg": 0.0,
            },
            {
                # The centrepiece IS the picture: the shader fills ART_LEVEL with
                # plain gold and RAINBOW_LEVEL with the spectral sheen instead of
                # running a procedural carrier over either.
                "art_solid": True,
                "grating_duty": REGION_DUTY,
                "carrier_angle_deg": 0.0,
            },
            (),
        )

    @classmethod
    def generate(
        cls,
        lat0: float = ATLANTIC_LAT0,
        lon0: float = ATLANTIC_LON0,
        extent_um: float = 2000.0,
        land_period_um: float = ATLANTIC_LAND_PERIOD_UM,
        usa_period_um: float = ATLANTIC_USA_PERIOD_UM,
        europe_period_um: float = ATLANTIC_EUROPE_PERIOD_UM,
        colombia_period_um: float = ATLANTIC_COLOMBIA_PERIOD_UM,
        furniture_period_um: float = ATLANTIC_FURNITURE_PERIOD_UM,
    ) -> GeneratedPattern:
        # Function-local: export_fine → plates → app.patterns → this module.
        from ...export_fine import (
            _clip_axis_grating,
            _concat_rects,
            _drc_rects,
            _erode_zone,
            _zone_solid_rects,
            check_region_periods,
        )

        extent = (float(extent_um), float(extent_um))
        n_map = _map_px(extent_um)
        pitch = float(extent_um) / n_map

        periods = cls.region_periods_um(
            land_period_um, usa_period_um, europe_period_um, colombia_period_um,
            furniture_period_um,
        )
        # Rect-count estimate. Each diffractive region emits one rect per gold
        # line per contiguous VERTICAL run of the region in that column, and a
        # globe's regions are blobs: a meridian-parallel line crosses land a
        # handful of times, never more than a few. 16 runs per line is generous.
        # The solid stars emit per-row runs, which the quarter-raster term covers.
        est = 16 * sum(
            int(math.ceil(float(extent_um) / p)) for p in periods.values()
        ) + n_map * n_map // 4
        check_lattice_budget(
            est, "globe-atlantic region gratings",
            extent_um=extent_um, min_period_um=min(periods.values()),
        )

        art = globe_regions(
            n_map, lat0=lat0, lon0=lon0,
            land_period_um=periods["land"], usa_period_um=periods["usa"],
            europe_period_um=periods["europe"],
            colombia_period_um=periods["colombia"],
            furniture_period_um=periods["graticule-limb"],
        )
        check_region_periods(art.regions)

        parts = []
        for rid, reg in sorted(art.regions.items()):
            zone = art.labels == rid
            if not zone.any():
                continue
            # The SAME one-cell glass gutter export_fine._emit_region_art opens,
            # so two gratings never meet in a sub-floor sliver at a coastline.
            zone = _erode_zone(zone, 1)
            if not zone.any():
                continue
            if reg.solid:
                parts.append(_zone_solid_rects(zone, pitch, extent))
            else:
                parts.append(
                    _clip_axis_grating(zone, pitch, extent, reg.period_um, reg.duty, 0.0)
                )

        rects = _drc_rects(_concat_rects(parts))
        front = MultiPolygon(
            [box(float(x0), float(y0), float(x1), float(y1)) for x0, x1, y0, y1 in rects]
        )

        kw = dict(
            lat0=lat0, lon0=lon0, extent_um=extent_um,
            land_period_um=land_period_um, usa_period_um=usa_period_um,
            europe_period_um=europe_period_um, colombia_period_um=colombia_period_um,
            furniture_period_um=furniture_period_um,
        )
        extra, recipe_data, _layer_names = cls.extra_metadata(**kw)
        return GeneratedPattern(
            front=front,
            # SINGLE LAYER: one written ply, so there is no back plate at all.
            back=MultiPolygon(),
            extent_um=extent,
            pixel_pitch_um=pitch,
            min_feature_um=cls.min_feature_um(**kw),
            extra=extra,
            recipe_data=recipe_data,
        )


@register_centerpiece_regions(GlobeAtlantic.slug)
def globe_atlantic_regions(n_px: int, params: dict) -> RegionArt:
    """The front face's region map — the one function every writer reads.

    ``params`` is the face's ``pattern_params``, which a box face leaves empty;
    each key falls back to the Pattern's own ParamSpec default so the plate, the
    catalog variant and the fab bake cannot drift apart.
    """
    d = GlobeAtlantic.defaults()
    periods = GlobeAtlantic.region_periods_um(
        float(params.get("land_period_um", d["land_period_um"])),
        float(params.get("usa_period_um", d["usa_period_um"])),
        float(params.get("europe_period_um", d["europe_period_um"])),
        float(params.get("colombia_period_um", d["colombia_period_um"])),
        float(params.get("furniture_period_um", d["furniture_period_um"])),
    )
    return globe_regions(
        int(n_px),
        lat0=float(params.get("lat0", d["lat0"])),
        lon0=float(params.get("lon0", d["lon0"])),
        land_period_um=periods["land"],
        usa_period_um=periods["usa"],
        europe_period_um=periods["europe"],
        colombia_period_um=periods["colombia"],
        furniture_period_um=periods["graticule-limb"],
    )
