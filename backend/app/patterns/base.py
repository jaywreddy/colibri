from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

# ``app.production`` imports nothing from ``app``, so reading the litho floor
# from it here is safe: the cycle that used to force a hand-mirrored copy
# (patterns -> effects.drc -> export_fine -> patterns) does not exist any more.
from ..production import LITHO_FLOOR_UM


@dataclass
class ParamSpec:
    name: str
    label: str
    type: str  # "float" | "int" | "bool" | "choice"
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = None
    choices: list[str] | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Substrate:
    thickness_um: float = 500.0
    material: str = "fused silica"
    n: float = 1.46


# ``LITHO_FLOOR_UM`` is re-exported from here (see the import above) because
# every generator already reads it off ``patterns.base``; it is DEFINED in
# ``app.production`` along with the rest of the process constants.
#
# Reported minima are products of float slider values, so a design that lands
# exactly ON the floor (a 4 µm grating at 50% duty is 2 µm line / 2 µm gap — the
# canonical minimum feature) must pass, not fail by a rounding bit.
_FLOOR_TOL_UM = 1e-6


def check_litho_floor(min_feature_um: float) -> None:
    """Refuse geometry the 2 µm process cannot print.

    Single choke point for the litho floor: every generator returns a
    GeneratedPattern (which calls this) and plate composition reads the same
    number through ``Pattern.metadata`` (which also calls this), so no manifest
    can advertise a sub-floor ``min_feature_um`` and no pattern-level SVG/GDS
    export — which does NOT run export_fine's drc_clean heal — can ship
    unprintable geometry. Message style follows ``check_lattice_budget`` (state
    the limit, name the dial); the /patterns and /boxes routes surface the
    ValueError verbatim as an HTTP 400.

    Generators must report the honest minimum over BOTH layers, since this is the
    only thing checking them. ``float('inf')`` (an empty layer) is accepted.
    """
    if min_feature_um < LITHO_FLOOR_UM - _FLOOR_TOL_UM:
        raise ValueError(
            f"min_feature_um={min_feature_um:g} µm is below the "
            f"{LITHO_FLOOR_UM:g} µm litho floor (2 µm line + 2 µm gap on the "
            "gold-on-quartz process). Increase the period or move the duty "
            "toward 0.5 — a pattern-level SVG/GDS export bypasses "
            "export_fine's DRC heal, so sub-floor features would reach fab "
            "geometry unrepaired."
        )


@dataclass
class GeneratedPattern:
    front: MultiPolygon
    back: MultiPolygon
    extent_um: tuple[float, float]
    pixel_pitch_um: float = 0.5
    min_feature_um: float = 2.0
    substrate: Substrate = field(default_factory=Substrate)
    extra: dict = field(default_factory=dict)
    # Per-variant overrides for the class-level render_recipe. Generators can
    # push path-typed data here (e.g. {"view_a_png": "view_a.png"}) that the
    # shader needs beyond the standard front/back textures. Stays empty for
    # patterns that don't need recipe-specific artifacts.
    recipe_data: dict = field(default_factory=dict)
    # Optional extra polygon layers that the backend will rasterize into
    # "<name>.png" alongside front/back. Used by stereo_lenticular to ship
    # view_a / view_b scenes separately from the front slit barrier. service.py
    # writes the resulting PNG URLs into recipe_data[<name>] automatically, so
    # generators only need to set this field — no manual path bookkeeping.
    extra_layers: dict[str, MultiPolygon] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_litho_floor(self.min_feature_um)


def _as_multipolygon(geom: BaseGeometry) -> MultiPolygon:
    if geom.is_empty:
        return MultiPolygon()
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    # GeometryCollection or similar — filter to polygons
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    return MultiPolygon(polys)


# Shader recipes the catalog may reference. Kept in sync with the frontend's
# RenderRecipe union / RECIPE_IDS map (frontend/src/api.ts): stereo_lenticular
# id 0, moire_interactive id 1, foliage_moire id 3 (every composed box plate).
# "phase_shift_overlay" (id 2) was RETIRED 2026-07 with zero users — its
# "switch" was a shader view-sign bias, not physics; the numeric ids 0/1/3
# stay stable with a permanent hole at 2.
RECIPE_NAMES = {
    "stereo_lenticular",
    "moire_interactive",
    "foliage_moire",
}


@dataclass(frozen=True)
class PatternMetadata:
    """Everything a variant advertises EXCEPT the geometry.

    The plate compositor needs exactly these five things from the central
    pattern — pitch (to size its canvas), min feature, ``extra``,
    ``recipe_data`` and the extra-layer names (to stamp into the plate
    manifest) — and none of its polygons. See ``Pattern.metadata``.
    """

    pixel_pitch_um: float
    min_feature_um: float
    extra: dict
    recipe_data: dict
    # Ordered, matching ``GeneratedPattern.extra_layers`` insertion order: the
    # ``<name>_png`` recipe_data URLs service.py appends are ordered by it.
    extra_layer_names: tuple[str, ...]


# Memo for the ``generate()`` metadata fallback below. Keyed on (slug, params)
# and holding only the (tiny) PatternMetadata — the polygons are dropped with
# the GeneratedPattern, which matters on the 13.7 GB host (CLAUDE.md). Without
# it a generator that overrides none of the accessors would pay a full generate
# per accessor for one metadata read.
_metadata_fallback_cache: dict[str, PatternMetadata] = {}
_METADATA_FALLBACK_CACHE_MAX = 32


class Pattern(ABC):
    slug: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    tags: ClassVar[list[str]] = []
    params: ClassVar[list[ParamSpec]] = []
    tier: ClassVar[int] = 1
    theme: ClassVar[str] = "Colombia"
    # Which shader recipe the frontend should use when rendering this
    # pattern. Must be one of RECIPE_NAMES — the shader's uRecipe switch
    # only knows the recipes the catalog actively uses.
    render_recipe: ClassVar[str] = "moire_interactive"

    @classmethod
    @abstractmethod
    def generate(cls, **kwargs) -> GeneratedPattern: ...

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {p.name: p.default for p in cls.params}

    @classmethod
    def descriptor(cls) -> dict:
        return {
            "slug": cls.slug,
            "name": cls.name,
            "description": cls.description,
            "tags": cls.tags,
            "tier": cls.tier,
            "theme": cls.theme,
            "render_recipe": cls.render_recipe,
            "params": [p.to_dict() for p in cls.params],
        }

    # --- cheap metadata (no geometry) ---------------------------------------
    # ``plates._raster_compose_plate`` consumes ONLY a central pattern's pitch,
    # min feature, ``extra`` and ``recipe_data`` — it draws the centerpiece from
    # the motif silhouettes itself and never touches the generator's polygons.
    # Generating the whole variant for four numbers cost 1-4 s per face (8-20 s
    # per cold six-face box), so every generator whose metadata is pure
    # arithmetic on its params overrides these three. An override MUST reuse the
    # exact expression its ``generate`` hands to GeneratedPattern (share a
    # module-level helper where the expression is more than a line) — the two
    # values are the same manifest field and cannot be allowed to disagree.
    #
    # The base implementations fall back to ``generate()``, so a generator that
    # MEASURES a field off its emitted raster (capybara-scanimation's min gold,
    # bitmap-halftone's realized duty) stays correct by simply not overriding it.

    @classmethod
    def pixel_pitch_um(cls, **params) -> float:
        return cls._metadata_via_generate(**params).pixel_pitch_um

    @classmethod
    def min_feature_um(cls, **params) -> float:
        return cls._metadata_via_generate(**params).min_feature_um

    @classmethod
    def extra_metadata(cls, **params) -> tuple[dict, dict, tuple[str, ...]]:
        """``(extra, recipe_data, extra_layer_names)`` for these params.

        Key ORDER matters: both dicts are spread into the plate manifest, whose
        JSON bytes are cached, so an override must build them in the same order
        as its ``generate``.
        """
        meta = cls._metadata_via_generate(**params)
        return meta.extra, meta.recipe_data, meta.extra_layer_names

    @classmethod
    def metadata(cls, **params) -> PatternMetadata:
        extra, recipe_data, layer_names = cls.extra_metadata(**params)
        min_feature_um = cls.min_feature_um(**params)
        # The plate compositor reads metadata INSTEAD of generating, so this is
        # the litho-floor choke point on that path — GeneratedPattern's own call
        # never runs there. See check_litho_floor.
        check_litho_floor(min_feature_um)
        return PatternMetadata(
            pixel_pitch_um=cls.pixel_pitch_um(**params),
            min_feature_um=min_feature_um,
            extra=extra,
            recipe_data=recipe_data,
            extra_layer_names=layer_names,
        )

    @classmethod
    def _metadata_via_generate(cls, **params) -> PatternMetadata:
        """Last resort: run ``generate()`` and read the metadata off the result."""
        key = f"{cls.slug}:{sorted(params.items())!r}"
        hit = _metadata_fallback_cache.get(key)
        if hit is not None:
            return hit
        gp = cls.generate(**params)
        meta = PatternMetadata(
            pixel_pitch_um=gp.pixel_pitch_um,
            min_feature_um=gp.min_feature_um,
            extra=gp.extra,
            recipe_data=gp.recipe_data,
            extra_layer_names=tuple(gp.extra_layers),
        )
        if len(_metadata_fallback_cache) >= _METADATA_FALLBACK_CACHE_MAX:
            _metadata_fallback_cache.clear()
        _metadata_fallback_cache[key] = meta
        return meta


registry: dict[str, type[Pattern]] = {}


def register(cls: type[Pattern]) -> type[Pattern]:
    registry[cls.slug] = cls
    return cls


def ensure_multipolygon(geom: BaseGeometry) -> MultiPolygon:
    return _as_multipolygon(geom)
