from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


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


RECIPE_NAMES = {
    "iridescent_grating",
    "stereo_lenticular",
    "moire_interactive",
    "near_field_carpet",
    "far_field_hologram",
    "stylized_amplitude",
}


class Pattern(ABC):
    slug: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    tags: ClassVar[list[str]] = []
    params: ClassVar[list[ParamSpec]] = []
    tier: ClassVar[int] = 1
    theme: ClassVar[str] = "Colombia"
    # Which shader recipe the frontend should use when rendering this
    # pattern. "stylized_amplitude" is the current flat-mask fallback, which
    # is what every pattern uses until it's upgraded to a physics-correct
    # recipe. See frontend/src/shaders/plate.frag for the switch.
    render_recipe: ClassVar[str] = "stylized_amplitude"

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


registry: dict[str, type[Pattern]] = {}


def register(cls: type[Pattern]) -> type[Pattern]:
    registry[cls.slug] = cls
    return cls


def ensure_multipolygon(geom: BaseGeometry) -> MultiPolygon:
    return _as_multipolygon(geom)
