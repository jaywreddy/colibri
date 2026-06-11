"""Frame generator: vines + foliage + flowers wrapped around a rectangular frame.

Public surface:
- ``generate_frame(rect, params) -> Scene``: run the chosen algorithm and
  emit a renderer-agnostic Scene of strokes + flower/leaf sprites.
- ``render_scene_to_image(scene, rect, params, pitch) -> PIL.Image``: the
  fast raster path used by ``plates.materialize_plate`` (no Shapely).
- ``render_scene_to_svg(scene, rect, params) -> str``: the fast SVG path
  used by the lazy fab export (``plates.ensure_plate_svg``).
- ``scene_to_multipolygon(scene, rect, params) -> MultiPolygon``: polygon-
  space render via ShapelyPen, kept for ``plates.compose_plate`` and tests.

The Scene contract is intentionally minimal (plain serializable dataclasses)
so one generator output can drive every renderer backend identically.
"""
from __future__ import annotations

from .geometry import RectFrame
from .scene import Scene, FlowerSprite, LeafSprite, Segment
from .api import generate_frame, scene_to_multipolygon, render_scene_to_image, render_scene_to_svg

__all__ = [
    "RectFrame",
    "Scene",
    "FlowerSprite",
    "LeafSprite",
    "Segment",
    "generate_frame",
    "scene_to_multipolygon",
    "render_scene_to_image",
    "render_scene_to_svg",
]
