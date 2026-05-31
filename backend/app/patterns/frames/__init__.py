"""Frame generator: vines + foliage + flowers wrapped around a rectangular frame.

Public surface:
- ``generate_frame(rect, theme, params, seed) -> Scene``: run the chosen
  algorithm and emit a renderer-agnostic Scene of strokes + flower/leaf sprites.
- ``scene_to_multipolygon(scene, theme, ...) -> MultiPolygon``: rasterize a
  Scene into the gold lithography mask for the front layer.

The Scene contract is intentionally minimal so the same generator output drives
both the fab mask (via ShapelyPen) and the in-browser preview (Canvas2D pen on
the frontend, fed the same JSON).
"""
from __future__ import annotations

from .geometry import RectFrame
from .scene import Scene, FlowerSprite, LeafSprite, Segment
from .api import generate_frame, scene_to_multipolygon

__all__ = [
    "RectFrame",
    "Scene",
    "FlowerSprite",
    "LeafSprite",
    "Segment",
    "generate_frame",
    "scene_to_multipolygon",
]
