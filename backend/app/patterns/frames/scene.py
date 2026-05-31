"""Renderer-agnostic Scene — the contract between growers and pens.

A Scene is the *only* thing an algorithm emits; ShapelyPen turns it into a
lithography mask, and the frontend Canvas2D pen turns the same JSON into the
live preview. Keeping primitives serializable (plain dataclasses, no numpy
arrays, no shapely objects) is what enables that split.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class Segment:
    """A single vine stroke. ``w`` is stroke width in μm; ``t`` is growth time."""

    x1: float
    y1: float
    x2: float
    y2: float
    w: float
    t: float


@dataclass
class FlowerSprite:
    """An anchor for a motif-library flower. The drawer is selected by ``type``."""

    x: float
    y: float
    size: float
    t: float
    type: str
    rot: float
    seed: int


@dataclass
class LeafSprite:
    """An anchor for a motif-library leaf."""

    x: float
    y: float
    angle: float
    size: float
    type: str
    t: float
    seed: int


@dataclass
class Scene:
    segments: list[Segment] = field(default_factory=list)
    flowers: list[FlowerSprite] = field(default_factory=list)
    leaves: list[LeafSprite] = field(default_factory=list)
    max_t: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [asdict(s) for s in self.segments],
            "flowers": [asdict(f) for f in self.flowers],
            "leaves": [asdict(l) for l in self.leaves],
            "max_t": self.max_t,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        return cls(
            segments=[Segment(**s) for s in data.get("segments", [])],
            flowers=[FlowerSprite(**f) for f in data.get("flowers", [])],
            leaves=[LeafSprite(**l) for l in data.get("leaves", [])],
            max_t=float(data.get("max_t", 0.0)),
        )
