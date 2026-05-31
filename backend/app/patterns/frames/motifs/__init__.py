"""Esmeralda motif library — flowers + leaves drawn via the Pen interface.

Each motif's draw function takes ``(pen, size, seed)`` and assumes the pen's
transform is already positioned at the motif's anchor. Size is the
characteristic length of the sprite in μm; seed feeds per-sprite stylistic
randomness (e.g. petal-count jitter).

The registry maps the type strings used in ``Scene.flowers[].type`` /
``Scene.leaves[].type`` to drawers — the renderer (any pen) calls the matching
drawer to emit gold strokes.
"""
from __future__ import annotations

from typing import Callable

from .flowers import orchid, coffee, heliconia, anthurium
from .leaves import wax_palm, plantain, fern, philodendron
from ..pen import Pen

FlowerDrawer = Callable[[Pen, float, int], None]
LeafDrawer = Callable[[Pen, float, int], None]

FLOWERS: dict[str, FlowerDrawer] = {
    "orchid": orchid.draw,
    "coffee": coffee.draw,
    "heliconia": heliconia.draw,
    "anthurium": anthurium.draw,
}

LEAVES: dict[str, LeafDrawer] = {
    "wax_palm": wax_palm.draw,
    "plantain": plantain.draw,
    "fern": fern.draw,
    "philodendron": philodendron.draw,
}

# Per-type size multiplier — keeps some motifs visually balanced (large
# motifs like heliconia get scaled down, small like coffee scaled up).
FLOWER_SIZE_MULT: dict[str, float] = {
    "orchid": 1.0,
    "coffee": 0.9,
    "heliconia": 1.15,
    "anthurium": 1.05,
}
LEAF_SIZE_MULT: dict[str, float] = {
    "wax_palm": 1.2,
    "plantain": 1.3,
    "fern": 1.0,
    "philodendron": 1.15,
}

__all__ = [
    "FLOWERS",
    "LEAVES",
    "FLOWER_SIZE_MULT",
    "LEAF_SIZE_MULT",
    "FlowerDrawer",
    "LeafDrawer",
]
