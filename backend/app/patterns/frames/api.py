"""Public façade: generate a Scene and convert it to a lithography mask."""
from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import MultiPolygon, box
from shapely.ops import unary_union

from ..base import ensure_multipolygon
from .algorithms import ALGORITHMS
from .geometry import RectFrame
from .motifs import FLOWERS, FLOWER_SIZE_MULT, LEAVES, LEAF_SIZE_MULT
from .scene import Scene
from .shapely_pen import ShapelyPen
from .themes import FrameTheme, get_theme


@dataclass
class FrameParams:
    """Generation knobs for a frame. Shared across algorithms.

    ``algorithm`` picks the grower (currently only ``colonize`` is implemented;
    L-system and flowing-border land in a later phase). ``frame_width_um``
    sets the decoration band thickness; default 12% of the shorter side gets
    applied if omitted.
    """

    algorithm: str = "colonize"
    theme_slug: str = "esmeralda"
    density: float = 1.0
    bloom: float = 0.6
    foliage: float = 0.6
    seed: int = 1
    frame_width_um: float | None = None
    stroke_width_um: float | None = None  # default: 0.45% of shorter side


def generate_frame(rect: RectFrame, params: FrameParams) -> Scene:
    theme = get_theme(params.theme_slug)
    algo = ALGORITHMS.get(params.algorithm)
    if algo is None:
        raise KeyError(f"Unknown frame algorithm: {params.algorithm}")
    return algo(
        rect,
        seed=params.seed,
        density=params.density,
        bloom=params.bloom,
        foliage=params.foliage,
        band_um=params.frame_width_um,
        flower_types=theme.flowers,
        leaf_types=theme.leaves,
    )


def scene_to_multipolygon(
    scene: Scene,
    rect: RectFrame,
    params: FrameParams,
) -> MultiPolygon:
    """Render a Scene to a single-color gold MultiPolygon for fab.

    Strokes turn into buffered polylines; flowers/leaves run their motif
    drawers against a ``ShapelyPen``. Everything unions into one front-layer
    mask, cropped to the rectangle.
    """
    short = min(rect.width_um, rect.height_um)
    pen = ShapelyPen()

    # --- vine segments ---
    for seg in scene.segments:
        pen.move_to(seg.x1, seg.y1)
        pen.line_to(seg.x2, seg.y2)
        pen.stroke_path(seg.w)

    # --- leaves ---
    for leaf in scene.leaves:
        drawer = LEAVES.get(leaf.type)
        if drawer is None:
            continue
        size = leaf.size * LEAF_SIZE_MULT.get(leaf.type, 1.0)
        pen.save()
        pen.translate(leaf.x, leaf.y)
        pen.rotate(leaf.angle)
        # Leaves are drawn with their petiole at the origin pointing +x.
        # The colonize tip is the petiole, so no extra offset.
        drawer(pen, size, leaf.seed)
        pen.restore()

    # --- flowers ---
    for flower in scene.flowers:
        drawer = FLOWERS.get(flower.type)
        if drawer is None:
            continue
        size = flower.size * FLOWER_SIZE_MULT.get(flower.type, 1.0)
        pen.save()
        pen.translate(flower.x, flower.y)
        pen.rotate(flower.rot)
        drawer(pen, size, flower.seed)
        pen.restore()

    mp = pen.finish()
    # Crop to the rectangle so any motif overshoot is trimmed cleanly.
    crop = box(rect.left, rect.bottom, rect.right, rect.top)
    return ensure_multipolygon(mp.intersection(crop))
