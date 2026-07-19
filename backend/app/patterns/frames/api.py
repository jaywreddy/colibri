"""Public façade: generate a Scene and convert it to a lithography mask."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage
from shapely.geometry import MultiPolygon

from .._helpers import crop_parts
from .algorithms import ALGORITHMS
from .geometry import RectFrame
from .motifs import FLOWERS, FLOWER_SIZE_MULT, LEAVES, LEAF_SIZE_MULT
from .raster_pen import RasterPen
from .scene import Scene
from .shapely_pen import ShapelyPen
from .svg_pen import SvgPen
from .themes import get_theme


@dataclass
class FrameParams:
    """Generation knobs for a frame. Shared across algorithms.

    ``algorithm`` picks the grower (currently only ``colonize`` is implemented;
    L-system and flowing-border land in a later phase). ``frame_width_um``
    sets the decoration band thickness; default 12% of the shorter side gets
    applied if omitted.
    """

    algorithm: str = "wreath"
    theme_slug: str = "esmeralda"
    density: float = 1.0
    bloom: float = 0.6
    foliage: float = 0.6
    seed: int = 1
    frame_width_um: float | None = None
    stroke_width_um: float | None = None  # default: 0.45% of shorter side
    # Whole-face fill (box-first moiré front carrier) vs perimeter band.
    fill_interior: bool = False
    # Frame-band composition knobs (band mode only; see colonize.generate).
    # Defaults are tuned so a box face renders the "tailored" engraved-frame
    # look at the production FrameSpec defaults (density/bloom/foliage =
    # 1.0/0.6/0.6) with no per-face overrides. Dial per-face for variety.
    edge_gradient: float = 0.8   # outer-edge density bias (0 flat .. 1 strong)
    understory: float = 0.85     # density of the outer-band small-leaf infill
    border_vine: float = 1.15    # continuous running-ornament border line
    corner_fans: float = 1.0     # size/reach of the corner fan compositions
    # Wreath-algorithm composition preset (band mode only; ignored by colonize):
    # "garland2" | "laurel" | "garland" | "clusters". garland2 is the lush
    # mixed-tropical default (ordered vine + diversity + full band depth);
    # laurel is the austere single-species classic. See wreath.py::STYLES.
    wreath_style: str = "garland2"


def generate_frame(rect: RectFrame, params: FrameParams) -> Scene:
    theme = get_theme(params.theme_slug)
    algo = ALGORITHMS.get(params.algorithm)
    if algo is None:
        raise KeyError(f"Unknown frame algorithm: {params.algorithm}")
    kwargs = dict(
        seed=params.seed,
        density=params.density,
        bloom=params.bloom,
        foliage=params.foliage,
        band_um=params.frame_width_um,
        flower_types=theme.flowers,
        leaf_types=theme.leaves,
        fill_interior=params.fill_interior,
        edge_gradient=params.edge_gradient,
        understory=params.understory,
        border_vine=params.border_vine,
        corner_fans=params.corner_fans,
    )
    # wreath_style is wreath-only; colonize's signature doesn't accept it.
    if params.algorithm == "wreath":
        kwargs["wreath_style"] = params.wreath_style
    return algo(rect, **kwargs)


def scene_to_multipolygon(
    scene: Scene,
    rect: RectFrame,
    params: FrameParams,
) -> MultiPolygon:
    """Render a Scene to a single-color gold MultiPolygon for fab.

    Strokes turn into buffered polylines; flowers/leaves run their motif
    drawers against a ``ShapelyPen``. Everything CONCATENATES into one
    front-layer mask (no union — see below), cropped to the rectangle.
    """
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

    # No union — O(N²) on dense motif scenes and the raster/SVG consumers are
    # fill-only, so overlap is free. The crop, however, is load-bearing for
    # the VECTOR consumers: boundary motifs overhang the rect by up to their
    # own size, and gold must never reach the foil keep-out. ``crop_parts``
    # clips only the boundary-crossing members (vectorized, linear) — no
    # whole-geometry GEOS overlay, which OOMed at default density.
    parts = pen.finish(merge=False)
    return crop_parts(parts.geoms, (rect.width_um, rect.height_um))


def render_scene_to_image(
    scene: Scene,
    rect: RectFrame,
    params: FrameParams,
    pixel_pitch_um: float,
    level_fn: "Callable[[str], int] | None" = None,
) -> Image.Image:
    """Fast raster path — paints the scene directly into a PIL ``L`` image.

    Bypasses Shapely entirely. ~100× faster than ``scene_to_multipolygon``
    followed by ``rasterize`` because there's no Polygon construction or
    validity-checking overhead per motif primitive.

    ``level_fn`` maps a motif key (``"vine"`` or a species type string) to the
    L graylevel the pen paints it at. When ``None`` (tests / standalone motif
    renders) everything paints at 255, i.e. a plain silhouette mask. The plate
    compositor passes a lookup that encodes each species' moiré angle bucket as
    the graylevel (see plates.frame_level / MOTIF_ANGLE_BUCKET), so the shader
    can give every motif its own fringe direction. Motifs are painted one type
    at a time so a per-motif fill level is a single flat value; the small
    max-blend at the end keeps overlaps at the higher (later) level rather than
    summing into an out-of-band value.
    """
    extent_um = (rect.width_um, rect.height_um)
    w_px = max(1, int(round(rect.width_um / pixel_pitch_um)))
    h_px = max(1, int(round(rect.height_um / pixel_pitch_um)))

    def lvl(key: str) -> int:
        return 255 if level_fn is None else int(level_fn(key))

    # Paint each motif family onto its own scratch layer at that family's flat
    # level, then max-composite. This keeps the graylevel EXACT (no additive
    # bleed at overlaps that would push a pixel out of its angle bucket) while
    # still being a couple of cheap numpy maxima, not per-motif compositing.
    import numpy as np
    from PIL import ImageChops, ImageDraw

    img = Image.new("L", (w_px, h_px), 0)

    def to_px(x: float, y: float) -> tuple[float, float]:
        px = (x + rect.width_um / 2.0) / pixel_pitch_um
        py = (rect.height_um / 2.0 - y) / pixel_pitch_um
        return px, py

    # --- vine segments — paint as wide lines directly (no pen overhead) ---
    vine_lvl = lvl("vine")
    vdraw = ImageDraw.Draw(img)
    for seg in scene.segments:
        p1 = to_px(seg.x1, seg.y1)
        p2 = to_px(seg.x2, seg.y2)
        w_px_seg = max(1, int(round(seg.w / pixel_pitch_um)))
        vdraw.line([p1, p2], fill=vine_lvl, width=w_px_seg)

    # --- leaves, grouped by type so each type paints at its own flat level ---
    leaves_by_type: dict[str, list] = {}
    for leaf in scene.leaves:
        if LEAVES.get(leaf.type) is not None:
            leaves_by_type.setdefault(leaf.type, []).append(leaf)
    for ltype, group in leaves_by_type.items():
        layer = Image.new("L", (w_px, h_px), 0)
        pen = RasterPen(layer, extent_um, pixel_pitch_um)
        drawer = LEAVES[ltype]
        for leaf in group:
            size = leaf.size * LEAF_SIZE_MULT.get(leaf.type, 1.0)
            pen.save()
            pen.translate(leaf.x, leaf.y)
            pen.rotate(leaf.angle)
            drawer(pen, size, leaf.seed)
            pen.restore()
        if lvl(ltype) != 255:
            layer = layer.point(lambda v, L=lvl(ltype): L if v else 0)
        img = ImageChops.lighter(img, layer)

    # --- flowers, grouped by type ---
    flowers_by_type: dict[str, list] = {}
    for flower in scene.flowers:
        if FLOWERS.get(flower.type) is not None:
            flowers_by_type.setdefault(flower.type, []).append(flower)
    for ftype, group in flowers_by_type.items():
        layer = Image.new("L", (w_px, h_px), 0)
        pen = RasterPen(layer, extent_um, pixel_pitch_um)
        drawer = FLOWERS[ftype]
        for flower in group:
            size = flower.size * FLOWER_SIZE_MULT.get(flower.type, 1.0)
            pen.save()
            pen.translate(flower.x, flower.y)
            pen.rotate(flower.rot)
            drawer(pen, size, flower.seed)
            pen.restore()
        if lvl(ftype) != 255:
            layer = layer.point(lambda v, L=lvl(ftype): L if v else 0)
        img = ImageChops.lighter(img, layer)

    return _despeckle(img)


def _despeckle(img: Image.Image, min_px: int = 9) -> Image.Image:
    """Drop tiny disconnected specks from a frame mask.

    At plate raster (~33μm pitch) the feathery frond edges shed a few
    single-pixel fragments and antialiasing crumbs — hairline necks between a
    leaflet tip and its rachis fall below one raster cell and detach. The lush
    foliage itself is ONE large connected component (>700k px), so removing
    every component smaller than a ~3×3 cell (``min_px``) erases the isolated
    dust the brief calls out ("no isolated specks below ~3 raster px") while
    touching well under 0.1% of the gold. Cheap: one labeled pass over an
    ``L`` mask, no geometry work.
    """
    arr = np.asarray(img) > 0
    if not arr.any():
        return img
    lbl, n = ndimage.label(arr)
    if n <= 1:
        return img
    counts = np.bincount(lbl.ravel())
    # counts[0] is the background; components are 1..n.
    small = np.nonzero(counts[1:] < min_px)[0] + 1
    if small.size == 0:
        return img
    kill = np.isin(lbl, small)
    cleaned = np.array(img)
    cleaned[kill] = 0
    return Image.fromarray(cleaned, mode="L")


def render_scene_to_svg(
    scene: Scene,
    rect: RectFrame,
    params: FrameParams,
    *,
    fill: str = "#E6BC50",
    stroke: str = "#E6BC50",
) -> str:
    """Fast SVG path — emits SVG fragments directly from vertex lists.

    Same scale as ``scene_to_multipolygon`` but skips Shapely entirely. The
    output is the body inside an ``<svg>`` element; the caller wraps it
    (``plates.ensure_plate_svg`` does this for fab bundles).
    """
    pen = SvgPen(fill=fill, stroke=stroke)

    # Vine segments as stroked polylines
    for seg in scene.segments:
        pen.move_to(seg.x1, seg.y1)
        pen.line_to(seg.x2, seg.y2)
        pen.stroke_path(seg.w)

    # Leaves
    for leaf in scene.leaves:
        drawer = LEAVES.get(leaf.type)
        if drawer is None:
            continue
        size = leaf.size * LEAF_SIZE_MULT.get(leaf.type, 1.0)
        pen.save()
        pen.translate(leaf.x, leaf.y)
        pen.rotate(leaf.angle)
        drawer(pen, size, leaf.seed)
        pen.restore()

    # Flowers
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

    return pen.finish()
