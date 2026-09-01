from __future__ import annotations

from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon

# Upper bound on raster dimensions. At pixel_pitch_um=1 this permits up to a
# 4000×4000 plate; at 0.5 a 2828×2828. Anything larger almost certainly means a
# pattern author picked a pathological pitch relative to extent (e.g. period
# 4μm with pitch=period/8 on a 2000μm plate → 8000×8000 = 64M pixels, which
# hung /patterns/generate indefinitely during E2E tests).
MAX_RASTER_PIXELS = 16_000_000


def rasterize(
    polys: MultiPolygon,
    extent_um: tuple[float, float],
    pitch_um: float,
) -> Image.Image:
    """Render a MultiPolygon (coords in μm, centered at origin) to a grayscale PNG.

    Pixel value 255 = gold present, 0 = transparent.
    """
    w = max(1, int(round(extent_um[0] / pitch_um)))
    h = max(1, int(round(extent_um[1] / pitch_um)))
    if w * h > MAX_RASTER_PIXELS:
        raise ValueError(
            f"Raster {w}x{h}={w * h} px exceeds cap {MAX_RASTER_PIXELS}. "
            f"Pick a coarser pixel_pitch_um (current {pitch_um}) or smaller extent_um {extent_um}."
        )
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)

    hx, hy = extent_um[0] / 2.0, extent_um[1] / 2.0

    def to_px(xy: tuple[float, float]) -> tuple[float, float]:
        x, y = xy
        return ((x + hx) / pitch_um, (hy - y) / pitch_um)

    if polys.is_empty:
        return img

    geoms = polys.geoms if isinstance(polys, MultiPolygon) else [polys]
    for poly in geoms:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        exterior = [to_px(p) for p in poly.exterior.coords]
        draw.polygon(exterior, fill=255)
        for ring in poly.interiors:
            draw.polygon([to_px(p) for p in ring.coords], fill=0)

    return img


# Thumbnail palettes per LITHO METAL (display sRGB). The masks themselves are
# metal-agnostic graylevel codes, so the chip is the only place a thumbnail has
# to make a colour choice — and it has to make the SAME one the 3D preview
# makes, or a chrome box shows six gold chips beside a platinum-lined render.
# (front layer, back layer, backdrop). The back layer is the dimmer
# second-surface sibling, matching the renderer's two-plane recession.
THUMBNAIL_PALETTES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "gold": ((230, 188, 80), (90, 72, 28), (12, 14, 18)),
    # Mask-grade chrome: bright neutral silver, the platinum-line read.
    "chrome": ((214, 219, 226), (84, 88, 95), (12, 14, 18)),
    # Low-reflective AR chrome: dark graphite linework.
    "chrome-ar": ((96, 100, 106), (38, 40, 44), (12, 14, 18)),
}

DEFAULT_THUMBNAIL_METAL = "gold"


def make_thumbnail(
    front: Image.Image,
    back: Image.Image,
    size: int = 256,
    metal: str = DEFAULT_THUMBNAIL_METAL,
) -> Image.Image:
    """Compose a thumbnail: front layer over the dimmer back layer on a dark field.

    ``metal`` selects the palette (see THUMBNAIL_PALETTES); an unknown name
    falls back to gold rather than raising, because a thumbnail is a chip and a
    wrong-coloured chip beats a 500 on the plate route.
    """
    fg, bg_layer, backdrop = THUMBNAIL_PALETTES.get(
        metal, THUMBNAIL_PALETTES[DEFAULT_THUMBNAIL_METAL]
    )
    w, h = front.size
    rgb = Image.new("RGB", (w, h), backdrop)
    rgb.paste(bg_layer, mask=back)
    rgb.paste(fg, mask=front)
    rgb.thumbnail((size, size), Image.Resampling.LANCZOS)
    return rgb
