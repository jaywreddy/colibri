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


def make_thumbnail(
    front: Image.Image, back: Image.Image, size: int = 256
) -> Image.Image:
    """Compose a thumbnail showing front (gold) over back (muted gold) on a dark field."""
    w, h = front.size
    rgb = Image.new("RGB", (w, h), (12, 14, 18))
    # back layer: muted gold
    rgb.paste((90, 72, 28), mask=back)
    # front layer: bright gold
    rgb.paste((230, 188, 80), mask=front)
    rgb.thumbnail((size, size), Image.Resampling.LANCZOS)
    return rgb
