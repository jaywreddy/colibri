from __future__ import annotations

import drawsvg as dw
from shapely.geometry import MultiPolygon, Polygon


def to_svg(
    polys: MultiPolygon,
    extent_um: tuple[float, float],
    fill: str = "#E6BC50",
    background: str | None = None,
) -> str:
    """Emit a standards-compliant SVG whose user units are μm, origin at the center.

    This is the authoritative vector form for the fab upgrade path
    (SVG -> shapely -> gdsfactory -> GDS -> DXF).
    """
    w, h = extent_um
    d = dw.Drawing(w, h, origin=(-w / 2, -h / 2))
    if background is not None:
        d.append(dw.Rectangle(-w / 2, -h / 2, w, h, fill=background))

    if polys.is_empty:
        return d.as_svg()

    geoms = polys.geoms if isinstance(polys, MultiPolygon) else [polys]
    for poly in geoms:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        path = dw.Path(fill=fill, stroke="none", fill_rule="evenodd")
        ext = list(poly.exterior.coords)
        _move_line(path, ext)
        for ring in poly.interiors:
            _move_line(path, list(ring.coords))
        d.append(path)

    return d.as_svg()


def _move_line(path, coords: list[tuple[float, float]]) -> None:
    if not coords:
        return
    x0, y0 = coords[0]
    # SVG y points down; our convention has y up, so flip.
    path.M(x0, -y0)
    for x, y in coords[1:]:
        path.L(x, -y)
    path.Z()
