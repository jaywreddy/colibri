"""SVG export contract: every polygon emitted, correct viewBox, μm units."""
from __future__ import annotations

import re

from shapely.geometry import MultiPolygon, Polygon

from app.export_svg import to_svg


def _square(cx: float, cy: float, s: float) -> Polygon:
    h = s / 2
    return Polygon(
        [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    )


def test_svg_contains_subpath_for_every_polygon() -> None:
    polys = MultiPolygon([_square(0, 0, 5), _square(20, 0, 5), _square(-20, 0, 5)])
    svg = to_svg(polys, (100.0, 100.0))
    # All-rect input takes the single-path fast path (see
    # test_export_svg_rects.py); the per-polygon count is now one closed
    # subpath ("M…Z") each, not one <path> element each.
    assert len(re.findall(r"<path\b", svg)) == 1
    (d_attr,) = re.findall(r'd="([^"]+)"', svg)
    assert d_attr.count("M") == 3 and d_attr.count("Z") == 3


def test_svg_viewbox_matches_extent_um() -> None:
    svg = to_svg(MultiPolygon([_square(0, 0, 4)]), (123.0, 77.0))
    # origin=(-w/2, -h/2), width=w, height=h
    assert 'viewBox="-61.5 -38.5 123.0 77.0"' in svg or 'viewBox="-61.5 -38.5 123 77"' in svg


def test_svg_of_empty_multipolygon_has_no_paths() -> None:
    svg = to_svg(MultiPolygon(), (50.0, 50.0))
    assert "<path" not in svg
