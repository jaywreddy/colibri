"""The all-rects fast path must render the same geometry as the general path.

``to_svg`` collapses hole-free axis-aligned rectangle input into a single
``<path>`` (perf: a plate layer is ~17k rects). The SVG is structurally
different from the per-polygon form — these tests pin that the RECTANGLES that
come back out are identical, and that anything the fast path cannot prove is a
rectangle still takes the general path.
"""
from __future__ import annotations

import re

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from app.export_svg import _as_rects, _move_line, to_svg


def _rect(x0: float, x1: float, y0: float, y1: float) -> Polygon:
    """A ring in ``raster_to_polygons`` order (lower-left CCW, y up)."""
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _rect_set(polys) -> set[tuple[float, float, float, float]]:
    return {
        (round(min(xs), 9), round(max(xs), 9), round(min(ys), 9), round(max(ys), 9))
        for poly in (polys.geoms if isinstance(polys, MultiPolygon) else [polys])
        for xs, ys in [poly.exterior.coords.xy]
    }


def _general_path_d(polys: MultiPolygon) -> list[str]:
    """The ``d`` attributes the pre-fast-path per-polygon writer produced."""
    out = []
    for poly in polys.geoms:
        path = _FakePath()
        _move_line(path, list(poly.exterior.coords))
        out.append(path.d)
    return out


class _FakePath:
    """drawsvg's Path ``d`` accumulation (elements.py::Path.append)."""

    def __init__(self) -> None:
        self.d = ""

    def _app(self, cmd: str, *args) -> None:
        if self.d:
            cmd = " " + cmd
        self.d += cmd + ",".join(map(str, args))

    def M(self, x, y) -> None:
        self._app("M", x, y)

    def L(self, x, y) -> None:
        self._app("L", x, y)

    def Z(self) -> None:
        self._app("Z")


def _subpath_texts(d_attr: str) -> list[list[tuple[str, str]]]:
    """Split an SVG ``d`` of closed polylines into per-subpath coordinate TEXT.

    A trailing repeat of the first vertex is dropped: the general writer walks
    the closed ring (5 coords) while the fast path relies on ``Z``.
    """
    subs = []
    for sub in d_attr.split("Z"):
        toks = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", sub)
        if not toks:
            continue
        pts = list(zip(toks[0::2], toks[1::2]))
        assert len(toks) % 2 == 0, f"odd coordinate count: {sub!r}"
        if len(pts) > 1 and pts[-1] == pts[0]:
            pts = pts[:-1]
        subs.append(pts)
    return subs


def _subpath_rects(d_attr: str) -> set[tuple[float, float, float, float]]:
    """Parse an SVG ``d`` of closed polylines back into y-up bboxes."""
    rects = set()
    for pts in _subpath_texts(d_attr):
        assert len(pts) == 4, f"not a rectangle subpath: {pts!r}"
        arr = np.asarray([[float(x), float(y)] for x, y in pts], dtype=np.float64)
        xs, ys = arr[:, 0], arr[:, 1]
        # Undo the y flip so the comparison is in our y-up convention.
        rects.add(
            (round(xs.min(), 9), round(xs.max(), 9), round(-ys.max(), 9), round(-ys.min(), 9))
        )
    return rects


_SYNTHETIC = MultiPolygon(
    [
        _rect(-10.0, -7.5, 0.0, 2.5),      # touches y = 0 (exercises the -0.0 flip)
        _rect(-7.5, -5.0, 0.0, 2.5),       # shares an edge with its neighbour
        _rect(2.5, 12.5, -30.0, -27.5),
        _rect(0.1, 0.30000000000000004, 4.0, 8.0),  # long float repr
    ]
)


def test_fast_path_emits_one_path_for_all_rect_input() -> None:
    svg = to_svg(_SYNTHETIC, (100.0, 100.0))
    assert len(re.findall(r"<path\b", svg)) == 1
    # Overlaps must fill, not XOR into holes (canonical winding + nonzero).
    assert 'fill-rule="nonzero"' in svg


def test_fast_path_rectangles_match_the_general_path() -> None:
    (d_attr,) = re.findall(r'd="([^"]+)"', to_svg(_SYNTHETIC, (100.0, 100.0)))
    fast = _subpath_rects(d_attr)
    general = set()
    for general_d in _general_path_d(_SYNTHETIC):
        general |= _subpath_rects(general_d)
    assert fast == general == _rect_set(_SYNTHETIC)


def test_fast_path_coordinate_text_matches_the_general_path() -> None:
    """Same vertex order AND same float formatting (incl. the ``-0.0`` y flip
    and 17-digit reprs) as the per-polygon writer, so a baked plate SVG differs
    only in the element split and the dropped redundant closing lineto."""
    (d_attr,) = re.findall(r'd="([^"]+)"', to_svg(_SYNTHETIC, (100.0, 100.0)))
    fast = _subpath_texts(d_attr)
    general = [sub for d in _general_path_d(_SYNTHETIC) for sub in _subpath_texts(d)]
    assert fast == general
    assert any("-0.0" in x or "-0.0" in y for sub in fast for x, y in sub)
    assert any("0.30000000000000004" in x for sub in fast for x, _ in sub)


def test_holed_polygon_keeps_the_general_path() -> None:
    outer = [(-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)]
    hole = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]
    polys = MultiPolygon([Polygon(outer, [hole]), _rect(30.0, 40.0, 0.0, 10.0)])
    svg = to_svg(polys, (200.0, 200.0))
    assert len(re.findall(r"<path\b", svg)) == 2
    assert 'fill-rule="evenodd"' in svg


def test_non_rect_polygon_keeps_the_general_path() -> None:
    tri = Polygon([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])
    svg = to_svg(MultiPolygon([tri, _rect(20.0, 30.0, 0.0, 5.0)]), (100.0, 100.0))
    assert len(re.findall(r"<path\b", svg)) == 2


def test_as_rects_rejects_degenerate_and_spurious_rings() -> None:
    # Zero-area sliver, diagonal ring, and a 4-vertex spur whose bbox lies
    # about its area — all must fall through to the general path.
    import shapely

    zero_area = Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 0.0), (0.0, 0.0)])
    diagonal = Polygon([(0.0, 0.0), (5.0, 5.0), (5.0, 0.0), (0.0, 0.0)])
    spur = Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (5.0, 0.0)])
    for bad in (zero_area, diagonal, spur):
        assert _as_rects(np.array([bad], dtype=object)) is None
    ok = np.array([_rect(0.0, 5.0, 0.0, 5.0)], dtype=object)
    assert shapely.get_num_coordinates(ok)[0] == 5
    assert _as_rects(ok) is not None


def test_as_rects_accepts_either_winding_and_start_vertex() -> None:
    ccw = _rect(0.0, 4.0, 1.0, 3.0)
    cw = Polygon([(0.0, 1.0), (0.0, 3.0), (4.0, 3.0), (4.0, 1.0)])
    for poly in (ccw, cw):
        rects = _as_rects(np.array([poly], dtype=object))
        assert rects is not None
        assert rects[0].tolist() == [0.0, 4.0, 1.0, 3.0]


def test_rect_fast_path_of_a_raster_run_matches_shapely_bboxes() -> None:
    """End-to-end on real ``raster_to_polygons`` output (the only producer the
    plate bake feeds ``to_svg``)."""
    from app.patterns._helpers import raster_to_polygons

    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[1, 1:4] = 1
    grid[2, 1:4] = 1
    grid[5, 0:8] = 1
    grid[6, 3] = 1
    polys = raster_to_polygons(grid, 2.0, (16.0, 16.0))
    (d_attr,) = re.findall(r'd="([^"]+)"', to_svg(polys, (16.0, 16.0)))
    assert _subpath_rects(d_attr) == _rect_set(polys)
