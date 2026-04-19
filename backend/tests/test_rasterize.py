"""Rasterization geometry tests: what goes in as polygons comes out at the
right pixel pitch and the right period."""
from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from app.rasterize import rasterize


def test_rasterize_empty_multipolygon_returns_blank_image() -> None:
    img = rasterize(MultiPolygon(), (100.0, 100.0), 0.5)
    arr = np.asarray(img)
    assert arr.shape == (200, 200)
    assert arr.max() == 0


def test_rasterize_dimensions_match_extent_over_pitch() -> None:
    img = rasterize(MultiPolygon(), (123.0, 77.0), 0.5)
    arr = np.asarray(img)
    assert arr.shape == (154, 246)  # (h, w) = (77/0.5, 123/0.5)


def test_rasterize_preserves_period_of_linear_grating() -> None:
    extent = (100.0, 40.0)
    pitch = 0.5
    period = 10.0  # um
    stripes = []
    x = -extent[0] / 2
    while x + period / 2 < extent[0] / 2:
        stripes.append(
            Polygon(
                [
                    (x, -extent[1] / 2),
                    (x + period / 2, -extent[1] / 2),
                    (x + period / 2, extent[1] / 2),
                    (x, extent[1] / 2),
                ]
            )
        )
        x += period
    img = rasterize(MultiPolygon(stripes), extent, pitch)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Take a center row; count edges (0->1 transitions).
    row = arr[arr.shape[0] // 2]
    edges = np.sum((row[1:] > 0.5) & (row[:-1] <= 0.5))
    expected_stripes = int(extent[0] / period)
    # Allow +-1 due to boundary rounding
    assert abs(edges - expected_stripes) <= 1, f"edges={edges} expected~{expected_stripes}"
