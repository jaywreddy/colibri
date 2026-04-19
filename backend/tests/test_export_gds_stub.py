"""Guard the fab-export path: importing the module must work, and invoking
to_gds without gdsfactory installed must raise NotImplementedError (not
ImportError) so callers can catch it uniformly."""
from __future__ import annotations

import pytest
from shapely.geometry import MultiPolygon, Polygon

from app.export_gds import to_gds, to_dxf


def _triangle() -> MultiPolygon:
    return MultiPolygon([Polygon([(0, 0), (1, 0), (0, 1)])])


def test_to_gds_raises_not_implemented_without_dependency(tmp_path):
    with pytest.raises(NotImplementedError):
        to_gds(_triangle(), tmp_path / "out.gds")


def test_to_dxf_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        to_dxf()
