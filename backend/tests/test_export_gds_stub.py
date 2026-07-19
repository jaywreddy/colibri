"""Guard the fab-export path: the module must import, to_gds must write a
real GDS now that gdsfactory ships with the wafer exporter, and to_dxf stays
a uniform NotImplementedError placeholder."""
from __future__ import annotations

import pytest
from shapely.geometry import MultiPolygon, Polygon

from app.export_gds import to_gds, to_dxf


def _triangle() -> MultiPolygon:
    return MultiPolygon([Polygon([(0, 0), (1, 0), (0, 1)])])


def test_to_gds_writes_a_file_now_that_gdsfactory_is_installed(tmp_path):
    # gdsfactory became a runtime dep with the wafer exporter, so the legacy
    # scaffold now runs for real: it must produce a non-trivial GDS file.
    # (In envs WITHOUT gdsfactory it still raises NotImplementedError.)
    out = to_gds(_triangle(), tmp_path / "out.gds")
    assert out.exists() and out.stat().st_size > 0


def test_to_dxf_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        to_dxf()
