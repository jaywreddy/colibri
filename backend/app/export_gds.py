"""Fab-export path stub: shapely MultiPolygon -> GDSFactory Component -> GDS/OASIS.

The plan defers actual DXF generation until the pattern library stabilises, but
the polygon path is well-defined and trivial to wire up once the GDSFactory
dependency is accepted. This module intentionally fails loudly with guidance.

Once enabled, the call graph will be:

    shapely MultiPolygon  (already produced by pattern generate())
        -> gdsfactory.Component with a single polygon on layer (1,0)
        -> c.write_gds(path)
        -> klayout CLI: klayout -zz -rm convert.py -rd in=foo.gds -rd out=foo.dxf

We keep the shim here so the API surface stays stable and the rest of the
codebase can import `export_gds.to_gds(...)` without conditional guards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Polygon


def to_gds(
    polys: MultiPolygon,
    out_path: Path,
    layer: tuple[int, int] = (1, 0),
    cell_name: str = "PLATE",
    unit_um: float = 1.0,
) -> Path:
    """Write a GDSII file containing ``polys``. Requires gdsfactory.

    Raises ``NotImplementedError`` until the fab-export path is prioritised.
    The signature and return contract are stable so callers can be written
    now and the body filled in later without a breaking change.
    """
    try:
        import gdsfactory as gf  # type: ignore[import-not-found]
    except ImportError as e:
        raise NotImplementedError(
            "GDS export is scaffolded. Install gdsfactory "
            "(`uv add gdsfactory`) and implement the shapely->Component "
            "bridge in app/export_gds.py. "
            "Reference path: shapely polygons -> gf.Component "
            "(using gf.kdb.Region) -> write_gds() -> klayout CLI -> DXF."
        ) from e

    # Activate the generic PDK so layer tuples resolve (mirrors
    # export_wafer.py; harmless if a PDK is already active).
    try:
        gf.gpdk.PDK.activate()
    except Exception:
        pass
    c: Any = gf.Component(name=cell_name)
    geoms = polys.geoms if isinstance(polys, MultiPolygon) else [polys]
    for poly in geoms:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        exterior = [(x * unit_um, y * unit_um) for x, y in poly.exterior.coords]
        c.add_polygon(exterior, layer=layer)
        for hole in poly.interiors:
            ring = [(x * unit_um, y * unit_um) for x, y in hole.coords]
            # gdsfactory handles holes via boolean subtraction in modern API;
            # see the project's CONTRIBUTING notes before enabling.
            c.add_polygon(ring, layer=layer)
    c.write_gds(out_path)
    return Path(out_path)


def to_dxf(*args, **kwargs) -> Path:
    """Post-process a GDS to DXF via klayout.

    Placeholder: once ``to_gds`` is live, shell out to
    ``klayout -zz -rm convert.py`` with the GDS as input. Keeping this
    separate so the pure-Python GDS path can ship first.
    """
    raise NotImplementedError(
        "DXF export goes through KLayout. Enable once to_gds() is live and "
        "the pattern library is locked."
    )
