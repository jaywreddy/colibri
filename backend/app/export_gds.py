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

# klayout database unit used for the per-polygon hole subtraction below. 1 nm is
# three orders of magnitude finer than the 2 um litho floor, so the boolean is
# lossless at our feature sizes.
_HOLE_BOOL_DBU_UM = 0.001


def hole_free_dpolygons(
    poly: Polygon,
    dx: float = 0.0,
    dy: float = 0.0,
    unit_um: float = 1.0,
) -> list[Any]:
    """One shapely polygon -> hole-free klayout ``DPolygon``s, translated by (dx, dy).

    GDSII has no interior ring: a hole drawn on the same layer as its exterior
    PRINTS AS GOLD, so every punched eye, monogram counter and
    ``raster_to_polygons`` interior fills in. The rings must therefore be
    SUBTRACTED, and each resulting polygon flattened to a single hole-free
    contour (``Polygon.resolve_holes`` joins every hole to the hull with a cut
    line, which is how a GDS represents one).

    This is a per-polygon boolean in klayout, run only when the polygon actually
    has interiors. CLAUDE.md's no-``unary_union`` rule targets whole-geometry
    GEOS booleans on the request-serving hot paths; this is an offline fab
    writer touching one polygon at a time, and the alternative is a wrong mask.
    """
    import klayout.db as kdb

    def _dpoly(coords: Any) -> Any:
        return kdb.DPolygon(
            [kdb.DPoint(x * unit_um + dx, y * unit_um + dy) for x, y in coords]
        )

    hull = _dpoly(poly.exterior.coords)
    if not poly.interiors:
        return [hull]
    dbu = _HOLE_BOOL_DBU_UM
    region = kdb.Region(hull.to_itype(dbu))
    for ring in poly.interiors:
        region -= kdb.Region(_dpoly(ring.coords).to_itype(dbu))
    region.merge()
    out: list[Any] = []
    for p in region.each():
        flat = p.dup()
        flat.resolve_holes()
        out.append(
            kdb.DPolygon(
                [kdb.DPoint(pt.x * dbu, pt.y * dbu) for pt in flat.each_point_hull()]
            )
        )
    return out


def to_gds(
    polys: MultiPolygon,
    out_path: Path,
    layer: tuple[int, int] = (1, 0),
    cell_name: str = "PLATE",
    unit_um: float = 1.0,
) -> Path:
    """Write a GDSII file containing ``polys``. Requires gdsfactory.

    Interior rings are subtracted from their exterior (see
    :func:`hole_free_dpolygons`) so a hole reads as bare quartz, not gold.
    Raises ``NotImplementedError`` in an environment without gdsfactory.
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
        # Holes are SUBTRACTED, not drawn — see hole_free_dpolygons.
        for dpoly in hole_free_dpolygons(poly, unit_um=unit_um):
            c.add_polygon(dpoly, layer=layer)
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
