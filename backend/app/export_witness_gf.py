"""The witness plate through gdsfactory: hierarchy and cell caching.

``export_witness.write_mask`` writes the plate flat-ish with klayout: one
BOUNDARY per rectangle, plus array references for the periodic sub-gratings.
That is 47 MB of GDSII for a plate whose OASIS is 1 MB, and the gap is almost
entirely the halftone bands — half a million rectangles at a fixed ~64 bytes
each, because a GDSII BOUNDARY spends five 4-byte coordinate PAIRS on a shape
that four numbers describe.

What GDSII does have is hierarchy. The bands are aperiodic in x, so no array can
absorb them, but their SIZES repeat enormously: widths are multiples of the
screen's column pitch and heights are quantised to the tone ladder, so 450,986
rectangles are only 404 distinct (width, height) pairs. Each pair becomes one
cell and each band one SREF at ~32 bytes — half a BOUNDARY.

gdsfactory's ``@gf.cell`` decorator is the caching: a component function is
memoised on its arguments, so ``band(w, h)`` called half a million times creates
404 cells and returns the cached one every other time. Instances themselves are
inserted through the underlying klayout cell, because half a million Python-level
``add_ref`` calls are the slow path and the geometry is the same either way.

Honest expectation, stated up front: this roughly HALVES the GDSII. It cannot
approach OASIS, because ~450 k references at ~32 bytes is a 14 MB floor that no
hierarchy removes — only a coarser asset would, and that is a design knob, not a
format one. The measured numbers are in the manifest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .witness_geom import LAYER_FRONT, LAYER_OUTLINE, PLATE_SIDE_UM


def write_mask_gf(
    plate: dict[str, Any],
    out_path: Path,
    *,
    formats: Sequence[str] = (".gds", ".oas"),
    quantum_nm: int = 1,
) -> list[Path]:
    """Write the plate hierarchically via gdsfactory.

    ``quantum_nm`` rounds cell dimensions before caching. 1 nm is exact for this
    geometry (every dimension is already on a 1 nm grid); a coarser quantum would
    merge near-identical cells at the cost of moving edges, and is deliberately
    NOT the default.
    """
    import gdsfactory as gf
    import klayout.db as kdb

    from .export_witness import LAYER_PAIR, _save_options

    gf.gpdk.PDK.activate()
    kcl = gf.kcl
    dbu = kcl.dbu  # 0.001 um

    def _i(v: float) -> int:
        return int(round(v / dbu))

    @gf.cell
    def band(w_nm: int, h_nm: int) -> gf.Component:
        """One rectangle, w x h, origin at its lower-left. Cached on (w, h)."""
        c = gf.Component()
        w, h = w_nm / 1000.0, h_nm / 1000.0
        c.add_polygon([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)], layer=LAYER_FRONT)
        return c

    top = gf.Component("WITNESS_5IN")
    tk = top.kdb_cell
    l_front = kcl.layer(*LAYER_FRONT)
    l_out = kcl.layer(*LAYER_OUTLINE)
    l_pair = kcl.layer(*LAYER_PAIR)

    # --- explicit rectangles -> SREF of a cached (w, h) cell -----------------
    n_sref = 0
    q = max(1, int(quantum_nm))
    for x0, x1, y0, y1 in np.asarray(plate["front"], dtype=np.float64):
        w_nm = max(q, int(round((x1 - x0) * 1000 / q)) * q)
        h_nm = max(q, int(round((y1 - y0) * 1000 / q)) * q)
        cell = band(w_nm, h_nm)
        tk.insert(kdb.CellInstArray(cell.kdb_cell.cell_index(),
                                    kdb.Trans(kdb.Vector(_i(x0), _i(y0)))))
        n_sref += 1
    n_cells_band = len({(int(round((x1 - x0) * 1000 / q)), int(round((y1 - y0) * 1000 / q)))
                        for x0, x1, y0, y1 in np.asarray(plate["front"], dtype=np.float64)})

    # --- periodic sub-gratings -> AREF of a cached stripe cell ---------------
    n_aref = 0
    for a in plate["arrays"]:
        rects = np.asarray(a["rects"], dtype=np.float64)
        if not len(rects):
            continue
        per = np.broadcast_to(np.asarray(a["period_um"], dtype=np.float64), (len(rects),))
        lin = np.broadcast_to(np.asarray(a["line_um"], dtype=np.float64), (len(rects),))
        phase = np.broadcast_to(np.asarray(a.get("phase_um", 0.0), dtype=np.float64),
                                (len(rects),))
        k0 = np.ceil((rects[:, 0] - phase - lin / 2.0) / per).astype(np.int64)
        k1 = np.floor((rects[:, 1] - phase - lin / 2.0) / per).astype(np.int64)
        for i in range(len(rects)):
            n = int(k1[i] - k0[i] + 1)
            if n <= 0:
                continue
            y0, y1 = rects[i, 2], rects[i, 3]
            cell = band(max(q, int(round(lin[i] * 1000 / q)) * q),
                        max(q, int(round((y1 - y0) * 1000 / q)) * q))
            x_start = phase[i] + k0[i] * per[i]
            tk.insert(kdb.CellInstArray(
                cell.kdb_cell.cell_index(),
                kdb.Trans(kdb.Vector(_i(x_start), _i(y0))),
                kdb.Vector(_i(per[i]), 0), kdb.Vector(0, 0), n, 1))
            n_aref += 1

    # --- boolean-derived and rotated polygons: nothing repeats, write flat ---
    n_poly = 0
    for pv in plate.get("free_polys", ()):
        tk.shapes(l_front).insert(kdb.Polygon(
            [kdb.Point(_i(x), _i(y)) for x, y in np.asarray(pv, dtype=np.float64)]))
        n_poly += 1

    # --- labels, outlines ----------------------------------------------------
    for x0, x1, y0, y1 in np.asarray(plate["labels"], dtype=np.float64):
        tk.shapes(l_front).insert(kdb.Box(_i(x0), _i(y0), _i(x1), _i(y1)))
    for x0, x1, y0, y1 in np.asarray(plate["outline"], dtype=np.float64):
        tk.shapes(l_out).insert(kdb.Box(_i(x0), _i(y0), _i(x1), _i(y1)))
    for x0, x1, y0, y1 in np.asarray(plate["pair_marks"], dtype=np.float64):
        tk.shapes(l_pair).insert(kdb.Box(_i(x0), _i(y0), _i(x1), _i(y1)))
    half = PLATE_SIDE_UM / 2.0
    tk.shapes(l_out).insert(kdb.Box(_i(-half), _i(-half), _i(half), _i(half)))

    stem = Path(out_path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sizes: dict[str, float] = {}
    for suffix in formats:
        p = stem.with_suffix(suffix)
        opts = _save_options(suffix, kdb)
        # gdsfactory would otherwise attach YAML metadata to every cell; a mask
        # shop does not want it and it is not small.
        top.write_gds(str(p), save_options=opts, with_metadata=False)
        written.append(p)
        sizes[suffix.lstrip(".")] = round(p.stat().st_size / 1e6, 2)

    plate["gds_gf"] = {
        "paths": [str(p) for p in written],
        "n_sref": n_sref,
        "n_band_cells": n_cells_band,
        "n_aref": n_aref,
        "n_flat_polys": n_poly,
        "size_mb_by_format": sizes,
    }
    return written
