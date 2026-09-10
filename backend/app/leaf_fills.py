"""Selectable fills for the SINGLE-PLY garland leaves (the photo side dies).

A one-ply face has no inner ply to beat against, so its leaves cannot be moiré
louvres; each motif family is instead written as its own fine diffractive
grating so it flashes spectral colour on its own view (see
``plates.SINGLE_PLY_LEAF_PERIOD_UM``). WHICH grating is a design choice with a
large file-size consequence, because the plate is written in CLEAR polarity:
``witness_dies.clear_field`` complements the metal inside the die box and
decomposes the (hole-riddled) result into convex pieces, so the written polygon
count tracks the METAL VERTEX count, not the metal area. Halving the number of
gold lines roughly halves the file.

The knobs live in ``plates`` (``SINGLE_PLY_LEAF_FILL``,
``SINGLE_PLY_LEAF_PERIOD_UM``, ``SINGLE_PLY_LEAF_HUE_PERIODS_UM``) and are
consumed by ``export_fine.build_plate_fine``'s single-ply branch. The default is
``"lines"`` at 6 µm, which is byte-for-byte the geometry that branch emitted
before this module existed — every other mode is opt-in.

Fills
-----
``lines``       one 50 %-duty line grating per family at ONE shared period, the
                family's own ANGLE fanned over the half turn. Colour is the same
                for every family; WHICH family flashes is set by the azimuth of
                the tilt. This is the shipping default.

``hue``         one 50 %-duty line grating per family, all at the SAME angle
                (axis-aligned, 0°), the family's own PERIOD taken from
                ``SINGLE_PLY_LEAF_HUE_PERIODS_UM``. Every family flashes at the
                same tilt, each in its own colour. Two wins: an axis-aligned
                grating goes through ``_clip_axis_grating``, whose column-merged
                run extraction emits ONE plate-frame rectangle per line per
                contiguous y-run (no rotation, so no off-DBU vertices and no
                decomposition slivers at the cut points); and the coarse rungs
                of the ladder cost proportionally fewer lines.

``hue2``        as ``hue``, but families alternate between 0° and 90° so there
                are two azimuths as well as the colour ladder. 90° is exact in
                DBU (the rotation is a coordinate swap), so it keeps the
                sliver-free property of ``hue``.

``crossed``     ``lines`` plus a second grating at +90° to it. 75 % gold, so the
                leaf is darker and each order carries less power; in CLEAR
                polarity the written data becomes a lattice of p/2 squares, one
                polygon per lattice site — see ``dots`` for why that is fatal.

``dots``        square gold pads on a square lattice, ``coverage`` fill. Orders
                in two planes at once (flashes at more azimuths, each weaker).
                Cost is AREA / p² polygons because nothing merges into runs; the
                emitter refuses above ``max_pads`` rather than build a file no
                shop will open. Provided so the cost can be measured, not
                recommended.

Every mode emits 4-vertex rectangles (axis-aligned in the grating-local frame),
which is the primitive ``export_fine`` and the DRC heal are built around.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

FILLS = ("lines", "hue", "hue2", "crossed", "dots")

#: Guard for :func:`dot_rects`. A pad lattice does not merge into runs, so its
#: polygon count is the leaf AREA over p²; 400 k is already an order of
#: magnitude past what the line fills cost for the same leaf.
MAX_PADS = 400_000


def min_period_um(fill: str, period_um: float, *, coverage: float = 0.5,
                  floor_um: float = 2.0) -> float:
    """Smallest legal period for ``fill`` at the 2 µm litho floor.

    ``lines`` / ``hue`` / ``hue2`` at 50 % duty: the line AND the gap are p/2, so
    p ≥ 2·floor = 4 µm. ``crossed``: the clear squares are p/2 on a side and the
    chrome necks between them are p/2, same bound. ``dots`` at coverage c: the
    pad is p·√c and the gap between pads is p(1 − √c), so p ≥ floor/(1 − √c) —
    6.83 µm at 50 % coverage, not 4.
    """
    if fill == "dots":
        return floor_um / max(1e-6, 1.0 - math.sqrt(coverage))
    return 2.0 * floor_um


def bucket_layers(
    fill: str,
    b: int,
    frame_count: int,
    fan0: float,
    period_um: float,
    hue_periods_um: tuple[float, ...],
) -> list[tuple[float, float]]:
    """The ``(period_um, angle_deg)`` line gratings family ``b`` is written with.

    ``fan0`` is the face's own random starting angle; the fan step is a half turn
    over ``frame_count`` families, so adjacent families never share an azimuth.
    ``dots`` returns an empty list — it is not a line grating and goes through
    :func:`dot_rects` instead.
    """
    fan = (fan0 + b * 180.0 / max(1, frame_count)) % 180.0
    if fill == "lines":
        return [(period_um, fan)]
    if fill == "crossed":
        return [(period_um, fan), (period_um, (fan + 90.0) % 180.0)]
    if fill in ("hue", "hue2"):
        if not hue_periods_um:
            raise ValueError(f"{fill}: SINGLE_PLY_LEAF_HUE_PERIODS_UM is empty")
        if fill == "hue":
            return [(float(hue_periods_um[b % len(hue_periods_um)]), 0.0)]
        p = float(hue_periods_um[(b // 2) % len(hue_periods_um)])
        return [(p, 0.0 if b % 2 == 0 else 90.0)]
    if fill == "dots":
        return []
    raise ValueError(f"unknown SINGLE_PLY_LEAF_FILL {fill!r}; expected one of {FILLS}")


def dot_rects(
    zone: np.ndarray,
    pitch_um: float,
    period_um: float,
    *,
    coverage: float = 0.5,
    max_pads: int = MAX_PADS,
) -> np.ndarray:
    """Square gold pads on a square lattice inside ``zone`` → plate-frame rects.

    ``zone`` is the boundary raster ``export_fine`` builds at ``pitch_um``
    (plate-centred, row 0 at the TOP, y up in µm) — the same convention
    :func:`export_fine._angled_grating_local_rects` reads. A pad is kept when its
    CENTRE lands in a set cell, so the leaf boundary is quantised to the raster
    exactly as a clipped grating's is.

    Returns ``(N, 4)`` ``[x0, x1, y0, y1]`` µm rectangles. Raises when the
    lattice would exceed ``max_pads``: a pad lattice cannot merge into runs, so
    N is the leaf area over p² and grows without bound as p falls.
    """
    if not zone.any():
        return np.empty((0, 4), dtype=float)
    h_px, w_px = zone.shape
    hx = w_px * pitch_um / 2.0
    hy = h_px * pitch_um / 2.0
    side = period_um * math.sqrt(max(1e-9, coverage))

    rows = np.flatnonzero(zone.any(axis=1))
    cols = np.flatnonzero(zone.any(axis=0))
    x_lo = cols[0] * pitch_um - hx
    x_hi = (cols[-1] + 1) * pitch_um - hx
    y_hi = hy - rows[0] * pitch_um
    y_lo = hy - (rows[-1] + 1) * pitch_um

    kx = np.arange(math.ceil(x_lo / period_um), math.floor(x_hi / period_um) + 1)
    ky = np.arange(math.ceil(y_lo / period_um), math.floor(y_hi / period_um) + 1)
    if kx.size == 0 or ky.size == 0:
        return np.empty((0, 4), dtype=float)
    if kx.size * ky.size > 40 * max_pads:
        raise ValueError(
            f"dot lattice bbox is {kx.size}x{ky.size} sites at p={period_um} um; "
            "coarsen the period or shrink the zone")

    xs = kx * period_um
    ys = ky * period_um
    # Row-chunked membership test: the bbox lattice of a garland ring is tens of
    # millions of sites, so never materialise it whole (the 13.7 GB host).
    out: list[np.ndarray] = []
    n = 0
    cx_idx = np.floor((xs + hx) / pitch_um).astype(int)
    keep_x = (cx_idx >= 0) & (cx_idx < w_px)
    xs_v, cx_v = xs[keep_x], cx_idx[keep_x]
    for y in ys:
        r = int(math.floor((hy - y) / pitch_um))
        if r < 0 or r >= h_px:
            continue
        row = zone[r]
        m = row[cx_v]
        if not m.any():
            continue
        x = xs_v[m]
        n += x.size
        if n > max_pads:
            raise ValueError(
                f"dot fill would emit more than {max_pads:,} pads at p={period_um} um "
                "(a pad lattice does not merge into runs; use a line fill or a much "
                "coarser period)")
        out.append(np.stack([x - side / 2.0, x + side / 2.0,
                             np.full(x.size, y - side / 2.0),
                             np.full(x.size, y + side / 2.0)], axis=1))
    if not out:
        return np.empty((0, 4), dtype=float)
    return np.concatenate(out, axis=0)


def describe(fill: str, period_um: float, hue_periods_um: tuple[float, ...],
             *, coverage: float = 0.5) -> dict[str, Any]:
    """Manifest-shaped summary of the fill actually written (for ``stats``)."""
    d: dict[str, Any] = {"fill": fill, "period_um": float(period_um),
                         "min_legal_period_um": round(min_period_um(fill, period_um,
                                                                    coverage=coverage), 3)}
    if fill in ("hue", "hue2"):
        d["hue_periods_um"] = [float(p) for p in hue_periods_um]
    if fill == "dots":
        d["coverage"] = float(coverage)
        d["pad_um"] = round(period_um * math.sqrt(coverage), 3)
    return d
