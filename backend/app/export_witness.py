"""Lay out and write the 5-inch witness plate: the DoE, the map, and the GDS.

A 127 mm square is about 24x the area the witness cells were first laid out on
(a 25.7 mm spare die off the box blank). That is not simply more room for the
same sixteen cells; it changes what the plate can ANSWER. On the small die every
cell was a single POINT — one screen pitch, one colour period, one image scale —
and a point tells you whether a mechanism works, never where its optimum is.
With this much glass each knob gets a LADDER, and the plate stops being a
sampler and becomes an experiment.

What the area buys, in order of value
-------------------------------------
1. B6 becomes a real portrait. At 3.95 mm the face spanned 45 of the eye's 87 um
   integration cells and read as a coarse thumbnail; the cell qualified the
   screen but could never say whether the picture was any good. At 36 mm it
   spans 414. The three colour variants sit side by side at that size, which is
   the only honest way to choose between them.
2. Every aesthetic knob gets a one-factor ladder around the reference point, so
   a single plate says which way each one wants to move.
3. The image-SCALE ladder becomes possible at all — the one question that
   cannot be answered by looking at a big cell or a small one alone.

One plate, so Group A needs a bond
----------------------------------
Everything in Groups B, C and D is single-layer and needs no registration,
which is why all the sweeps live there. The two-layer cells (A1, A2, A4, A6 and
the C6 beat ladder) have nothing to register against on a single plate, so they
are emitted as a FRONT die and a BACK die side by side inside a dicing frame:
cut the pair, flip one, bond. That is the only part of this plate that costs a
process step the rest does not.

There is also only one gold layer. Both members of a pair are the same physical
write, so the writer puts every cell on ``LAYER_FRONT``; ``LAYER_PAIR`` carries
outlines only, marking which die is the back ply of which pair. Treating the
"back" cells as a second mask layer would double-expose the plate.

Rectangle economy
-----------------
Geometry is rect-space throughout and periodic sub-gratings are deferred as
ARRAY REFERENCES rather than polygons — see :func:`write_gds`. Flattened, the
colour cells alone would be several million rectangles; as arrays they are a
few hundred unit cells and one instance per band. ``--flat`` expands everything
for a shop that will not take arrays, and prints what that costs.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import witness_cells as wc
from .patterns.bitmap import colourplan as cp
from .patterns.bitmap import imageprep as ip
from .witness_geom import (
    EDGE_MARGIN_UM,
    GUTTER_UM,
    LABEL_H_UM,
    LAYER_FRONT,
    LAYER_LABEL,
    LAYER_OUTLINE,
    PLATE_SIDE_UM,
    REF_BASE_PERIOD_UM,
    REF_COARSEN_PX,
    REF_DUTY,
    REF_SCREEN_UM,
    REF_SPREAD,
    REF_TONE_STEPS,
    USABLE_UM,
    Cell,
    CellArt,
    _cat,
    _frame_rects,
    _text_rects,
)

LAYER_PAIR = (21, 0)     # outline-only: marks the back ply of a bonded pair
MM = 1000.0

# --- the design of experiments ----------------------------------------------
#
# One-factor-at-a-time ladders bracketing a common reference, not a factorial.
# The knobs are not independent — tone_steps is capped by line_period, and the
# colour base period is gated by the litho floor — so a factorial would spend
# most of its cells on combinations that are unbuildable by construction. OFAT
# around a known-good centre answers "which way does this want to move", which
# is the actual question at this stage.

SCALE_LADDER_MM = (4.0, 8.0, 14.0)
"""No 22 mm rung: the 30 mm headline cells already sit at the top of the
ladder, so it would cost 484 mm2 to duplicate a size the plate has."""
SCREEN_LADDER_UM = (20.0, 30.0, 44.0, 60.0)
STEPS_LADDER = (8, 12, 16, 22)
BASE_PERIOD_LADDER_UM = (4.0, 5.0, 6.5, 8.0)
DUTY_LADDER = (0.35, 0.50, 0.65)
SPREAD_LADDER = (1.20, 1.45, 1.90)
COARSEN_LADDER_PX = (3, 7, 14)
UNSHARP_LADDER = (0.0, 0.75, 1.5)
RESOLUTION_LADDER_UM = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0)
C3_DUTY_LADDER = (0.30, 0.40, 0.50, 0.60, 0.70)
BEAT_LADDER_UM = (800.0, 1635.0, 3000.0, 5000.0)

SWEEP_MM = 10.0
"""Every one-factor sweep cell. Uniform on purpose: the shelf packer's waste is
almost entirely mixed row heights, and bucketing the sizes took the plate from
133 mm of content (over a 119 mm budget) to under 105. 10 mm is 115 eye-cells
across at 300 mm — enough to judge tone, colour cast and screen visibility,
which is all a one-factor comparison is asked to settle."""
LADDER_MM = 5.0
"""C2 and C3 rungs. These are read under a loupe, not at arm's length."""
PAIR_MM = 11.0
"""Two-layer cells. Each occupies twice this plus a gutter, for its back die."""
C6_MM = 7.0
"""The beat ladder's rungs, and the one size chosen by the PACKER rather than
by optics. At 8.8 mm the fourth rung fell off the end of its row and took a
whole 9.7 mm shelf with it for 19 mm of content — 11 mm of the plate for one
cell. At 7 mm all four finish the row, and a 7 mm cell still shows 4.3 bands at
the shipping 1635 um beat, which is enough to count."""

PORTRAIT_MM = 30.0
"""Headline cell size. 30 mm is 344 eye-cells across at 300 mm — a photograph,
not a thumbnail — and three of them plus gutters leave room in the same row.
36 mm reads slightly better still, but the three headline cells then take 3888
of the plate's 14161 mm2 and the ladders below stop fitting."""


def _plan(mode: str, **kw: Any) -> cp.ColourPlan:
    """A variant of the reference plan with one knob moved."""
    base = {
        "plain": cp.PAULA_PLAIN,
        "hue": cp.PAULA_HUE,
        "zones": cp.PAULA_ZONES,
    }[mode]
    if not kw:
        return base
    d = {k: v for k, v in base.__dict__.items()}
    d.update(kw)
    return cp.ColourPlan(**d)


def _halftone_cell(
    cid: str, title: str, side_mm: float, *, mode: str = "zones",
    axis: str = "", level: str = "", note: str = "", label: str = "",
    plan_kw: dict[str, Any] | None = None,
    prep: ip.PrepSpec | None = None,
    line_period_um: float = REF_SCREEN_UM, tone_steps: int = REF_TONE_STEPS,
) -> Cell:
    plan = _plan(mode, **(plan_kw or {}))
    side = side_mm * MM

    def build(cx: float, cy: float, w: float, h: float) -> CellArt:
        return wc.build_halftone(
            cx, cy, w, h, plan=plan, prep=prep,
            line_period_um=line_period_um, tone_steps=tone_steps,
        )

    return Cell(cid=cid, title=title, group="B", w_um=side, h_um=side,
                build=build, note=note, axis=axis, level=level, label=label)


def _patch_cell(cid: str, title: str, side_mm: float, group: str,
                axis: str = "", level: str = "", note: str = "",
                label: str = "", **kw: Any) -> Cell:
    side = side_mm * MM

    def build(cx: float, cy: float, w: float, h: float) -> CellArt:
        return wc.build_grating_patch(cx, cy, w, h, **kw)

    return Cell(cid=cid, title=title, group=group, w_um=side, h_um=side,
                build=build, note=note, axis=axis, level=level, label=label)


def doe_cells() -> list[list[Cell]]:
    """Every cell on the plate, grouped into bands. Order IS the layout order."""
    bands: list[list[Cell]] = []

    # --- B6: the headline. Three variants, same size, same prep, same screen.
    bands.append([
        _halftone_cell("B6a", "portrait / plain gold", PORTRAIT_MM, mode="plain", label="B6a PLAIN",
                       axis="colour mode", level="plain",
                       note="the control: identical code path, empty period field"),
        _halftone_cell("B6b", "portrait / hue-mapped", PORTRAIT_MM, mode="hue", label="B6b HUE",
                       axis="colour mode", level="hue",
                       note="period from each pixel's own hue, whole frame"),
        _halftone_cell("B6c", "portrait / zone-mapped", PORTRAIT_MM, mode="zones", label="B6c ZONES",
                       axis="colour mode", level="zones",
                       note="flowers by hue, sweater and glasses authored"),
    ])

    # --- scale ladder: the one question a single size cannot answer.
    bands.append([
        _halftone_cell(f"SZ{mm:g}", f"scale {mm:g} mm", mm, mode="zones",
                       label=f"SZ {mm:g}mm",
                       axis="image scale", level=f"{mm:g} mm",
                       note=f"{int(mm*1000/87)} eye-cells across at 300 mm")
        for mm in SCALE_LADDER_MM
    ] + [
        _halftone_cell(f"SP{p:g}", f"screen {p:g} um", SWEEP_MM, mode="zones",
                       label=f"SP {p:g}um",
                       line_period_um=p,
                       tone_steps=min(REF_TONE_STEPS, int(p / 2.0)),
                       axis="screen pitch", level=f"{p:g} um",
                       note=f"tone depth capped at {int(p/2.0)} steps by the 2 um floor")
        for p in SCREEN_LADDER_UM
    ])

    # --- tone depth and colour base period.
    bands.append([
        _halftone_cell(f"TS{n}", f"{n} tone steps", SWEEP_MM, mode="zones",
                       label=f"TS {n}",
                       tone_steps=n, axis="tone steps", level=str(n),
                       note=f"finest band {REF_SCREEN_UM/n:.2f} um")
        for n in STEPS_LADDER
    ] + [
        _halftone_cell(f"CP{d:g}", f"colour base {d:g} um", SWEEP_MM, mode="zones",
                       label=f"CP {d:.1f}um", plan_kw={"base_period_um": d},
                       axis="colour base period", level=f"{d:g} um",
                       note="gated on C2 below 4.8 um" if d < 4.82 else "")
        for d in BASE_PERIOD_LADDER_UM
    ])

    # --- the three shading knobs that only exist because of the period field.
    bands.append(
        [
            _halftone_cell(f"DU{c:.2f}", f"sub-duty {c:.2f}", SWEEP_MM, mode="zones",
                           label=f"DU {c:.2f}",
                           plan_kw={"duty": c}, axis="sub-grating duty",
                           level=f"{c:.2f}",
                           note="saturation vs brightness; also moves the line width")
            for c in DUTY_LADDER
        ] + [
            _halftone_cell(f"SR{s:.2f}", f"spread {s:.2f}", SWEEP_MM, mode="zones",
                           label=f"SR {s:.2f}",
                           plan_kw={"spread": s}, axis="ladder spread",
                           level=f"{s:.2f}",
                           note="red/blue period ratio — how far apart two zones read")
            for s in SPREAD_LADDER
        ] + [
            _halftone_cell(f"GR{px}", f"coarsen {px} px", SWEEP_MM, mode="zones",
                           label=f"GR {px}px",
                           plan_kw={"coarsen_px": px}, axis="colour coarsening",
                           level=f"{px} px",
                           note="patch size of the colour field — noise vs intent")
            for px in COARSEN_LADDER_PX
        ]
    )

    # --- prep gains, and the two single-layer B cells.
    prep_cells = [
        _halftone_cell(f"PU{a:.2f}", f"unsharp {a:.2f}", SWEEP_MM, mode="zones",
                       label=f"PU {a:.2f}",
                       prep=ip.PrepSpec(tone_steps=REF_TONE_STEPS, unsharp_amount=a),
                       axis="prep: local contrast", level=f"{a:.2f}",
                       note="the screen discards detail below its pitch")
        for a in UNSHARP_LADDER
    ] + [
        _halftone_cell("PF0", "no falloff", SWEEP_MM, mode="zones", label="PF off",
                       prep=ip.PrepSpec(tone_steps=REF_TONE_STEPS, falloff=0.0),
                       axis="prep: subject falloff", level="off"),
        _halftone_cell("PF30", "falloff 0.30", SWEEP_MM, mode="zones", label="PF 0.30",
                       prep=ip.PrepSpec(tone_steps=REF_TONE_STEPS, falloff=0.30),
                       axis="prep: subject falloff", level="0.30"),
        _halftone_cell("PL0", "no linearise", SWEEP_MM, mode="zones", label="PL off",
                       prep=ip.PrepSpec(tone_steps=REF_TONE_STEPS, linearize=False),
                       axis="prep: linearisation", level="off",
                       note="the classic error: a 0.25 midtone prints as 0.54"),
    ]
    bands.append(prep_cells + [
        _patch_cell("B1", "rainbow 4.4 um @ 45", SWEEP_MM, "B", period_um=4.4,
                    angle_deg=45.0, note="the shipping accent period"),
        Cell("B2", "chirp 22 -> 3 um", "B", 2 * SWEEP_MM * MM, SWEEP_MM * MM,
             lambda cx, cy, w, h: wc.build_chirp(cx, cy, w, h),
             note="graded fan; also a continuous resolution check"),
    ])

    # --- Group D: colour with the photograph taken out of it.
    d_cells = [
        Cell(f"D{i+1}", t, "D", SWEEP_MM * MM, SWEEP_MM * MM,
             (lambda d=d, ht=ht: (lambda cx, cy, w, h: wc.build_colour_band(
                 cx, cy, w, h, period_um=d, hold_tone=ht)))(),
             note=n)
        for i, (t, d, ht, n) in enumerate([
            ("band + 5.0 um", 5.0, True, "does a gratinged band hold its tone?"),
            ("band + 4.15 um", 4.15, True, "blue end of the ladder"),
            ("band + 6.02 um", 6.02, True, "red end of the ladder"),
            ("band, tone not held", 5.0, False, "exact bands, darker — the control"),
        ])
    ]
    bands.append(d_cells + [
        _patch_cell("D5", "solid 5.0 um patch", SWEEP_MM, "D", period_um=5.0,
                    note="no halftone at all — the colour control"),
        _patch_cell("D6", "solid 4.0 um patch", SWEEP_MM, "D", period_um=4.0,
                    note="finer: holds colour under a wider source"),
    ])

    # --- Group C ladders.
    bands.append([
        _patch_cell(f"C2{chr(97+i)}", f"{d:g} um", LADDER_MM, "C", period_um=d,
                    axis="C2 resolution", level=f"{d:g} um",
                    note="below the DRC floor" if d < 4.0 else "")
        for i, d in enumerate(RESOLUTION_LADDER_UM)
    ] + [
        _patch_cell(f"C3{chr(97+i)}", f"duty {c:.2f}", LADDER_MM, "C",
                    period_um=8.0, duty=c, axis="C3 duty", level=f"{c:.2f}",
                    note="2nd order nulls here" if abs(c - 0.5) < 1e-9 else "")
        for i, c in enumerate(C3_DUTY_LADDER)
    ])

    # --- two-layer band: everything that needs a bonded pair.
    two: list[Cell] = [
        Cell("A1", "barrier image switch", "A", PAIR_MM * MM, PAIR_MM * MM,
             lambda cx, cy, w, h: wc.build_barrier_switch(cx, cy, w, h),
             note="quarter-period registered: clean image head-on", two_layer=True),
        Cell("A2", "scanimation, 4 phase", "A", PAIR_MM * MM, PAIR_MM * MM,
             lambda cx, cy, w, h: wc.build_scanimation(cx, cy, w, h),
             note="tightest cell on the plate — wants 4x A1's registration",
             two_layer=True),
        Cell("A4", "shading moire", "A", PAIR_MM * MM, PAIR_MM * MM,
             lambda cx, cy, w, h: wc.build_shading_moire(cx, cy, w, h),
             note="also stands in for A5: rotate the back die a few degrees",
             two_layer=True),
        Cell("A6", "moire magnifier", "A", PAIR_MM * MM, PAIR_MM * MM,
             lambda cx, cy, w, h: wc.build_moire_magnifier(cx, cy, w, h),
             note="sampler is gold WITH holes, not sparse dots", two_layer=True),
    ] + [
        Cell(f"C6{chr(97+i)}", f"beat {b:g} um", "C", C6_MM * MM, C6_MM * MM,
             (lambda b=b: (lambda cx, cy, w, h: wc.build_shading_moire(
                 cx, cy, w, h, beat_um=b)))(),
             axis="C6 beat", level=f"{b:g} um", two_layer=True,
             note=f"{C6_MM*MM/b:.1f} bands across this cell")
        for i, b in enumerate(BEAT_LADDER_UM)
    ]
    bands.append(two)
    return bands


# --- layout -----------------------------------------------------------------


@dataclass
class Placed:
    cell: Cell
    cx: float
    cy: float
    art: CellArt = field(default_factory=CellArt)
    pair_cx: float | None = None
    """Centre of the BACK die, for a two-layer cell."""


def layout(bands: Iterable[list[Cell]]) -> tuple[list[Placed], dict[str, Any]]:
    """Shelf-pack the bands top-down, wrapping a band that overruns the width.

    Deliberately a packer rather than a hand-placed grid: cell sizes are still
    being tuned, and a hand-placed map goes stale the moment one ladder gains a
    rung. The order within a band is preserved so a ladder always reads
    left-to-right.
    """
    x_lo = -USABLE_UM / 2.0
    y = USABLE_UM / 2.0
    placed: list[Placed] = []
    rows: list[dict[str, Any]] = []
    overflow: list[str] = []

    def flush(row: list[Cell], y: float) -> float:
        if not row:
            return y
        row_h = max(c.h_um for c in row) + LABEL_H_UM
        x = x_lo
        for c in row:
            span = c.w_um * (2.0 if c.two_layer else 1.0) + (
                GUTTER_UM if c.two_layer else 0.0
            )
            pl = Placed(cell=c, cx=x + c.w_um / 2.0,
                        cy=y - LABEL_H_UM - c.h_um / 2.0)
            if c.two_layer:
                pl.pair_cx = pl.cx + c.w_um + GUTTER_UM
            placed.append(pl)
            x += span + GUTTER_UM
        rows.append({
            "y_top_mm": round(y / MM, 2),
            "height_mm": round(row_h / MM, 2),
            "cells": [c.cid for c in row],
        })
        return y - row_h - GUTTER_UM

    # Packed CONTINUOUSLY across bands, not flushed at every band boundary.
    # Flushing per band cost 54 mm of the 119 available to half-empty rows —
    # the D row used 73 mm of width and the next ladder started below it anyway.
    # Order is still the band order, so a ladder stays contiguous and reads
    # left to right; it may simply wrap mid-ladder, which the map makes clear.
    row: list[Cell] = []
    row_w = 0.0
    for band in bands:
        for c in band:
            w = c.w_um * (2.0 if c.two_layer else 1.0) + (
                GUTTER_UM if c.two_layer else 0.0
            )
            if row and row_w + GUTTER_UM + w > USABLE_UM:
                y = flush(row, y)
                row, row_w = [], 0.0
            row.append(c)
            row_w += (GUTTER_UM if row_w else 0.0) + w
    y = flush(row, y)

    used = USABLE_UM / 2.0 - y
    if used > USABLE_UM:
        overflow = [p.cell.cid for p in placed
                    if p.cy - p.cell.h_um / 2.0 < -USABLE_UM / 2.0]
    return placed, {
        "rows": rows,
        "height_used_mm": round(used / MM, 2),
        "height_available_mm": round(USABLE_UM / MM, 2),
        "fits": not overflow,
        "overflow": overflow,
    }


# --- build ------------------------------------------------------------------


def build_plate(
    bands: Iterable[list[Cell]] | None = None, *, verbose: bool = True
) -> dict[str, Any]:
    """Run every cell builder and collect the plate's geometry and manifest."""
    bands = list(bands if bands is not None else doe_cells())
    placed, lay = layout(bands)
    front: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    outline: list[np.ndarray] = []
    pair_marks: list[np.ndarray] = []
    arrays: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for p in placed:
        c = p.cell
        t0 = time.perf_counter()
        art = c.build(p.cx, p.cy, c.w_um, c.h_um)
        p.art = art
        front.append(art.front)
        for a in art.arrays:
            arrays.append(a)
        outline.append(_frame_rects(p.cx, p.cy, c.w_um, c.h_um))
        labels.append(_text_rects(c.label or c.cid, p.cx, p.cy + c.h_um / 2.0 + LABEL_H_UM * 0.5,
                                  LABEL_H_UM * 0.62))
        n_back = 0
        if c.two_layer and p.pair_cx is not None:
            dx = p.pair_cx - p.cx
            b = art.back.copy()
            if len(b):
                b[:, 0] += dx
                b[:, 1] += dx
                front.append(b)
            n_back = len(b)
            pair_marks.append(_frame_rects(p.pair_cx, p.cy, c.w_um, c.h_um))
            outline.append(_frame_rects(p.pair_cx, p.cy, c.w_um, c.h_um))
            labels.append(_text_rects(
                (c.label or c.cid) + " B", p.pair_cx, p.cy + c.h_um / 2.0 + LABEL_H_UM * 0.5,
                LABEL_H_UM * 0.62))
        n_arr = sum(int(np.size(a["period_um"])) for a in art.arrays)
        dt = time.perf_counter() - t0
        manifest.append({
            "cid": c.cid, "title": c.title, "group": c.group,
            "axis": c.axis, "level": c.level, "note": c.note,
            "x_mm": round(p.cx / MM, 3), "y_mm": round(p.cy / MM, 3),
            "w_mm": round(c.w_um / MM, 3), "h_mm": round(c.h_um / MM, 3),
            "two_layer": c.two_layer,
            "n_rects": int(len(art.front)) + n_back,
            "n_array_bands": n_arr,
            "build_s": round(dt, 2),
            "stats": art.stats,
        })
        if verbose:
            print(f"  {c.cid:5s} {c.title[:34]:34s} "
                  f"{len(art.front)+n_back:>9,} rects  {n_arr:>8,} arrays  {dt:5.1f}s")

    return {
        "placed": placed,
        "layout": lay,
        "front": _cat(*front),
        "outline": _cat(*outline),
        "labels": _cat(*labels),
        "pair_marks": _cat(*pair_marks),
        "arrays": arrays,
        "manifest": manifest,
    }


def flat_rect_count(plate: dict[str, Any]) -> int:
    """Rectangles the plate would hold with every array expanded."""
    from .patterns.bitmap import screenrects as sr

    n = int(len(plate["front"]))
    for a in plate["arrays"]:
        n += sr.stripe_plan(a["rects"], a["period_um"],
                            a["line_um"] / a["period_um"])["total"]
    return n


# --- writers ----------------------------------------------------------------


def _insert_rects(cell: Any, layer: int, rects: np.ndarray, kdb: Any) -> None:
    shapes = cell.shapes(layer)
    for x0, x1, y0, y1 in rects:
        shapes.insert(kdb.DBox(x0, y0, x1, y1))


def _save_options(suffix: str, kdb: Any) -> Any:
    """Writer settings per format.

    OASIS with CBLOCKs is the whole trick: 158 MB of GDSII becomes 6.5 MB, a
    24x saving with no geometry change at all. Two reasons it wins so heavily
    here. GDSII spends a fixed ~64 bytes on every rectangle — a BOUNDARY record
    plus five 4-byte coordinate PAIRS to describe four numbers — while OASIS has
    a native rectangle record with delta-encoded varint coordinates; that alone
    is the 4.7x you get before any compression. CBLOCKs then deflate each cell,
    and a halftone's coordinates are enormously redundant (every band on a line
    shares two y values, every stripe steps by one period).

    ``oasis_recompress`` — klayout's search for repetitions it can turn into
    OASIS repetition records — is deliberately left OFF. It was measured at
    exactly the same 6.5 MB, because the periodic geometry has already been
    arrayed by hand in :func:`write_mask`; there is nothing left for it to find,
    and it is not free to run.
    """
    o = kdb.SaveLayoutOptions()
    if suffix == ".oas":
        o.format = "OASIS"
        o.oasis_compression_level = 2
        o.oasis_write_cblocks = True
    else:
        o.format = "GDS2"
        o.gds2_write_timestamps = False   # byte-identical rebuilds
    return o


def write_mask(
    plate: dict[str, Any],
    out_path: Path,
    *,
    flat: bool = False,
    formats: Sequence[str] = (".oas",),
) -> list[Path]:
    """Write the plate as OASIS and/or GDSII.

    ``out_path`` names the stem; one file is written per entry in ``formats``.
    OASIS is the default because it is both the smaller and the more modern
    interchange format, and every current mask shop and pattern generator reads
    it; GDSII is written alongside on request for a flow that will not.

    Periodic sub-gratings go in as ARRAY REFERENCES: one unit cell per distinct
    (line width, band height) pair — about 260 of them, because band heights are
    quantised to the screen's tone ladder — and one instance per band. Flattened
    the same geometry is millions of rectangles, which is the difference between
    a file a mask shop can open and one it cannot.

    The two end stripes of each band are clipped by the band edge, so they are
    emitted as explicit boxes and the array covers only the interior.
    """
    import klayout.db as kdb

    ly = kdb.Layout()
    ly.dbu = 0.001
    top = ly.create_cell("WITNESS_5IN")
    l_front = ly.layer(*LAYER_FRONT)
    l_out = ly.layer(*LAYER_OUTLINE)
    l_pair = ly.layer(*LAYER_PAIR)

    _insert_rects(top, l_front, plate["front"], kdb)
    _insert_rects(top, l_front, plate["labels"], kdb)
    _insert_rects(top, l_out, plate["outline"], kdb)
    _insert_rects(top, l_pair, plate["pair_marks"], kdb)
    half = PLATE_SIDE_UM / 2.0
    top.shapes(l_out).insert(kdb.DBox(-half, -half, half, half))

    n_inst = 0
    n_box = 0
    if flat:
        from .patterns.bitmap import screenrects as sr

        for a in plate["arrays"]:
            r = sr.stripe_rects(a["rects"], a["period_um"],
                                a["line_um"] / a["period_um"],
                                max_rects=200_000_000)
            _insert_rects(top, l_front, r, kdb)
            n_box += len(r)
    else:
        unit: dict[tuple[int, int], Any] = {}

        def unit_cell(line_nm: int, h_nm: int) -> Any:
            key = (line_nm, h_nm)
            c = unit.get(key)
            if c is None:
                c = ly.create_cell(f"STRIPE_{line_nm}_{h_nm}")
                c.shapes(l_front).insert(
                    kdb.DBox(0.0, 0.0, line_nm / 1000.0, h_nm / 1000.0)
                )
                unit[key] = c
            return c

        for a in plate["arrays"]:
            rects = np.asarray(a["rects"], dtype=np.float64)
            per = np.broadcast_to(np.asarray(a["period_um"], dtype=np.float64),
                                  (len(rects),))
            lin = np.broadcast_to(np.asarray(a["line_um"], dtype=np.float64),
                                  (len(rects),))
            phase = float(a.get("phase_um", 0.0))
            line = lin
            # Whole stripes only, centre-in — the same selection stripe_plan
            # makes, so the GDS matches any preview drawn from stripe_rects.
            k0 = np.ceil((rects[:, 0] - phase - line / 2.0) / per).astype(np.int64)
            k1 = np.floor((rects[:, 1] - phase - line / 2.0) / per).astype(np.int64)
            for i in range(len(rects)):
                y0, y1 = rects[i, 2], rects[i, 3]
                d, w_line = per[i], lin[i]
                n = int(k1[i] - k0[i] + 1)
                if n <= 0:
                    continue
                c = unit_cell(int(round(w_line * 1000)), int(round((y1 - y0) * 1000)))
                x_start = phase + k0[i] * d
                if n == 1:
                    top.insert(kdb.DCellInstArray(
                        c.cell_index(), kdb.DTrans(kdb.DVector(x_start, y0))))
                else:
                    top.insert(kdb.DCellInstArray(
                        c.cell_index(),
                        kdb.DTrans(kdb.DVector(x_start, y0)),
                        kdb.DVector(d, 0.0), kdb.DVector(0.0, 0.0), n, 1,
                    ))
                n_inst += 1

    stem = Path(out_path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sizes: dict[str, float] = {}
    for suffix in formats:
        p = stem.with_suffix(suffix)
        ly.write(str(p), _save_options(suffix, kdb))
        written.append(p)
        sizes[suffix.lstrip(".")] = round(p.stat().st_size / 1e6, 2)
    plate["gds"] = {
        "path": str(written[0]) if written else "",
        "paths": [str(p) for p in written],
        "flat": flat,
        "n_array_instances": n_inst,
        "n_edge_boxes": n_box,
        "n_unit_cells": 0 if flat else len(unit),
        "size_mb": sizes.get(written[0].suffix.lstrip(".") if written else "", 0.0),
        "size_mb_by_format": sizes,
    }
    return written


def write_gds(plate: dict[str, Any], out_path: Path, *, flat: bool = False) -> Path:
    """Back-compat single-file GDSII write."""
    return write_mask(plate, out_path, flat=flat, formats=(".gds",))[0]


def write_map_svg(plate: dict[str, Any], out_path: Path) -> Path:
    """A one-page plate map: where every cell is and what it is a rung of."""
    s = PLATE_SIDE_UM
    sc = 900.0 / s
    GROUP_FILL = {"A": "#7c4dff", "B": "#00b8a9", "C": "#f0a202", "D": "#e0457b"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 940" '
        f'width="900" height="940"><rect width="900" height="940" fill="#0d1113"/>',
        f'<rect x="0" y="0" width="900" height="900" fill="none" '
        f'stroke="#3a4449" stroke-width="2"/>',
    ]
    for m in plate["manifest"]:
        x = (m["x_mm"] * MM + s / 2) * sc
        y = (s / 2 - m["y_mm"] * MM) * sc
        w = m["w_mm"] * MM * sc
        h = m["h_mm"] * MM * sc
        fill = GROUP_FILL.get(m["group"], "#888")
        parts.append(
            f'<rect x="{x-w/2:.1f}" y="{y-h/2:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" fill-opacity="0.22" stroke="{fill}" stroke-width="1"/>'
        )
        if w > 26:
            parts.append(
                f'<text x="{x:.1f}" y="{y:.1f}" fill="#e6eef1" font-size="{min(13, w/4):.0f}" '
                f'font-family="monospace" text-anchor="middle">{m["cid"]}</text>'
            )
        if m["two_layer"]:
            parts.append(
                f'<rect x="{x-w/2+w+GUTTER_UM*sc:.1f}" y="{y-h/2:.1f}" '
                f'width="{w:.1f}" height="{h:.1f}" fill="{fill}" fill-opacity="0.10" '
                f'stroke="{fill}" stroke-dasharray="3 2" stroke-width="1"/>'
            )
    lg = plate["layout"]
    parts.append(
        f'<text x="10" y="918" fill="#9fb0b6" font-size="14" font-family="monospace">'
        f'5in witness plate &#183; {len(plate["manifest"])} cells &#183; '
        f'{lg["height_used_mm"]:.1f} of {lg["height_available_mm"]:.1f} mm used'
        f'</text>'
    )
    parts.append('</svg>')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/witness/witness-5in",
                    help="output stem; the suffix comes from --formats")
    ap.add_argument("--formats", default="oas",
                    help="comma list of oas,gds (default oas: 6.5 MB vs 158)")
    ap.add_argument("--flat", action="store_true",
                    help="expand array references into polygons (much larger)")
    ap.add_argument("--map", default="data/witness/witness-5in-map.svg")
    ap.add_argument("--manifest", default="data/witness/witness-5in.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report, write nothing")
    a = ap.parse_args(argv)

    print(f"building the {PLATE_SIDE_UM/MM:.0f} mm witness plate ...")
    t0 = time.perf_counter()
    plate = build_plate()
    lg = plate["layout"]
    print(f"\n{len(plate['manifest'])} cells, "
          f"{lg['height_used_mm']:.1f} / {lg['height_available_mm']:.1f} mm of height, "
          f"fits={lg['fits']}")
    n_flat = flat_rect_count(plate)
    print(f"rects: {len(plate['front']):,} explicit  |  "
          f"{n_flat:,} if fully flattened "
          f"({n_flat/max(1,len(plate['front'])):.1f}x)")

    if not a.dry_run:
        fmts = tuple("." + f.strip().lstrip(".") for f in a.formats.split(","))
        paths = write_mask(plate, Path(a.out), flat=a.flat, formats=fmts)
        for p_ in paths:
            mb_ = plate["gds"]["size_mb_by_format"][p_.suffix.lstrip(".")]
            print(f"mask {p_}  {mb_} MB")
        print(f"     {plate['gds']['n_array_instances']:,} array refs over "
              f"{plate['gds']['n_unit_cells']} unit cells")
        svg = write_map_svg(plate, Path(a.map))
        print(f"map  {svg}")
        mp = Path(a.manifest)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({
            "plate_side_um": PLATE_SIDE_UM,
            "layout": lg,
            "gds": plate.get("gds", {}),
            "flat_rect_count": n_flat,
            "cells": plate["manifest"],
        }, indent=2), encoding="utf-8")
        print(f"manifest {mp}")
    print(f"total {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
