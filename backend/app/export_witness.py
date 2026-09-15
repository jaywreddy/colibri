"""Lay out and write the 5-inch production plate: the dies, the dicing grid, the GDS.

The plate is the box. Its first claim is the production plies
(``witness_dies.production_cells``: the lid and the front as single-layer
diffraction mappings, every candidate photograph as a side, and spares), and
what is left of the field carries the bench cells that read THIS process:
polarity, CD, duty, the diffraction period ladder, the halftone wedge and the
acuity ladder. Every cell is one gold layer on one ply — the two-layer
experiments went with the bonded design (2026-09-15).

The layout is a DICING GRID, not a packer
-----------------------------------------
The first plate was cleaved by hand and broke; this one is cut on a saw, and a
saw cuts straight lines across whatever is in front of it. So the plate is laid
out in ROWS of one height with full-width horizontal streets between them,
and inside a row the dies sit side by side with full-height vertical streets:

    1. cut every horizontal street across the whole plate  -> strips
    2. cut every vertical street of a strip across the strip -> dies
    3. a strip's leftover may carry a COLUMN of short cells; those pieces get
       their own horizontal cuts once the column piece is free

``layout`` places the cells that way and returns the cut list (``dicing``) in
millimetres from the plate centre; ``write_dicing_md`` writes it out for the
saw operator and ``write_map_svg`` draws it. Streets are ``GUTTER_UM`` wide
(1 mm: a 0.3 mm blade with room for its wander); the corner L-ticks every die
carries reach into them so the operator sees where the lines run.

One gold layer
--------------
Every cell is on ``LAYER_FRONT``. The plate is a darkfield write with positive
resist and the file holds the openings (``polarity=CLEAR``, see witness_dies).

Rectangle economy
-----------------
Geometry is rect-space throughout and periodic sub-gratings are deferred as
ARRAY REFERENCES rather than polygons — see :func:`write_gds`. ``--flat``
expands everything for a shop that will not take arrays, and prints what that
costs.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import witness_cells as wc
from . import witness_moire as wm
from .patterns.bitmap import colourplan as cp
from .witness_geom import (
    BOX_COMB_UM,
    CLEAR,
    GLASS_MATERIAL,
    METAL,
    PARALLAX_UM_PER_DEG,
    PLY_UM,
    P_MIN_UM,
    fresnel_number,
    GUTTER_UM,
    LABEL_H_UM,
    LAYER_FRONT,
    LAYER_OUTLINE,
    PLATE_SIDE_UM,
    USABLE_UM as _GEOM_USABLE_UM,
    Cell,
    CellArt,
    _cat,
    _frame_rects,
    _text_rects,
)

LAYER_PAIR = (21, 0)     # outline-only: marks the back die of a two-layer experiment cell
LAYER_DICE = (2, 0)      # annotation: the saw's street centrelines (never chrome)

TICK_REACH_UM = 550.0
"""How far a die's corner dicing tick reaches into the street beyond the die
edge (``export_blank.DICE_TICK_GAP_UM + DICE_TICK_LEN_UM``). The packer keeps
every cell this much inside the geometric usable square, so no written shape —
tick included — lands in the blank's 4 mm edge margin."""
USABLE_UM = _GEOM_USABLE_UM - 2.0 * TICK_REACH_UM
MM = 1000.0
ROW_GUTTER_UM = GUTTER_UM
"""Between rows: a real saw street, the same width as the one between dies.
Every row boundary IS a cut on this plate."""
ROW_SLACK_UM = 1500.0
"""A cell may ride a row up to this much taller than itself (top-aligned; the
strip is cut at the row's height and the piece trimmed, or, for a bench cell,
simply left a little tall). Production dies never use it — they define their
rows — so their cut dimensions stay exact."""


def _label_h(cell_h_um: float) -> float:
    """Gold label band for a row of this height. Proportional, floored so it
    stays legible under a loupe and capped so a big cell does not waste a
    millimetre on a five-character name."""
    return max(400.0, min(LABEL_H_UM, 0.10 * cell_h_um))

# --- the design of experiments ----------------------------------------------
#
# Derived from docs/witness-physics-plan.md, and organised by MECHANISM rather
# than by effect. The ordering principle is section 0 of that plan: a cell earns
# its area by returning a number that could change the box design, not by
# demonstrating that something works.
#
# An earlier revision spent 40% of the plate on photographic portraits, one per
# parameter. A portrait is a subjective read of many coupled variables at once,
# and H-WEDGE gave the same information objectively in a twentieth of the area;
# the portraits went, and the subjective call -- which picture goes on a wall --
# is now made on the production SIDE DIES themselves (witness_dies.SIDE_PHOTOS).
#
# 2026-09-10: the plate carries ten production plies (lid + front pairs, six
# photo sides), so the experiment set is cut to the cells this box's bench
# reads: polarity, CD, duty, the two-layer registration and switch cells at the
# box's own pitches, one near-field pair either side of the design, one halftone
# wedge at the photo screen, and single rungs of the diffraction and moire
# ladders. 23 cells.
#
# What went is NOT kept as an empty tuple with a live loop over it: that is a
# cell which silently does not exist, and it is how a plate ships missing an
# experiment nobody noticed. The dropped rungs -- BEAT, BEAT-duty, ROTATION,
# HARMONIC, SCREEN-angle, the D-SWATCH base-period x spread grid, the CROSS and
# BAND swatches, the VEC pitch x angle grid, the MAG magnifier pair, P-SCAN, and
# the portrait ladders (steps, duty, coarsen, unsharp, scale) -- live in git
# history at 0799344, and their builders in witness_cells' EXPERIMENTS section.
RESOLUTION_LADDER_UM = (0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0)
C3_DUTY_LADDER = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
PERIOD_LADDER_UM = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0)
NEAR_FIELD_LADDER_UM = (20.0, 64.0)
"""Kept for the physics tests (``wm.build_near_field`` and the Fresnel
boundary): no near-field cell rides the plate since the two-layer design went."""
SCREEN_LADDER_UM = (20.0, 30.0, 44.0, 60.0)
WEDGE_SCREENS_UM = (44.0,)          # the photo screen; 20 and 60 were cut

BEAT_H_MM = 8.0
"""Height of every beat cell. Beat fringes are spaced along ONE axis — across
the lines for a pitch beat, along them for a rotation beat — so these cells need
WIDTH, not area. Drawn square, the 6000 um rung forced a 20.9 mm row that was
33% full and cost the plate 10 mm of height for one cell."""


def _beat_w_mm(beat_um: float, min_mm: float = 6.0, fringes: float = 5.0) -> float:
    """Width that holds ``fringes`` of a beat, floored so a fine beat still gets
    a cell you can put an eye to.

    Five fringes, not three: the cell is read by COUNTING fringes and inverting
    back to the pitch error through ``p/delta``, so the count is the measurement
    and one more fringe is one more significant figure. Reshaping these cells
    wide-and-short freed the height to afford it."""
    return max(min_mm, round(fringes * beat_um / 1000.0, 1))

PORTRAIT_MM = 12.0
"""Cell size for a halftone PORTRAIT: 138 eye-cells across at 300 mm, enough to
judge a colour treatment. No portrait rides this plate any more — the two colour
SIDES are the portraits, at their 13.3 mm art box — but
``tools/dev/render_witness_preview.py``
renders its off-plate previews at this size, so the number stays here where the
plate's other cell sizes are."""
SWEEP_MM = 10.0
PAIR_MM = 8.0
WEDGE_W_MM = 30.0
WEDGE_H_MM = 4.0


def _cell(cid, title, block, w_mm, h_mm, build, *, label="", note="",
          axis="", level="", two_layer=False, takes_polarity=False) -> Cell:
    return Cell(cid=cid, title=title, group=block[0].upper(),
                w_um=w_mm * MM, h_um=h_mm * MM, build=build, note=note,
                label=label or cid, axis=axis, level=level,
                two_layer=two_layer, block=block, takes_polarity=takes_polarity)


def _plan(mode: str, **kw: Any) -> cp.ColourPlan:
    """A variant of the reference colour plan with one knob moved.

    No cell on this plate takes a plan any more (the colour question is answered
    by the photo SIDE dies, which carry their own); it is kept for
    ``tools/dev/render_witness_preview.py``, which renders the three treatments
    off-plate from the same reference plans."""
    base = {"plain": cp.PAULA_PLAIN, "hue": cp.PAULA_HUE, "zones": cp.PAULA_ZONES}[mode]
    if not kw:
        return base
    d = {k: v for k, v in base.__dict__.items()}
    d.update(kw)
    return cp.ColourPlan(**d)


def doe_cells() -> list[list[Cell]]:
    """Every cell on the plate, grouped into blocks. Order IS the layout order."""
    from .witness_dies import production_cells

    B: list[list[Cell]] = []

    # === PRODUCTION -- four box faces that come off the plate as plies =======
    # Laid out first so they take the top of the plate whole; the packer fills
    # the width beside them with experiment cells (see ``layout``'s pockets).
    B.append(production_cells())

    # === METROLOGY -- read these first ======================================
    m: list[Cell] = [
        _cell("M-POL", "polarity witness", "metrology", 6.0, 6.0,
              lambda cx, cy, w, h: wm.build_polarity_witness(cx, cy, w, h),
              label="M-POL CLEAR",
              note="square-with-a-hole: unmistakable if the write is inverted"),
    ]
    # Ladders are ONE cell each, not one per rung: see build_ladder_strip for
    # why that bought back a third of the plate.
    for dense in (True, False):
        tag = "dense" if dense else "iso"
        m.append(_cell(
            f"M-CD-{tag}", f"CD ladder, {tag}", "metrology", 36.0, 5.0,
            (lambda dense=dense: (lambda cx, cy, w, h, polarity=METAL:
                wm.build_ladder_strip(
                    cx, cy, w, h, polarity=polarity,
                    rungs=wm.cd_rungs(RESOLUTION_LADDER_UM, dense=dense))))(),
            label=f"M-CD 0.8-8um {tag}", axis="M-CD resolution",
            level="0.8 -> 8.0 um", takes_polarity=True,
            note="finest rung that resolves. Iso and dense do not print alike"))
    for pd in (10.0, 5.0):
        m.append(_cell(
            f"M-DUTY{pd:g}", f"duty ladder at {pd:g} um", "metrology",
            26.0, 5.0,
            (lambda pd=pd: (lambda cx, cy, w, h, polarity=METAL:
                wm.build_ladder_strip(cx, cy, w, h, polarity=polarity,
                                      rungs=wm.duty_rungs(pd, C3_DUTY_LADDER))))(),
            label=f"M-DUTY .30-.70 @{pd:g}um", axis="M-DUTY bias", level=f"{pd:g} um",
            takes_polarity=True,
            note="the 0.50 rung must show no 2nd order; 0.40 vs 0.60 gives the sign"))
    B.append(m)

    # === DIFFRACTION ========================================================
    d_cells: list[Cell] = [
        _cell("D-PER", "period ladder", "diffraction", 36.0, 6.0,
              lambda cx, cy, w, h, polarity=METAL: wm.build_ladder_strip(
                  cx, cy, w, h, polarity=polarity,
                  rungs=wm.period_rungs(PERIOD_LADDER_UM)),
              label="D-PER 2-20um", axis="D-PER period", level="2 -> 20 um",
              takes_polarity=True,
              note="hue and fan width against period. Read under a lamp AND "
                   "under room light -- the difference is the source-width result"),
    ]
    B.append(d_cells)

    # === HALFTONE ===========================================================
    hf: list[Cell] = []
    for sp in WEDGE_SCREENS_UM:
        hf.append(_cell(
            f"WEDGE{sp:g}", f"step wedge, {sp:g} um screen", "halftone",
            WEDGE_W_MM, WEDGE_H_MM,
            (lambda sp=sp: (lambda cx, cy, w, h, polarity=METAL:
                wm.build_step_wedge(cx, cy, w, h, line_period_um=sp,
                                    tone_steps=min(22, int(sp / 2)),
                                    polarity=polarity)))(),
            label=f"WEDGE {sp:g}um", axis="H-WEDGE screen", level=f"{sp:g} um",
            takes_polarity=True,
            note="THE dot-gain instrument; inverts into the prep's gain"))
    hf.append(_cell(
        "H-ACU", "acuity ladder", "halftone", 26.0, 5.0,
        lambda cx, cy, w, h, polarity=METAL: wm.build_ladder_strip(
            cx, cy, w, h, polarity=polarity, rungs=wm.period_rungs(SCREEN_LADDER_UM)),
        label="H-ACU 20-60um", axis="H-ACU pitch", level="20 -> 60 um",
        takes_polarity=True,
        note="at what pitch do you SEE the lines? 43.5 um is the calculation"))
    # No portrait cells: the two colour SIDES (DIE-LEFT zones, DIE-RIGHT hue)
    # are the portraits now, at their 13.3 mm art box, and the plain control
    # is the H-WEDGE.
    B.append(hf)

    # === TWO-LAYER ==========================================================
    # Gone with the bonded design (2026-09-15). M-VERN, P-RULE, B-MOVE, the NF
    # and SWAP ladders measured a gap the box no longer has; their builders
    # stay in witness_cells / witness_moire (the physics tests still pin them).
    return B


def clear_by_boolean(
    cx: float, cy: float, w: float, h: float,
    rects: np.ndarray, polys: np.ndarray,
) -> list[np.ndarray]:
    """Cell box minus its metal, as polygons — the generic clear-field inverse.

    Used for every cell that does not know its own inverse in closed form. It IS
    a GEOS-style boolean, which CLAUDE.md restricts, and that restriction is
    honoured by scope: this runs over ONE cell of a few hundred to a few thousand
    shapes, never over the plate. The cells that would make it a whole-plate
    operation — the halftone portraits, at 10^5 to 10^6 shapes each — invert
    analytically instead and never reach here.

    The guard below is what keeps that true rather than merely intended.
    """
    import klayout.db as kdb

    n = len(rects) + len(polys)
    if n > wm.MAX_BOOLEAN_SHAPES:
        raise ValueError(
            f"cell at ({cx:.0f}, {cy:.0f}) has {n:,} shapes; the per-cell "
            f"boolean inverter caps at {wm.MAX_BOOLEAN_SHAPES:,}. A cell this "
            f"large must supply its own analytic complement (takes_polarity)."
        )
    dbu = 0.001
    def _i(v: float) -> int:
        return int(round(v / dbu))

    box = kdb.Region(kdb.Box(_i(cx - w / 2), _i(cy - h / 2),
                             _i(cx + w / 2), _i(cy + h / 2)))
    metal = kdb.Region()
    for x0, x1, y0, y1 in np.asarray(rects, dtype=np.float64):
        metal.insert(kdb.Box(_i(x0), _i(y0), _i(x1), _i(y1)))
    for pv in polys:
        metal.insert(kdb.Polygon([kdb.Point(_i(x), _i(y))
                                  for x, y in np.asarray(pv, dtype=np.float64)]))
    metal.merge()
    out: list[np.ndarray] = []
    for poly in (box - metal).each():
        flat = poly.dup()
        flat.resolve_holes()
        out.append(np.array([[pt.x * dbu, pt.y * dbu]
                             for pt in flat.each_point_hull()], dtype=np.float64))
    return out


# --- layout -----------------------------------------------------------------


@dataclass
class Placed:
    cell: Cell
    cx: float
    cy: float
    art: CellArt = field(default_factory=CellArt)
    pair_cx: float | None = None
    """Centre of the BACK die, for a two-layer cell."""
    back_free: list = field(default_factory=list)
    """Clear-field polygons for the BACK die, when it was inverted by boolean."""


def _span(c: Cell) -> float:
    """Width a cell occupies in a row: its die, plus the back die and a street."""
    if c.two_layer:
        return c.w_um + GUTTER_UM + c.back_dims[0]
    return c.w_um


def _cell_h(c: Cell) -> float:
    """Height a cell costs in a row: the taller die plus its label band."""
    h = max(c.h_um, c.back_dims[1])
    return h + _label_h(h)


@dataclass
class _Column:
    """A stack of short cells in a row's leftover width. The column is one
    piece after the strip's vertical cuts; its own horizontal cuts free the
    cells (step 3 of the dicing protocol)."""
    x0: float
    w: float
    y_top: float
    y_bot: float
    y: float = 0.0
    cells: list = field(default_factory=list)
    cuts: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.y = self.y_top

    def try_place(self, c: Cell) -> tuple[float, float] | None:
        if _span(c) > self.w + 1e-6:
            return None
        h = _cell_h(c)
        y_top = self.y if not self.cells else self.y - ROW_GUTTER_UM
        if y_top - h < self.y_bot - 1e-6:
            return None
        if self.cells:
            self.cuts.append(self.y - ROW_GUTTER_UM / 2.0)
        self.cells.append(c.cid)
        self.y = y_top - h
        return self.x0, y_top


@dataclass
class _Row:
    """One dicing strip: cells of (about) one height side by side."""
    h: float
    y_top: float
    x: float
    cells: list = field(default_factory=list)
    xs: list = field(default_factory=list)      # x0 of every die / column piece
    x1s: list = field(default_factory=list)     # x1 of every die / column piece
    columns: list = field(default_factory=list)

    @property
    def y_bot(self) -> float:
        return self.y_top - self.h


def layout(bands: Iterable[list[Cell]]) -> tuple[list[Placed], dict[str, Any]]:
    """Place the cells as a DICING GRID and return the cut list.

    Rows are opened by the tallest cells first and each row holds cells of its
    own height (a bench cell may be up to ``ROW_SLACK_UM`` shorter than its
    row); a cell that finds no row of its height and no height budget for a
    new one stacks into a COLUMN in the leftover width of an existing row.
    Horizontal streets run the full plate width between rows; vertical streets
    run the full strip height between dies (and column pieces); a column's own
    cuts are listed with it. Within a height class the order of the bands is
    kept, so a ladder still reads left to right.

    A two-layer experiment cell's back die sits one street to the right of its
    front die at the same y (its own size); no production cell is two-layer.
    """
    x_lo = -USABLE_UM / 2.0
    x_hi = USABLE_UM / 2.0
    y_hi = USABLE_UM / 2.0
    y_lo = -USABLE_UM / 2.0
    placed: list[Placed] = []
    rows: list[_Row] = []
    overflow: list[str] = []

    def put(c: Cell, x: float, y_top: float, row: _Row | None) -> None:
        pl = Placed(cell=c, cx=x + c.w_um / 2.0, cy=y_top - c.h_um / 2.0)
        if c.two_layer:
            pl.pair_cx = x + c.w_um + GUTTER_UM + c.back_dims[0] / 2.0
        placed.append(pl)
        if row is not None:
            row.cells.append(c.cid)

    def rows_h() -> float:
        return sum(r.h for r in rows) + ROW_GUTTER_UM * max(0, len(rows) - 1)

    cells = [c for band in bands for c in band]
    # tallest first; stable, so a band's own order survives inside a class
    order = sorted(range(len(cells)), key=lambda i: -_cell_h(cells[i]))
    for i in order:
        c = cells[i]
        span, h = _span(c), _cell_h(c)
        # 1. a row of this height (or a little taller, for a bench cell) with room
        home = None
        for r in rows:
            slack = r.h - h
            if 0 <= slack <= (ROW_SLACK_UM if c.block != "production" else 1e-6) \
                    and r.x + (GUTTER_UM if r.cells else 0.0) + span <= x_hi + 1e-6:
                home = r
                break
        if home is not None:
            x = home.x + (GUTTER_UM if home.cells else 0.0)
            put(c, x, home.y_top, home)
            home.xs.append(x); home.x1s.append(x + span)
            home.x = x + span
            continue
        # 2. a new row, if the height budget allows
        if span <= x_hi - x_lo + 1e-6 and rows_h() + (ROW_GUTTER_UM if rows else 0.0) + h <= y_hi - y_lo + 1e-6:
            y_top = y_hi - rows_h() - (ROW_GUTTER_UM if rows else 0.0)
            r = _Row(h=h, y_top=y_top, x=x_lo)
            rows.append(r)
            put(c, x_lo, y_top, r)
            r.xs.append(x_lo); r.x1s.append(x_lo + span)
            r.x = x_lo + span
            continue
        # 3. a column in an existing row's leftover
        done = False
        for r in sorted(rows, key=lambda r: -r.h):
            for col in r.columns:
                hit = col.try_place(c)
                if hit is not None:
                    put(c, *hit, r)
                    done = True
                    break
            if done:
                break
            x0 = r.x + (GUTTER_UM if r.cells else 0.0)
            if x0 + span <= x_hi + 1e-6 and h <= r.h + 1e-6:
                col = _Column(x0=x0, w=span, y_top=r.y_top, y_bot=r.y_bot)
                hit = col.try_place(c)
                assert hit is not None
                r.columns.append(col)
                put(c, *hit, r)
                r.xs.append(x0); r.x1s.append(x0 + span)
                r.x = x0 + span
                done = True
                break
        if not done:
            overflow.append(c.cid)

    # --- the cut list: EDGE cuts -----------------------------------------------
    # Every cut runs along a DIE EDGE with the blade wholly in the street, so
    # the die comes out at its drawn size whatever the kerf. A 1 mm street
    # between two dies therefore takes TWO cuts (one per edge, blade under
    # 0.5 mm); a centred single cut would leave (street - kerf)/2 of extra
    # glass on each die - 0.35 mm with a 0.3 mm blade, which the box's nesting
    # cannot absorb. Rows: the die top and the die bottom of each strip (the
    # label band below the dies is waste). Bench cells riding a taller row
    # under ROW_SLACK_UM simply come out a little tall.
    by_cid = {p.cell.cid: p for p in placed}
    strips = []
    y_cuts: list[float] = []
    for k, r in enumerate(rows):
        die_h = max(by_cid[c].cell.h_um for c in r.cells)
        y_top_die = r.y_top
        y_bot_die = r.y_top - die_h
        y_cuts += [y_top_die, y_bot_die]
        x_edges = sorted(set(r.xs) | set(r.x1s))
        strips.append({
            "row": k + 1,
            "y_top_mm": round(r.y_top / MM, 3), "y_bot_mm": round(r.y_bot / MM, 3),
            "height_mm": round(r.h / MM, 3),
            "die_top_mm": round(y_top_die / MM, 3), "die_bot_mm": round(y_bot_die / MM, 3),
            "die_height_mm": round(die_h / MM, 3),
            "cells": list(r.cells),
            # every die edge in the strip: cut here, blade OUTSIDE the die
            "x_cuts_mm": [round(x / MM, 3) for x in x_edges],
            "columns": [{"x0_mm": round(col.x0 / MM, 3), "x1_mm": round((col.x0 + col.w) / MM, 3),
                         "cells": list(col.cells),
                         "y_cuts_mm": sorted({round(v / MM, 3)
                                              for c in col.cells
                                              for v in (by_cid[c].cy + by_cid[c].cell.h_um / 2,
                                                        by_cid[c].cy - by_cid[c].cell.h_um / 2)},
                                             reverse=True)}
                        for col in r.columns],
        })
    used = rows_h()
    return placed, {
        "rows": [{"y_top_mm": s["y_top_mm"], "height_mm": s["height_mm"], "cells": s["cells"]}
                 for s in strips],
        "dicing": {
            "street_um": GUTTER_UM,
            "edge_cuts": True,
            # full-width horizontal cuts: the top and bottom edge of every strip's dies
            "y_cuts_mm": sorted({round(y / MM, 3) for y in y_cuts}, reverse=True),
            "strips": strips,
            "protocol": ["1. horizontal cuts along every die top/bottom edge, full plate width -> strips (label bands and streets fall away)",
                         "2. per strip, vertical cuts along every die edge across the strip -> dies / column pieces",
                         "3. per column piece, its own horizontal cuts along the cell edges -> cells",
                         "blade wholly in the street, touching the edge line: two cuts per 1 mm street, blade under 0.5 mm"],
        },
        "height_used_mm": round(used / MM, 2),
        "height_available_mm": round(USABLE_UM / MM, 2),
        "fits": not overflow,
        "overflow": overflow,
    }


def write_dicing_md(plate: dict[str, Any], out_path: Path) -> Path:
    """The saw operator's sheet: every cut in mm from the plate centre (x
    right, y up, looking at the CHROME side), strip by strip."""
    lg = plate["layout"]
    dc = lg["dicing"]
    by_cid = {m["cid"]: m for m in plate["manifest"]}
    L: list[str] = []
    L.append("# Dicing plan — 5\" production plate\n")
    L.append(f"Coordinates in mm from the plate centre, x right, y up, chrome side up. "
             f"Streets are {dc['street_um']/MM:.1f} mm wide. EVERY CUT IS AN EDGE CUT: the line given is a die "
             f"edge and the blade sits wholly in the street touching that line, so the die comes out at its drawn "
             f"size whatever the kerf; a street between two dies takes two cuts (blade under 0.5 mm). "
             f"A die's corner L-ticks sit in the street {0.15:.2f} mm off its edge. Every cut line has SAW-LANE "
             f"MARKS in the chrome: a horizontal cut is marked by a {DICE_MARK_LEN_UM/MM:.1f} mm clear bar at the "
             f"left and right edge of the field; a strip's vertical cuts are marked by bars in the street just "
             f"above its dies and in the label band just below them (the label bands and streets are waste).\n")
    L.append("## Protocol\n")
    for s in dc["protocol"]:
        L.append(f"- {s}")
    L.append("")
    L.append("## Step 1 — horizontal edge cuts (full width)\n")
    L.append("| cut | y (mm) | which edge |\n|---|---|---|")
    edge_of: dict[float, list[str]] = {}
    for st in dc["strips"]:
        edge_of.setdefault(st["die_top_mm"], []).append(f"strip {st['row']} top")
        edge_of.setdefault(st["die_bot_mm"], []).append(f"strip {st['row']} bottom")
    for i, y in enumerate(dc["y_cuts_mm"], 1):
        L.append(f"| H{i} | {y:+.3f} | {', '.join(edge_of.get(y, ['?']))} |")
    L.append("")
    L.append("## Step 2 — vertical streets, per strip\n")
    for s in dc["strips"]:
        L.append(f"### Strip {s['row']}: dies y {s['die_bot_mm']:+.3f} .. {s['die_top_mm']:+.3f} mm "
                 f"(die height {s['die_height_mm']:.3f} mm; the label band below, down to {s['y_bot_mm']:+.3f}, is waste)\n")
        L.append("| cell | x centre (mm) | die w x h (mm) | what |\n|---|---|---|---|")
        for cid in s["cells"]:
            m = by_cid.get(cid)
            if m is None:
                continue
            L.append(f"| {cid} | {m['x_mm']:+.3f} | {m['w_mm']:.2f} x {m['h_mm']:.2f} | {m['title']} |")
        L.append("")
        if s["x_cuts_mm"]:
            L.append("vertical edge cuts at x = " + ", ".join(f"{x:+.3f}" for x in s["x_cuts_mm"])
                     + " mm (each is a die edge; blade on the street side of it)\n")
        for col in s["columns"]:
            L.append(f"- column piece x {col['x0_mm']:+.3f} .. {col['x1_mm']:+.3f} mm holds "
                     f"{', '.join(col['cells'])}; step 3 cuts at y = "
                     + (", ".join(f"{y:+.3f}" for y in col["y_cuts_mm"]) or "none") + " mm")
        L.append("")
    L.append("## Production dies\n")
    L.append("| cell | face | cut size (mm) | rotated | notes |\n|---|---|---|---|---|")
    for m in plate["manifest"]:
        if m["block"] != "production":
            continue
        st = m.get("stats", {})
        L.append(f"| {m['cid']} | {st.get('face','')} | {m['w_mm']:.2f} x {m['h_mm']:.2f} | "
                 f"{'yes' if st.get('rotated') else 'no'} | {m['note']} |")
    L.append("")
    L.append("All production dies are written MIRRORED (x -> -x) for the chrome-down stack; a rotated die "
             "was mirrored first, then turned +90 deg. Each carries a tick-code ID in the foil-fold band "
             "(bars = face index + 1: front 1, back 2, left 3, right 4, top 5, bottom 6).")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


# --- build ------------------------------------------------------------------


def build_plate(
    bands: Iterable[list[Cell]] | None = None, *, verbose: bool = True,
    polarity: str = CLEAR,
) -> dict[str, Any]:
    """Run every cell builder and collect the plate's geometry and manifest.

    ``polarity`` defaults to CLEAR because that is what the plate is: a darkfield
    write with positive resist, where the data defines where the chrome comes
    OFF. Building in METAL is for inspection and for the area arithmetic that
    checks the two are complements.
    """
    bands = list(bands if bands is not None else doe_cells())
    placed, lay = layout(bands)
    if not lay["fits"]:
        raise ValueError(f"witness plate does not fit the usable field: {lay['overflow']} overflow; "
                         f"{lay['height_used_mm']} of {lay['height_available_mm']} mm used")
    front: list[np.ndarray] = []
    free: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    outline: list[np.ndarray] = []
    pair_marks: list[np.ndarray] = []
    arrays: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for p in placed:
        c = p.cell
        t0 = time.perf_counter()
        if c.takes_polarity:
            art = c.build(p.cx, p.cy, c.w_um, c.h_um, polarity=polarity)
        else:
            art = c.build(p.cx, p.cy, c.w_um, c.h_um)
            if polarity == CLEAR:
                # This cell does not know its own inverse, so take it with a
                # per-cell boolean. Its BACK die inverts separately, against the
                # same cell box, because the two dies are different plies.
                art.free_polys = clear_by_boolean(
                    p.cx, p.cy, c.w_um, c.h_um, art.front, art.polys)
                if len(art.back):
                    back_clear = clear_by_boolean(
                        p.cx, p.cy, c.w_um, c.h_um, art.back, [])
                else:
                    back_clear = []
                art.front = np.empty((0, 4))
                art.polys = []
                art.back = np.empty((0, 4))
                art.stats["clear_polys"] = len(art.free_polys)
                p.back_free = back_clear
        p.art = art
        front.append(art.front)
        for a in art.arrays:
            arrays.append(a)
        outline.append(_frame_rects(p.cx, p.cy, c.w_um, c.h_um))
        # Bottom-LEFT of the cell, left-aligned: a plate is scanned under a
        # microscope in a raster, and a label in a consistent corner is found
        # without hunting. Centred-above put it closer to the cell ABOVE it.
        lab_h = _label_h(c.h_um)
        labels.append(_text_rects(
            c.label or c.cid, p.cx - c.w_um / 2.0,
            p.cy - c.h_um / 2.0 - lab_h * 0.5, lab_h * 0.66, anchor="left"))
        free.extend(art.free_polys)
        free.extend(art.polys)
        n_back = 0
        bw, bh = c.back_dims
        if c.two_layer and p.pair_cx is not None:
            dx = p.pair_cx - p.cx
            b = art.back.copy()
            if len(b):
                b[:, 0] += dx
                b[:, 1] += dx
                front.append(b)
            n_back = len(b)
            for pv in list(p.back_free) + list(art.back_polys):
                q = np.asarray(pv, dtype=np.float64).copy()
                q[:, 0] += dx
                free.append(q)
            n_back += len(p.back_free) + len(art.back_polys)
            for a in art.back_arrays:
                rr = np.asarray(a["rects"], dtype=np.float64).copy()
                rr[:, 0] += dx
                rr[:, 1] += dx
                arrays.append({**a, "rects": rr,
                               "phase_um": np.asarray(a.get("phase_um", 0.0),
                                                      dtype=np.float64) + dx})
            pair_marks.append(_frame_rects(p.pair_cx, p.cy, bw, bh))
            outline.append(_frame_rects(p.pair_cx, p.cy, bw, bh))
            lab_b = _label_h(bh)
            labels.append(_text_rects(
                (c.label or c.cid).split(" ")[0] + " B", p.pair_cx - bw / 2.0,
                p.cy - bh / 2.0 - lab_b * 0.5, lab_b * 0.66, anchor="left"))
        n_arr = sum(int(np.size(a["period_um"])) for a in art.arrays + art.back_arrays)
        dt = time.perf_counter() - t0
        manifest.append({
            "cid": c.cid, "title": c.title, "group": c.group,
            "block": c.block, "label": c.label or c.cid,
            "axis": c.axis, "level": c.level, "note": c.note,
            "x_mm": round(p.cx / MM, 3), "y_mm": round(p.cy / MM, 3),
            "w_mm": round(c.w_um / MM, 3), "h_mm": round(c.h_um / MM, 3),
            "two_layer": c.two_layer,
            "back_w_mm": round(bw / MM, 3), "back_h_mm": round(bh / MM, 3),
            "n_rects": int(len(art.front)) + n_back,
            "n_polys": int(len(art.free_polys)) + int(len(art.polys)) + int(len(art.back_polys)),
            "n_array_bands": n_arr,
            "build_s": round(dt, 2),
            "stats": art.stats,
        })
        if verbose:
            print(f"  {c.cid:14s} {c.title[:30]:30s} "
                  f"{len(art.front)+n_back:>8,}r {len(art.free_polys)+len(art.polys):>7,}p "
                  f"{n_arr:>7,}a {dt:5.1f}s")

    return {
        "placed": placed,
        "layout": lay,
        "polarity": polarity,
        "free_polys": free,
        # the saw-lane marks are DATA (clear in the chrome), so they ride ``front``
        "front": _cat(*front, _dice_edge_marks(lay)),
        "outline": _cat(*outline),
        "labels": _cat(*labels),
        "pair_marks": _cat(*pair_marks),
        "dice_lines": _dice_line_rects(lay),
        "arrays": arrays,
        "manifest": manifest,
    }


DICE_MARK_LEN_UM = 500.0
DICE_MARK_W_UM = 80.0
"""Saw-lane marks WRITTEN IN THE CHROME (clear bars the operator sights the
blade on): for every full-width horizontal cut a bar at the left and right
edge of the field, and for every vertical cut of a strip a bar in the street
just above and just below that strip (after the horizontal cuts these are the
strip's own top and bottom edges). 0.5 mm long so a mark at the field's edge
stays inside the blank's 4 mm handling margin, 80 µm wide like the corner
L-ticks."""


def _dice_edge_marks(lay: dict[str, Any]) -> np.ndarray:
    """Clear saw-lane marks (chrome-layer DATA) at the ends of every cut."""
    half = USABLE_UM / 2.0
    edge0 = half + TICK_REACH_UM - DICE_MARK_LEN_UM   # 58.95 .. 59.45 mm: past the cells, inside the margin
    edge1 = half + TICK_REACH_UM
    w = DICE_MARK_W_UM / 2.0
    out = []
    dc = lay["dicing"]
    for y in dc["y_cuts_mm"]:
        yc = y * MM
        out.append((-edge1, -edge0, yc - w, yc + w))
        out.append((edge0, edge1, yc - w, yc + w))
    for s in dc["strips"]:
        yt = s["die_top_mm"] * MM             # the dies' top edge
        yb = s["die_bot_mm"] * MM             # the dies' bottom edge (label band below it)
        for x in s["x_cuts_mm"]:
            xc = x * MM
            # above the dies: the street above the strip (or the top margin)
            y1 = min(yt + GUTTER_UM, edge1)
            out.append((xc - w, xc + w, y1 - DICE_MARK_LEN_UM, y1))
            # below the dies: the label band under them (waste once cut)
            out.append((xc - w, xc + w, yb - DICE_MARK_LEN_UM, yb))
    return np.asarray(out, dtype=np.float64) if out else np.empty((0, 4))


def _dice_line_rects(lay: dict[str, Any], w_um: float = 20.0) -> np.ndarray:
    """The saw's street centrelines as thin rects on the annotation layer:
    horizontal cuts across the usable field, vertical cuts across their strip,
    column cuts across their column piece."""
    half = USABLE_UM / 2.0
    out = []
    dc = lay["dicing"]
    for y in dc["y_cuts_mm"]:
        out.append((-half, half, y * MM - w_um / 2, y * MM + w_um / 2))
    for s in dc["strips"]:
        yt, yb = s["die_top_mm"] * MM, s["die_bot_mm"] * MM
        for x in s["x_cuts_mm"]:
            out.append((x * MM - w_um / 2, x * MM + w_um / 2, yb, yt))
        for col in s["columns"]:
            for y in col["y_cuts_mm"]:
                out.append((col["x0_mm"] * MM, col["x1_mm"] * MM, y * MM - w_um / 2, y * MM + w_um / 2))
    return np.asarray(out, dtype=np.float64) if out else np.empty((0, 4))


def flat_rect_count(plate: dict[str, Any]) -> int:
    """Rectangles the plate would hold with every array expanded."""
    from .patterns.bitmap import screenrects as sr

    n = int(len(plate["front"]))
    for a in plate["arrays"]:
        n += sr.stripe_plan(a["rects"], a["period_um"],
                            a["line_um"] / a["period_um"],
                            phase_um=a.get("phase_um", 0.0))["total"]
    return n


# --- writers ----------------------------------------------------------------


DBU_UM = 0.001
"""The writer's database unit. Everything below goes in as INTEGER DBU."""


def _to_dbu(a: np.ndarray, dbu_um: float = DBU_UM) -> np.ndarray:
    """µm floats -> the exact integer DBU klayout itself would have produced.

    Handing klayout a ``DBox``/``DPolygon`` makes it do this conversion per
    coordinate, in C++, one call at a time. Doing it here in one numpy pass and
    inserting integer ``Box``/``Polygon`` is measurably cheaper (below) — but
    only if the rounding is IDENTICAL, or the file moves by a nanometre.
    klayout rounds half AWAY FROM ZERO (``0.0005 -> 1``, ``-0.0005 -> -1``,
    ``1.2345 -> 1235``); Python's ``round`` and ``np.rint`` round half to EVEN
    and would give 1234. Hence trunc(x + copysign(0.5, x)).

    It also MULTIPLIES by 1/dbu rather than dividing by dbu, and the two are
    not the same double: 34.1585/0.001 is 34158.499999999996 and rounds down
    where 34.1585*1000.0 is 34158.5 and rounds up. Dividing here mismatched
    klayout on 77 of 1000 random coordinates. Multiplying matches it on all of
    them (and on 300k random + exact-half values, and byte-for-byte on a
    written GDS).
    """
    v = np.asarray(a, dtype=np.float64) * (1.0 / dbu_um)
    return np.trunc(v + np.copysign(0.5, v)).astype(np.int64)


def _insert_rects(cell: Any, layer: int, rects: np.ndarray, kdb: Any) -> None:
    """Insert ``(N,4)`` [x0,x1,y0,y1] µm rects as integer boxes.

    ``insert_box`` (the typed entry point) over ``insert`` (which resolves the
    overload set per call) and ``Box`` over ``DBox``: measured on 100k rects,
    0.85 s -> 0.17 s.
    """
    if not len(rects):
        return
    shapes = cell.shapes(layer)
    ins = shapes.insert_box
    Box = kdb.Box
    for x0, x1, y0, y1 in _to_dbu(np.asarray(rects, dtype=np.float64)).tolist():
        ins(Box(x0, y0, x1, y1))


def _insert_polys(cell: Any, layer: int, polys, kdb: Any) -> int:
    """Insert µm vertex rings as integer polygons.

    Three costs came off the plate's 780k written clear pieces here: the
    per-coordinate double->int conversion (done in numpy above), the
    ``DPoint``/``Point`` object per vertex (klayout's list-of-pairs Polygon
    constructor takes the tuples directly), and the overload resolution on
    ``Shapes.insert`` (``insert_polygon`` is typed). Measured on 100k 6-vertex
    polygons: 2.29 s -> 0.63 s.
    """
    shapes = cell.shapes(layer)
    ins = shapes.insert_polygon
    Poly = kdb.Polygon
    n = 0
    for pv in polys:
        pts = _to_dbu(pv).tolist()
        if len(pts) < 3:
            continue
        ins(Poly(pts))
        n += 1
    return n


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
    ly.dbu = DBU_UM
    top = ly.create_cell("WITNESS_5IN")
    l_front = ly.layer(*LAYER_FRONT)
    l_out = ly.layer(*LAYER_OUTLINE)
    l_pair = ly.layer(*LAYER_PAIR)
    l_dice = ly.layer(*LAYER_DICE)

    _insert_rects(top, l_front, plate["front"], kdb)
    _insert_rects(top, l_dice, plate.get("dice_lines", np.empty((0, 4))), kdb)
    _insert_polys(top, l_front, plate.get("free_polys", ()), kdb)
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
                                phase_um=a.get("phase_um", 0.0),
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
            phase = np.broadcast_to(
                np.asarray(a.get("phase_um", 0.0), dtype=np.float64), (len(rects),))
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
                x_start = phase[i] + k0[i] * d
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
    """A one-page plate map from the manifest: block colour, etched label, and
    the bonded band shaded. Keyed on ``block`` — an earlier version keyed on the
    single-letter group and coloured one block of five."""
    s = PLATE_SIDE_UM
    sc = 900.0 / s
    FILL = {"production": "#d6b04a", "moire": "#e0457b", "diffraction": "#00b8a9",
            "parallax": "#7c4dff", "halftone": "#f0a202", "metrology": "#9fb0b6"}
    man = plate["manifest"]
    two = [m for m in man if m["two_layer"]]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 960" '
        f'width="900" height="960"><rect width="900" height="960" fill="#0d1113"/>',
        f'<rect x="0" y="0" width="900" height="900" fill="none" stroke="#3a4449" stroke-width="2"/>',
    ]
    two = [m for m in two if m["block"] != "production"]
    if two:
        y_top = max(m["y_mm"] * MM + m["h_mm"] * MM / 2 for m in two)
        y_bot = min(m["y_mm"] * MM - m["h_mm"] * MM / 2 for m in two)
        parts.append(f'<rect x="2" y="{(s/2 - y_top)*sc:.1f}" width="896" '
                     f'height="{(y_top - y_bot)*sc:.1f}" fill="#7c4dff" fill-opacity=".07"/>')
    for m in man:
        x = (m["x_mm"] * MM + s / 2) * sc
        y = (s / 2 - m["y_mm"] * MM) * sc
        w = m["w_mm"] * MM * sc
        h = m["h_mm"] * MM * sc
        fill = FILL.get(m["block"], "#888")
        parts.append(f'<rect x="{x-w/2:.1f}" y="{y-h/2:.1f}" width="{w:.1f}" height="{h:.1f}" '
                     f'fill="{fill}" fill-opacity=".22" stroke="{fill}" stroke-width="1"/>')
        if w > 40:
            fs = max(7, min(11, w / max(6, len(m["label"]) * 0.75)))
            parts.append(f'<text x="{x:.1f}" y="{y+3:.1f}" fill="#e6eef1" font-size="{fs:.0f}" '
                         f'font-family="monospace" text-anchor="middle">{m["label"]}</text>')
        if m["two_layer"]:
            bw = m.get("back_w_mm", m["w_mm"]) * MM * sc
            bh = m.get("back_h_mm", m["h_mm"]) * MM * sc
            parts.append(f'<rect x="{x+w/2+GUTTER_UM*sc:.1f}" y="{y-bh/2:.1f}" width="{bw:.1f}" '
                         f'height="{bh:.1f}" fill="{fill}" fill-opacity=".10" stroke="{fill}" '
                         f'stroke-dasharray="3 2" stroke-width="1"/>')
    lg = plate["layout"]
    dc = lg.get("dicing")
    if dc:
        for yc in dc["y_cuts_mm"]:
            yy = (s / 2 - yc * MM) * sc
            parts.append(f'<line x1="{(s/2 - USABLE_UM/2)*sc:.1f}" y1="{yy:.1f}" x2="{(s/2 + USABLE_UM/2)*sc:.1f}" '
                         f'y2="{yy:.1f}" stroke="#ff5252" stroke-width="1" stroke-dasharray="6 3"/>')
        for st in dc["strips"]:
            y0 = (s / 2 - st["die_top_mm"] * MM) * sc
            y1 = (s / 2 - st["die_bot_mm"] * MM) * sc
            for xc in st["x_cuts_mm"]:
                xx = (xc * MM + s / 2) * sc
                parts.append(f'<line x1="{xx:.1f}" y1="{y0:.1f}" x2="{xx:.1f}" y2="{y1:.1f}" '
                             f'stroke="#ff5252" stroke-width="1" stroke-dasharray="6 3"/>')
            for col in st["columns"]:
                xa = (col["x0_mm"] * MM + s / 2) * sc
                xb = (col["x1_mm"] * MM + s / 2) * sc
                for yc in col["y_cuts_mm"]:
                    yy = (s / 2 - yc * MM) * sc
                    parts.append(f'<line x1="{xa:.1f}" y1="{yy:.1f}" x2="{xb:.1f}" y2="{yy:.1f}" '
                                 f'stroke="#ff8a80" stroke-width="1" stroke-dasharray="3 3"/>')
    x = 10
    for blk, col in FILL.items():
        parts.append(f'<rect x="{x}" y="912" width="12" height="12" fill="{col}" fill-opacity=".6"/>'
                     f'<text x="{x+16}" y="922" fill="#9fb0b6" font-size="12" font-family="monospace">{blk}</text>')
        x += 16 + 8 * len(blk) + 22
    parts.append(f'<text x="10" y="948" fill="#9fb0b6" font-size="12" font-family="monospace">'
                 f'{len(man)} cells &#183; {lg["height_used_mm"]:.1f} of {lg["height_available_mm"]:.0f} mm used '
                 f'&#183; red dashes = saw cuts (full-width rows first, then across each strip) &#183; gold = production dies</text>')
    parts.append("</svg>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/witness/witness-5in",
                    help="output stem; the suffix comes from --formats")
    ap.add_argument("--formats", default="gds,oas",
                    help="comma list of gds,oas. Both by default: GDSII is the "
                         "primary deliverable, OASIS the 1 MB copy of it")
    ap.add_argument("--writer", choices=("klayout", "gf"), default="klayout",
                    help="klayout: flat-ish, 5 s, 40 MB GDS. gf: gdsfactory "
                         "hierarchy with @cell-cached band cells, 100 s, 31 MB GDS. "
                         "Measured; the floor is ~470k references either way")
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
    t_built = time.perf_counter()
    print(f"built in {t_built - t0:.1f}s")
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
        if a.writer == "gf":
            from .export_witness_gf import write_mask_gf
            paths = write_mask_gf(plate, Path(a.out), formats=fmts)
            plate["gds"] = {**plate.get("gds", {}), **plate["gds_gf"],
                            "path": str(paths[0]), "writer": "gdsfactory",
                            "size_mb": plate["gds_gf"]["size_mb_by_format"][
                                paths[0].suffix.lstrip(".")]}
        else:
            t_w = time.perf_counter()
            paths = write_mask(plate, Path(a.out), flat=a.flat, formats=fmts)
            plate["gds"]["writer"] = "klayout"
            plate["gds"]["write_s"] = round(time.perf_counter() - t_w, 1)
            print(f"written in {plate['gds']['write_s']}s")
        for p_ in paths:
            mb_ = plate["gds"]["size_mb_by_format"][p_.suffix.lstrip(".")]
            print(f"mask {p_}  {mb_} MB")
        print(f"     {plate['gds']['n_array_instances']:,} array refs over "
              f"{plate['gds']['n_unit_cells']} unit cells")
        svg = write_map_svg(plate, Path(a.map))
        print(f"map  {svg}")
        dmd = write_dicing_md(plate, Path(a.out).with_name("DICING.md"))
        print(f"dicing {dmd}")
        mp = Path(a.manifest)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({
            "plate_side_um": PLATE_SIDE_UM,
            "layout": lg,
            "gds": plate.get("gds", {}),
            "polarity": plate["polarity"],
            "n_boolean_polys": len(plate["free_polys"]),
            "flat_rect_count": n_flat,
            "cells": plate["manifest"],
        }, indent=2), encoding="utf-8")
        print(f"manifest {mp}")
    print(f"total {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
