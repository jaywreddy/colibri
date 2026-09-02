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
cut the pair, stack them chrome-up, bond. Neither die is mirrored — the back
die is a translation of its design position — so a die must NOT be flipped. That is the only part of this plate that costs a
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
from . import witness_moire as wm
from .patterns.bitmap import colourplan as cp
from .patterns.bitmap import imageprep as ip
from .witness_geom import (
    CLEAR,
    METAL,
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
ROW_GUTTER_UM = 800.0
"""Between rows. Most of this plate is never diced, so it does not need a saw
street; the bonded pairs carry their own frame."""


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
# The previous revision spent 40% of the plate on photographic portraits, one
# per parameter. A portrait is a subjective read of many coupled variables at
# once; H-WEDGE and D-SWATCH give the same information objectively in a
# twentieth of the area. Portraits survive only where the question genuinely is
# subjective -- which of these three goes on the lid -- and they are now 5%.

RESOLUTION_LADDER_UM = (0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0)
C3_DUTY_LADDER = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
PERIOD_LADDER_UM = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0)
BEAT_LADDER_UM = (500.0, 1000.0, 1635.0, 3000.0, 6000.0)
BEAT_DUTY_LADDER = (0.25, 0.50, 0.75)
ROTATION_LADDER_DEG = (1.0, 2.0, 4.0, 8.0)
HARMONIC_DUTY_LADDER = (0.42, 0.50, 0.58)
SCREEN_ANGLE_LADDER_DEG = (0.0, 45.0, 90.0)
BASE_PERIOD_LADDER_UM = (4.0, 5.0, 6.5, 8.0)
SPREAD_LADDER = (1.20, 1.45, 1.90)
NEAR_FIELD_LADDER_UM = (20.0, 30.0, 44.0, 64.0, 100.0)
SWITCH_COMB_LADDER_UM = (100.0, 173.0, 250.0, 350.0)
SCAN_PHASE_LADDER = (2, 4, 6)
SCREEN_LADDER_UM = (20.0, 30.0, 44.0, 60.0)
STEPS_LADDER = (8, 12, 16, 22)
DUTY_LADDER = (0.35, 0.50, 0.65)
COARSEN_LADDER_PX = (3, 7, 14)
UNSHARP_LADDER = (0.0, 0.75, 1.5)
SCALE_LADDER_MM = (4.0, 8.0)
"""The 12 mm rung is PORT-Z itself, so the ladder does not repeat it."""

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
"""Headline cell. 138 eye-cells across at 300 mm -- enough to judge which colour
treatment you want, which is the only question a portrait answers better than an
instrument. 30 mm read better still and cost 2700 mm2; the plan's section 0 is
why that trade went the other way."""
SWEEP_MM = 10.0
LADDER_MM = 4.0
PAIR_MM = 8.0
SWATCH_MM = 6.0
WEDGE_W_MM = 30.0
WEDGE_H_MM = 4.0


def _cell(cid, title, block, w_mm, h_mm, build, *, label="", note="",
          axis="", level="", two_layer=False, takes_polarity=False) -> Cell:
    return Cell(cid=cid, title=title, group=block[0].upper(),
                w_um=w_mm * MM, h_um=h_mm * MM, build=build, note=note,
                label=label or cid, axis=axis, level=level,
                two_layer=two_layer, block=block, takes_polarity=takes_polarity)


def _plan(mode: str, **kw: Any) -> cp.ColourPlan:
    """A variant of the reference colour plan with one knob moved."""
    base = {"plain": cp.PAULA_PLAIN, "hue": cp.PAULA_HUE, "zones": cp.PAULA_ZONES}[mode]
    if not kw:
        return base
    d = {k: v for k, v in base.__dict__.items()}
    d.update(kw)
    return cp.ColourPlan(**d)


def _halftone(cid, title, side_mm, *, mode="zones", label="", axis="", level="",
              note="", plan_kw=None, prep=None,
              line_period_um=REF_SCREEN_UM, tone_steps=REF_TONE_STEPS) -> Cell:
    plan = _plan(mode, **(plan_kw or {}))

    def build(cx, cy, w, h, polarity=METAL):
        return wc.build_halftone(cx, cy, w, h, plan=plan, prep=prep,
                                 line_period_um=line_period_um,
                                 tone_steps=tone_steps, polarity=polarity)

    return _cell(cid, title, "halftone", side_mm, side_mm, build, label=label,
                 note=note, axis=axis, level=level, takes_polarity=True)


def doe_cells() -> list[list[Cell]]:
    """Every cell on the plate, grouped into blocks. Order IS the layout order."""
    B: list[list[Cell]] = []

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
    for px, py in ((5.0, 5.0), (5.0, 8.0)):
        d_cells.append(_cell(
            f"CROSS{px:g}x{py:g}", f"crossed {px:g}/{py:g} um", "diffraction",
            SWATCH_MM, SWATCH_MM,
            (lambda px=px, py=py: (lambda cx, cy, w, h, polarity=METAL:
                wm.build_crossed(cx, cy, w, h, period_x_um=px, period_y_um=py,
                                 polarity=polarity)))(),
            takes_polarity=True,
            label=f"CROSS {px:g}/{py:g}", axis="D-CROSS", level=f"{px:g}/{py:g} um",
            note="2-D orders; also the cheapest check of the union identity"))
    d_cells.append(_cell(
        "D-CHIRP", "chirp 22 -> 3 um", "diffraction", 24.0, 6.0,
        lambda cx, cy, w, h, polarity=METAL: wc.build_chirp(cx, cy, w, h,
                                                             polarity=polarity),
        label="D-CHIRP", note="graded fan; crosses the floor at 4.0 um",
        takes_polarity=True))
    for bp in BASE_PERIOD_LADDER_UM:
        for sp in SPREAD_LADDER:
            d_cells.append(_cell(
                f"SW{bp:g}/{sp:.2f}", f"swatch {bp:g} um x{sp:.2f}", "diffraction",
                SWATCH_MM, SWATCH_MM,
                (lambda bp=bp, sp=sp: (lambda cx, cy, w, h, polarity=METAL:
                    wm.build_swatch(cx, cy, w, h, base_period_um=bp, spread=sp,
                                    polarity=polarity)))(),
                label=f"SW {bp:g}/{sp:.2f}", axis="D-SWATCH ladder", takes_polarity=True,
                level=f"base {bp:g} um, spread {sp:.2f}",
                note="whole hue ladder side by side -- replaces 8 portraits"))
    # D-BAND: a BANDED grating, not a continuous one. Built at tone 0.25 so the
    # held-tone band is 22 um (4.4 periods of 5 um) and the unheld one 11 um
    # (2.2 periods): the pair measures what holding tone costs. At tone 0.5
    # with tone held the band is the whole 44 um period and the cell is just a
    # grating with a phase reset — which is what the H cells were until a
    # reviewer read the manifest's band_um.
    for i, (t, per, ht) in enumerate([
        ("band 22 um + 5.0 um, held", 5.0, True),
        ("band 22 um + 4.15 um, held", 4.15, True),
        ("band 22 um + 6.02 um, held", 6.02, True),
        ("band 11 um + 5.0 um, not held", 5.0, False),
    ]):
        d_cells.append(_cell(
            f"BAND{i+1}", t, "diffraction", SWATCH_MM, SWATCH_MM,
            (lambda per=per, ht=ht: (lambda cx, cy, w, h:
                wc.build_colour_band(cx, cy, w, h, period_um=per, tone=0.25,
                                     hold_tone=ht)))(),
            label=f"BAND {per:g}{'H' if ht else ''}", axis="D-BAND",
            level=f"{per:g} um", note="does a BANDED grating still diffract, and what does holding tone cost?"))
    B.append(d_cells)

    # === MOIRE, single layer -- the largest block ===========================
    mo: list[Cell] = []
    for b_um in BEAT_LADDER_UM:
        mm = _beat_w_mm(b_um)
        mo.append(_cell(
            f"BEAT{b_um:g}", f"beat {b_um:g} um", "moire", mm, BEAT_H_MM,
            (lambda b_um=b_um: (lambda cx, cy, w, h:
                wm.build_beat(cx, cy, w, h, beat_um=b_um)))(),
            label=f"BEAT {b_um:g}", axis="B-BEAT spacing", level=f"{b_um:g} um",
            note=f"sized for {mm*1000/b_um:.1f} fringes; p/delta amplifies pitch error"))
    for c in BEAT_DUTY_LADDER:
        mo.append(_cell(
            f"BCON{c:.2f}", f"beat duty {c:.2f}", "moire",
            _beat_w_mm(1635.0), BEAT_H_MM,
            (lambda c=c: (lambda cx, cy, w, h:
                wm.build_beat_contrast(cx, cy, w, h, duty=c)))(),
            label=f"BCON {c:.2f}", axis="B-CONT duty", level=f"{c:.2f}",
            note="transmission: low duty brighter at a fifth of the contrast; reflection ranks them the other way"))
    for a in ROTATION_LADDER_DEG:
        mo.append(_cell(
            f"ROT{a:g}", f"rotation {a:g} deg", "moire",
            _beat_w_mm(63.5 / (2 * math.sin(math.radians(a) / 2)), min_mm=8.0),
            BEAT_H_MM,
            (lambda a=a: (lambda cx, cy, w, h:
                wm.build_rotation_beat(cx, cy, w, h, angle_deg=a)))(),
            label=f"ROT {a:g}deg", axis="B-ROT angle", level=f"{a:g} deg",
            note="fringes run ALONG the lines, not across"))
    for pa, pb, ang in [(63.5, 63.5, 1.0), (63.5, 66.07, 1.0), (63.5, 70.0, 1.0),
                        (63.5, 63.5, 3.0), (63.5, 66.07, 3.0), (63.5, 70.0, 3.0),
                        (63.5, 63.5, 6.0), (63.5, 66.07, 6.0), (63.5, 70.0, 6.0)]:
        mo.append(_cell(
            f"VEC{pb:g}/{ang:g}", f"vector {pb:g} um @ {ang:g} deg", "moire",
            LADDER_MM + 1.0, LADDER_MM + 1.0,
            (lambda pa=pa, pb=pb, ang=ang: (lambda cx, cy, w, h:
                wm.build_vector_beat(cx, cy, w, h, period_a_um=pa,
                                     period_b_um=pb, angle_deg=ang)))(),
            label=f"VEC {pb:g}/{ang:g}", axis="B-VEC pitch x angle",
            level=f"{pb:g} um @ {ang:g} deg",
            note="the general |k1-k2| formula the frame relies on"))
    for c in HARMONIC_DUTY_LADDER:
        mo.append(_cell(
            f"HARM{c:.2f}", f"harmonic, duty {c:.2f}", "moire", 10.0, BEAT_H_MM,
            (lambda c=c: (lambda cx, cy, w, h:
                wm.build_harmonic(cx, cy, w, h, duty=c)))(),
            label=f"HARM {c:.2f}", axis="B-HARM duty", level=f"{c:.2f}",
            note=("nominal: only the 1.65' beat" if abs(c - 0.5) < 1e-9
                  else "biased: the 6.4' (2,3) beat appears")))
    for ang in SCREEN_ANGLE_LADDER_DEG:
        mo.append(_cell(
            f"SCR{ang:g}", f"screen at {ang:g} deg", "moire", SWEEP_MM, BEAT_H_MM,
            (lambda ang=ang: (lambda cx, cy, w, h:
                wm.build_screen_over_carrier(cx, cy, w, h, screen_angle_deg=ang)))(),
            label=f"SCR {ang:g}deg", axis="B-SCREEN angle", level=f"{ang:g} deg",
            note="does perpendicular really kill the screen/carrier beat?"))
    B.append(mo)

    # === HALFTONE ===========================================================
    hf: list[Cell] = []
    for sp in (20.0, 44.0, 60.0):
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
    hf += [
        _halftone("PORT-P", "portrait / plain gold", PORTRAIT_MM, mode="plain",
                  label="PORT PLAIN", axis="H-PORT colour mode", level="plain",
                  note="the control: same code path, empty period field"),
        _halftone("PORT-H", "portrait / hue-mapped", PORTRAIT_MM, mode="hue",
                  label="PORT HUE", axis="H-PORT colour mode", level="hue",
                  note="period from every pixel's own hue"),
        _halftone("PORT-Z", "portrait / zone-mapped", PORTRAIT_MM, mode="zones",
                  label="PORT ZONES", axis="H-PORT colour mode", level="zones",
                  note="flowers by hue, sweater and glasses authored"),
    ]
    for mm in SCALE_LADDER_MM:
        hf.append(_halftone(
            f"SZ{mm:g}", f"scale {mm:g} mm", mm, mode="zones",
            label=f"SZ {mm:g}mm", axis="H-SCALE size", level=f"{mm:g} mm",
            note=f"{int(mm*1000/87)} eye-cells across at 300 mm"))
    B.append(hf)

    # === TWO-LAYER -- everything the bond is actually for ===================
    two: list[Cell] = [
        _cell("M-VERN", "registration vernier", "metrology", 6.0, 6.0,
              lambda cx, cy, w, h, polarity=METAL: wc.build_vernier(
                  cx, cy, w, h, polarity=polarity),
              label="M-VERN", two_layer=True, takes_polarity=True,
              note="80/88 um, beat 880 um, gain p/delta = 10x. Read before anything else"),
        _cell("P-RULE", "parallax ruler", "parallax", SWEEP_MM, 6.0,
              lambda cx, cy, w, h, polarity=METAL: wm.build_parallax_ruler(
                  cx, cy, w, h, polarity=polarity),
              label="P-RULE", two_layer=True, takes_polarity=True,
              note="reads t/n directly: 2.19 deg per tooth at a 2.29 mm quartz pair"),
        _cell("B-MOVE", "beat, across the gap", "moire", _beat_w_mm(1635.0),
              BEAT_H_MM,
              lambda cx, cy, w, h, polarity=METAL: wc.build_shading_moire(
                  cx, cy, w, h, polarity=polarity),
              label="B-MOVE", two_layer=True, takes_polarity=True,
              note="same geometry as BEAT1635 -- the ONLY difference is motion"),
    ]
    for p_um in NEAR_FIELD_LADDER_UM:
        two.append(_cell(
            f"NF{p_um:g}", f"near field {p_um:g} um", "moire", PAIR_MM, PAIR_MM,
            (lambda p_um=p_um: (lambda cx, cy, w, h, polarity=METAL:
                wm.build_near_field(cx, cy, w, h, period_um=p_um,
                                    polarity=polarity)))(),
            label=f"NF {p_um:g}um", axis="E-NF pitch", level=f"{p_um:g} um",
            two_layer=True, takes_polarity=True,
            note="Fresnel N = 1/2 null at 42 um: 30-44 degraded, >= 64 intact"))
    for comb in SWITCH_COMB_LADDER_UM:
        two.append(_cell(
            f"SWAP{comb:g}", f"switch, comb {comb:g} um", "parallax",
            PAIR_MM, PAIR_MM,
            (lambda comb=comb: (lambda cx, cy, w, h, polarity=METAL:
                wc.build_barrier_switch(cx, cy, w, h, comb_um=comb,
                                        polarity=polarity)))(),
            label=f"SWAP {comb:g}um", axis="P-SWAP comb", level=f"{comb:g} um",
            two_layer=True, takes_polarity=True,
            note="straddle-registered: 50/50 blend head-on, clean A/B at +-p/4. Witness angles"))
    for n_ph in SCAN_PHASE_LADDER:
        two.append(_cell(
            f"SCAN{n_ph}", f"scanimation, {n_ph} phase", "parallax",
            PAIR_MM, PAIR_MM,
            (lambda n_ph=n_ph: (lambda cx, cy, w, h, polarity=METAL:
                wc.build_scanimation(cx, cy, w, h, phases=n_ph,
                                     polarity=polarity)))(),
            label=f"SCAN {n_ph}", axis="P-SCAN phases", level=str(n_ph),
            two_layer=True, takes_polarity=True, note=f"wants {n_ph}x the registration of a 2-phase switch"))
    for mag, ps in (("-10", 66.0), ("-31.6", 61.9)):
        two.append(_cell(
            f"MAG{mag}", f"magnifier M={mag}", "moire", SWEEP_MM, SWEEP_MM,
            (lambda ps=ps: (lambda cx, cy, w, h, polarity=METAL:
                wc.build_moire_magnifier(cx, cy, w, h, sampler_um=60.0,
                                         motif_um=ps, polarity=polarity)))(),
            label=f"MAG {abs(float(mag)):g}x", axis="B-MAG magnification", level=f"M={mag}",
            two_layer=True, takes_polarity=True,
            note="M = p_s/(p_s - p_m), negative: the ghost is inverted. Sampling needs two planes"))
    B.append(two)
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
        # Label band scales with the cell. A fixed 0.9 mm band plus a 1.5 mm
        # gutter is 2.4 mm of overhead, which on a 4 mm cell is 60%; that, and
        # not the cell sizes, is what was costing the plate its height.
        row_h = max(c.h_um for c in row) + _label_h(max(c.h_um for c in row))
        x = x_lo
        for c in row:
            span = c.w_um * (2.0 if c.two_layer else 1.0) + (
                GUTTER_UM if c.two_layer else 0.0
            )
            lab_h = _label_h(max(cc.h_um for cc in row))
            pl = Placed(cell=c, cx=x + c.w_um / 2.0,
                        cy=y - lab_h - c.h_um / 2.0)
            if c.two_layer:
                pl.pair_cx = pl.cx + c.w_um + GUTTER_UM
            placed.append(pl)
            x += span + GUTTER_UM
        rows.append({
            "y_top_mm": round(y / MM, 2),
            "height_mm": round(row_h / MM, 2),
            "cells": [c.cid for c in row],
        })
        return y - row_h - ROW_GUTTER_UM

    # Packed CONTINUOUSLY across bands, not flushed at every band boundary.
    # Flushing per band cost 54 mm of the 119 available to half-empty rows —
    # the D row used 73 mm of width and the next ladder started below it anyway.
    # Order is still the band order, so a ladder stays contiguous and reads
    # left to right; it may simply wrap mid-ladder, which the map makes clear.
    row: list[Cell] = []
    row_w = 0.0
    for band in bands:
        # Sort each block by DESCENDING height before packing. A shelf packer
        # pays the tallest cell's height for every cell in the row, so
        # interleaving 4 mm ladder rungs with 20 mm beat cells wasted more than
        # half the plate (45% efficiency, 136 mm of content for 74 mm of cells).
        # The sort is stable, so a ladder — whose rungs are all one height —
        # still reads left to right in order.
        for c in sorted(band, key=lambda c: -c.h_um):
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
        if c.two_layer and p.pair_cx is not None:
            dx = p.pair_cx - p.cx
            b = art.back.copy()
            if len(b):
                b[:, 0] += dx
                b[:, 1] += dx
                front.append(b)
            n_back = len(b)
            for pv in p.back_free:
                q = pv.copy()
                q[:, 0] += dx
                free.append(q)
            n_back += len(p.back_free)
            for a in art.back_arrays:
                rr = np.asarray(a["rects"], dtype=np.float64).copy()
                rr[:, 0] += dx
                rr[:, 1] += dx
                arrays.append({**a, "rects": rr,
                               "phase_um": np.asarray(a.get("phase_um", 0.0),
                                                      dtype=np.float64) + dx})
            pair_marks.append(_frame_rects(p.pair_cx, p.cy, c.w_um, c.h_um))
            outline.append(_frame_rects(p.pair_cx, p.cy, c.w_um, c.h_um))
            labels.append(_text_rects(
                (c.label or c.cid) + " B", p.pair_cx, p.cy + c.h_um / 2.0 + LABEL_H_UM * 0.5,
                LABEL_H_UM * 0.62))
        n_arr = sum(int(np.size(a["period_um"])) for a in art.arrays + art.back_arrays)
        dt = time.perf_counter() - t0
        manifest.append({
            "cid": c.cid, "title": c.title, "group": c.group,
            "block": c.block, "label": c.label or c.cid,
            "axis": c.axis, "level": c.level, "note": c.note,
            "x_mm": round(p.cx / MM, 3), "y_mm": round(p.cy / MM, 3),
            "w_mm": round(c.w_um / MM, 3), "h_mm": round(c.h_um / MM, 3),
            "two_layer": c.two_layer,
            "n_rects": int(len(art.front)) + n_back,
            "n_polys": int(len(art.free_polys)) + int(len(art.polys)),
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
                            a["line_um"] / a["period_um"],
                            phase_um=a.get("phase_um", 0.0))["total"]
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
    for pv in plate.get("free_polys", ()):
        top.shapes(l_front).insert(
            kdb.DPolygon([kdb.DPoint(float(x), float(y)) for x, y in pv]))
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
    FILL = {"moire": "#e0457b", "diffraction": "#00b8a9", "parallax": "#7c4dff",
            "halftone": "#f0a202", "metrology": "#9fb0b6"}
    man = plate["manifest"]
    two = [m for m in man if m["two_layer"]]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 960" '
        f'width="900" height="960"><rect width="900" height="960" fill="#0d1113"/>',
        f'<rect x="0" y="0" width="900" height="900" fill="none" stroke="#3a4449" stroke-width="2"/>',
    ]
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
            parts.append(f'<rect x="{x-w/2+w+GUTTER_UM*sc:.1f}" y="{y-h/2:.1f}" width="{w:.1f}" '
                         f'height="{h:.1f}" fill="{fill}" fill-opacity=".10" stroke="{fill}" '
                         f'stroke-dasharray="3 2" stroke-width="1"/>')
    lg = plate["layout"]
    x = 10
    for blk, col in FILL.items():
        parts.append(f'<rect x="{x}" y="912" width="12" height="12" fill="{col}" fill-opacity=".6"/>'
                     f'<text x="{x+16}" y="922" fill="#9fb0b6" font-size="12" font-family="monospace">{blk}</text>')
        x += 16 + 8 * len(blk) + 22
    parts.append(f'<text x="10" y="948" fill="#9fb0b6" font-size="12" font-family="monospace">'
                 f'{len(man)} cells &#183; {lg["height_used_mm"]:.1f} of {lg["height_available_mm"]:.0f} mm used '
                 f'&#183; dashed = back die &#183; shaded band = everything that needs a bond</text>')
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
            paths = write_mask(plate, Path(a.out), flat=a.flat, formats=fmts)
            plate["gds"]["writer"] = "klayout"
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
