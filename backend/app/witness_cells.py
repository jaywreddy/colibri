"""Geometry for every witness-plate cell, in rect space.

One builder per effect. They all return :class:`~app.witness_geom.CellArt` and
they all work analytically — a lamellar grating is arithmetic on line indices,
not a rasterized lattice — so a 2 um grating across a 12 mm patch costs 6000
rectangles and no raster at all. That is what lets a 127 mm plate be built on a
13.7 GB host.

The halftone builder is the exception that proves the rule: it has to look at a
photograph, so it samples one at the resolution the cell actually needs
(:func:`asset_px_for`) rather than at the plate's own cell pitch.

The file is in two parts. Above the OFF-PLATE banner are the builders the
current plate's DoE calls (``export_witness.doe_cells``) plus
:func:`build_halftone_bands`, which the BOX's photo faces also go through
(``plates.photo_band_rects``) so a picture is screened identically either way.
Below it is ``build_colour_band``, kept with the question it answers for the
next plate. The two-ply builders that used to sit there went with the two-ply
design on 2026-09-16 — see the banner.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .patterns.bitmap import colourplan as cp
from .patterns.bitmap import imageprep as ip
from .patterns.bitmap import screenrects as sr
from .patterns.bitmap.colourzone import MIN_FEATURE_UM
from .witness_geom import (
    METAL,
    PORTRAIT_CROP,
    REF_SCREEN_UM,
    REF_TONE_STEPS,
    SOURCE_PHOTO,
    CellArt,
    _cat,
    _grating_rects,
)

_SRC_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def photo_path() -> Path:
    """The source photograph: ``photos/<SOURCE_PHOTO>`` at the repo root, or the
    root itself where it lived before the photos folder existed."""
    root = Path(__file__).resolve().parents[2]
    for cand in (root / "photos" / SOURCE_PHOTO, root / SOURCE_PHOTO):
        if cand.is_file():
            return cand
    return root / "photos" / SOURCE_PHOTO


def portrait_source(size: int = 1400) -> tuple[np.ndarray, np.ndarray]:
    """``(gray, rgb)`` for the reference crop, both at ``size`` square.

    Loaded through ``imageprep.load_gray`` and ``colourplan.load_rgb`` with the
    SAME crop, which is what guarantees the darkness map and the period field
    are in register — a half-pixel disagreement would colour the wrong petal.
    """
    if size not in _SRC_CACHE:
        p = photo_path()
        if not p.is_file():
            raise FileNotFoundError(f"source photo not found: {p}")
        _SRC_CACHE[size] = (
            ip.load_gray(p, crop=PORTRAIT_CROP, size=size),
            cp.load_rgb(p, crop=PORTRAIT_CROP, size=size),
        )
    return _SRC_CACHE[size]


def asset_px_for(
    extent_um: float, line_period_um: float, oversample: float = 1.25
) -> int:
    """Source resolution to prep for a cell of this size.

    Rectangle count is set by the ASSET, not by the plate (see ``screenrects``),
    so this is the real cost dial — and 1.25 px per halftone line is already
    past what anyone can see.

    The eye's integration cell is 87 um, which at the reference 44 um screen is
    exactly TWO line periods. A 30 mm cell therefore resolves 345 elements
    across however finely it is written, while an oversample of 2.0 preps 1364
    source pixels and spends a rectangle on each — four times the detail the
    viewer can resolve, at four times the file. 1.25 keeps a margin over acuity
    for the sharpening to work with and roughly halves the plate's GDS.
    """
    n_lines = max(1.0, extent_um / line_period_um)
    return int(max(160, min(2200, round(n_lines * oversample))))


# --- the headline cell ------------------------------------------------------


def build_halftone_bands(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    coverage: np.ndarray,
    period_id: np.ndarray | None,
    periods: dict[int, float],
    duty: float = 0.5,
    line_period_um: float = REF_SCREEN_UM,
    tone_steps: int = REF_TONE_STEPS,
    defer_arrays: bool = True,
    polarity: str = METAL,
) -> tuple[CellArt, dict[str, Any]]:
    """The line screen itself: coverage + a period field -> band rectangles.

    Split out of :func:`build_halftone` so a caller with its OWN prepared image
    can reuse the band logic instead of copying it. The witness plate's portrait
    cells go through ``build_halftone`` (which loads the fixed ``SOURCE_PHOTO``
    and preps it here); the box's ``photo-halftone`` faces go through
    ``plates.photo_band_rects``, which prepares an arbitrary asset — with its own
    edge fade, which is why it hands in a finished COVERAGE map rather than a
    photograph. Both then land in this one function, so the plate and the box
    screen a picture identically.

    ``period_id`` is ``None`` for a plain screen (solid gold bands, no
    sub-grating). Returns ``(art, report)``; the report carries the band counts
    the callers fold into their own stats.
    """
    steps = max(2, min(int(tone_steps), int(line_period_um / MIN_FEATURE_UM)))
    bands, pid, band_report = sr.screen_bands(
        coverage,
        extent_um=min(w, h),
        line_period_um=line_period_um,
        tone_steps=steps,
        period_id=period_id,
        origin=(cx, cy),
        emit=polarity,
    )
    plain, coloured, per = sr.split_by_colour(bands, pid, periods)

    # In CLEAR polarity the sub-grating inverts too: the clear stripes are the
    # gaps of the metal ones, which is duty 1-c at phase c*d. The phase is
    # per-rectangle because d differs from rung to rung.
    sub_duty = duty if polarity == METAL else 1.0 - duty
    sub_phase = 0.0 if polarity == METAL else per * duty

    art = CellArt(front=plain)
    n_stripes = 0
    if len(coloured):
        sp = sr.stripe_plan(coloured, per, sub_duty, phase_um=sub_phase)
        n_stripes = sp["total"]
        if defer_arrays:
            art.arrays.append(
                {
                    "rects": coloured,
                    "period_um": per,
                    "line_um": per * sub_duty,
                    "phase_um": sub_phase,
                }
            )
        else:
            # Real rectangles: band ends held to the 2 um floor, and the CLEAR
            # side built as the exact complement of the METAL one (so the two
            # tile every band; see stripe_rects_floored).
            art.front = _cat(art.front, sr.stripe_rects_floored(
                coloured, per, duty, metal=(polarity == METAL)))

    report = {
        "tone_steps": steps,
        "finest_band_um": round(line_period_um / steps, 3),
        "n_band_rects": int(len(bands)),
        "n_plain_rects": int(len(plain)),
        "n_coloured_bands": int(len(coloured)),
        "n_stripes_if_flat": int(n_stripes),
        "rects_per_line": band_report["rects_per_line"],
    }
    return art, report


def build_halftone(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    plan: cp.ColourPlan,
    prep: ip.PrepSpec | None = None,
    line_period_um: float = REF_SCREEN_UM,
    tone_steps: int = REF_TONE_STEPS,
    defer_arrays: bool = True,
    polarity: str = METAL,
) -> CellArt:
    """One halftone portrait cell of the reference photo, plain or colour-shaded.

    OFF-PLATE since 2026-09-10: the two colour SIDE dies are the portraits now,
    at their 13.3 mm art box, and the plain control is H-WEDGE. Kept here
    rather than under the
    EXPERIMENTS banner because it is the reference caller of
    :func:`build_halftone_bands` — the metal/clear complement property that both
    the plate and the box's photo faces depend on is pinned through it
    (``tests/test_witness.py``), and it is what
    ``tools/dev/render_witness_preview.py`` renders its treatments from.

    The whole three-variant question reduces to which ``plan`` is passed: a
    ``plain`` plan yields an empty period field and no sub-grating at all, so
    the control cell runs the same code path rather than a different one. Any
    difference on the finished plate is the colour and nothing else.
    """
    steps = max(2, min(int(tone_steps), int(line_period_um / MIN_FEATURE_UM)))
    px = asset_px_for(min(w, h), line_period_um)
    gray, rgb = portrait_source(px)
    spec = prep or ip.PrepSpec(tone_steps=steps)
    if spec.tone_steps != steps:
        spec = ip.PrepSpec(**{**spec.__dict__, "tone_steps": steps})
    dark = ip.prep_darkness(gray, spec)

    ids, periods, field_report = cp.build_period_field(rgb, plan)
    art, report = build_halftone_bands(
        cx,
        cy,
        w,
        h,
        coverage=dark,
        period_id=None if plan.mode == "plain" else ids,
        periods=periods,
        duty=plan.duty,
        line_period_um=line_period_um,
        tone_steps=steps,
        defer_arrays=defer_arrays,
        polarity=polarity,
    )

    art.stats = {
        "polarity": polarity,
        "mode": plan.mode,
        "extent_um": round(min(w, h), 1),
        "asset_px": px,
        "line_period_um": line_period_um,
        "tone_steps": report["tone_steps"],
        "finest_band_um": report["finest_band_um"],
        "n_band_rects": report["n_band_rects"],
        "n_plain_rects": report["n_plain_rects"],
        "n_coloured_bands": report["n_coloured_bands"],
        "n_stripes_if_flat": report["n_stripes_if_flat"],
        "frac_coloured": field_report["frac_coloured"],
        "rungs_used": field_report["rungs_used"],
        "base_period_um": plan.base_period_um,
        "hue_equalize": field_report.get("hue_equalize", False),
        "all_used_rungs_printable": field_report["all_used_rungs_printable"],
        "rects_per_line": report["rects_per_line"],
        # Eye cells across the picture at 300 mm; under ~150 it is a thumbnail.
        "eye_cells_across": int(min(w, h) / 87.0),
    }
    return art


# --- single-layer test structures -------------------------------------------


def build_grating_patch(
    cx: float, cy: float, w: float, h: float, *,
    period_um: float, duty: float = 0.5, angle_deg: float = 0.0,
    polarity: str = METAL,
) -> CellArt:
    """A bare lamellar patch — the rungs of every period and duty ladder."""
    vertical = abs(angle_deg) < 1e-9
    emit_duty = duty if polarity == METAL else 1.0 - duty
    phase = 0.0 if polarity == METAL else period_um * duty
    if vertical or abs(angle_deg - 90.0) < 1e-9:
        r = _grating_rects(cx, cy, w, h, period_um, emit_duty,
                           phase_um=phase, vertical=vertical)
    else:
        # Off-axis patches are cut on the projected axis and clipped, so the
        # writer never has to handle a rotated polygon for a test structure.
        r = _grating_rects(cx, cy, w * 1.6, h, period_um, emit_duty,
                           phase_um=phase, vertical=True)
        r = r[(r[:, 1] > cx - w / 2) & (r[:, 0] < cx + w / 2)]
        r[:, 0] = np.clip(r[:, 0], cx - w / 2, cx + w / 2)
        r[:, 1] = np.clip(r[:, 1], cx - w / 2, cx + w / 2)
    line = period_um * duty
    return CellArt(
        front=r,
        stats={
            "polarity": polarity,
            "period_um": period_um,
            "duty": duty,
            "line_um": round(line, 3),
            "gap_um": round(period_um - line, 3),
            "clears_litho_floor": bool(
                line >= MIN_FEATURE_UM - 1e-9
                and period_um - line >= MIN_FEATURE_UM - 1e-9
            ),
            "n_rects": int(len(r)),
        },
    )


# --- OFF-PLATE builders -----------------------------------------------------
#
# No cell on the current plate calls ``build_colour_band`` or ``build_halftone``
# (above). They are kept — not deleted — because each answers a question a
# FUTURE plate will ask with the photograph taken out of it, and both stay
# pinned by ``tests/test_witness.py`` for the metal/clear complement property,
# which is the one thing that must not rot while they sit here.
#
# The TWO-PLY builders that used to live here went on 2026-09-16 with the rest
# of the two-ply optics: the vernier (C1), the barrier switch (P-SWAP), the
# shading moire, the moire magnifier (B-MAG), the scanimation (P-SCAN) and the
# chirp (D-CHIRP). The box is six single plies, so none of them can be measured
# on its stock at all; they are in git history at 22d1634.


def build_colour_band(
    cx: float, cy: float, w: float, h: float, *,
    tone: float = 0.5, line_period_um: float = REF_SCREEN_UM,
    tone_steps: int = REF_TONE_STEPS, period_um: float = 5.0, duty: float = 0.5,
    hold_tone: bool = True,
) -> CellArt:
    """D-BAND — a FLAT-tone halftone whose bands carry a sub-grating.

    The clean version of the colour question, with the photograph taken out of
    it: one tone, one period, so the only thing the cell can tell you is
    whether a gratinged band still holds its tone and still diffracts. Beside
    a portrait cell it separates "the colour works" from "the picture works".
    """
    steps = max(2, min(int(tone_steps), int(line_period_um / MIN_FEATURE_UM)))
    eff = min(tone / duty, 1.0) if hold_tone else tone
    band_h = (math.floor(eff * steps) / steps) * line_period_um
    n = max(1, int(h / line_period_um))
    cyc = cy + h / 2.0 - (np.arange(n) + 0.5) * line_period_um
    bands = np.empty((n, 4), dtype=np.float64)
    bands[:, 0], bands[:, 1] = cx - w / 2.0, cx + w / 2.0
    bands[:, 2], bands[:, 3] = cyc - band_h / 2.0, cyc + band_h / 2.0
    art = CellArt(front=np.empty((0, 4)))
    art.arrays.append(
        {
            "rects": bands,
            "period_um": np.full(n, period_um),
            "line_um": np.full(n, period_um * duty),
            "phase_um": 0.0,
        }
    )
    art.stats = {
        "tone": tone,
        "hold_tone": hold_tone,
        "band_um": round(band_h, 3),
        "period_um": period_um,
        "line_um": round(period_um * duty, 3),
        "periods_per_band": round(band_h / period_um, 2),
        "clears_litho_floor": bool(period_um * duty >= MIN_FEATURE_UM - 1e-9),
        "enough_periods_for_a_spectrum": bool(band_h >= 2.0 * period_um),
        "n_bands": n,
    }
    return art
