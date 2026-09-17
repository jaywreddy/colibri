"""Witness-plate primitives: constants, the cell contract, and rect utilities.

Split from ``export_witness`` (which lays the plate out and writes it) so the
geometry builders in ``witness_cells`` can import these without a cycle. The
import order is witness_geom -> witness_cells -> export_witness.

Geometry is built in RECT SPACE throughout - ``(N, 4)`` ``[x0, x1, y0, y1]``
arrays of micrometres, plate-centred, +y up - never as a full-plate raster. A
127 mm plate at the 2 um halftone cell would be 3.6 x 10^9 lattice cells; see
``patterns.bitmap.screenrects`` for why that number never gets built.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

# --- plate constants --------------------------------------------------------

PLATE_SIDE_UM = 127_000.0            # 5 inch square
EDGE_MARGIN_UM = 4_000.0             # handling / chuck exclusion
USABLE_UM = PLATE_SIDE_UM - 2.0 * EDGE_MARGIN_UM
GUTTER_UM = 1_000.0                  # = the blank's hand-scribe street
LABEL_H_UM = 900.0                   # gold cell-ID text under each cell

LAYER_FRONT = (10, 0)
LAYER_OUTLINE = (1, 0)
LAYER_LABEL = (3, 0)                 # annotation text, not gold

# The reference point every ladder brackets. These are the numbers the box
# currently ships or the analysis settled on, so a sweep reads as "the
# reference, plus or minus" rather than as an unanchored grid.
# --- the glass -------------------------------------------------------------
# This plate IS the box stock: the production dies are box plies, so the plate's
# thickness and index set every gap-scaled family of the design. Change these
# two numbers and the carrier, comb, monogram pitch, parallax rate, near-field
# limit and swap angles all follow; ``test_witness`` pins them against what
# ``plates._carrier_recipe_data`` computes for the same glass.
PLY_UM = 2250.0
GLASS_N = 1.4585
GLASS_MATERIAL = "fused quartz"
_BASELINE_GAP_UM = 500.0 / 1.46     # plates.BASE_PARALLAX_GAP_UM: the 500 um quartz design point
LAMBDA_UM = 0.55


def gap_scale() -> float:
    """How much slower this glass walks parallax than the 500 um baseline."""
    return (PLY_UM / GLASS_N) / _BASELINE_GAP_UM


def _snap_half(v: float) -> float:
    return round(v * 2.0) / 2.0


def beat_delta(p: float, beat_um: float) -> float:
    """Pitch increment that beats against ``p`` at ``beat_um``: p(p+d)/d = beat.

    Solved FROM the beat, never the other way round: the beat is the thing the
    eye reads and the thing a design picks, and pinning delta instead is what
    made an earlier witness cell read 0.075 where it should have read 0.47.
    ``patterns.effects.gratings.beat_delta_um`` is the same solve on the plate
    side; ``tests/test_optics_math.py`` pins the two against the identity.
    """
    if beat_um <= p:
        raise ValueError(f"beat {beat_um} must exceed the pitch {p}")
    return p * p / (beat_um - p)


def combined_beat_um(p1: float, p2: float, angle_deg: float) -> float:
    """The general VECTOR beat of two gratings: ``|k1 - k2|``.

    ``p_beat = p1 p2 / sqrt(p1^2 + p2^2 - 2 p1 p2 cos alpha)``. Reduces to the
    pitch beat ``p1 p2 / |p1 - p2|`` at alpha = 0 and to ``p / (2 sin(alpha/2))``
    at p1 = p2 — the two special cases the garland's angle fan and the lid's
    shading moire are each designed on, which is why the general form is kept in
    one place rather than written out twice.
    """
    a = math.radians(angle_deg)
    den = math.sqrt(p1 * p1 + p2 * p2 - 2.0 * p1 * p2 * math.cos(a))
    return float("inf") if den < 1e-12 else p1 * p2 / den


def harmonic_beats(p1: float, p2: float, duty: float,
                   max_order: int = 3) -> list[dict[str, Any]]:
    """Every ``|m/p1 - n/p2|`` beat of two square gratings, and its amplitude.

    The amplitude of harmonic ``k`` of a duty-``c`` square wave is ``c sinc(kc)``,
    which VANISHES for even ``k`` at exactly c = 0.5. So a beat that needs an
    even harmonic has amplitude identically zero at nominal duty and appears the
    moment the process drifts: the halftone screen (44 um) over the box carrier
    has a (2,3) beat at 6.4 arcmin — plainly visible banding across a
    photograph — that the nominal design does not have. That is the argument for
    the duty ladder on the plate, and it is pinned in
    ``tests/test_optics_math.py``.
    """
    def amp(k: int, c: float) -> float:
        x = math.pi * k * c
        return abs(c * math.sin(x) / x) if x else 0.0

    out = []
    for m in range(1, max_order + 1):
        for n in range(1, max_order + 1):
            f = abs(m / p1 - n / p2)
            if f < 1e-12:
                continue
            beat = 1.0 / f
            if beat > 60000.0:
                continue
            out.append({
                "m": m, "n": n,
                "beat_um": round(beat, 1),
                "arcmin": round(beat / 87.0, 2),
                "amplitude": round(amp(m, duty) * amp(n, duty), 5),
            })
    return sorted(out, key=lambda r: -r["beat_um"])


# The carrier is sized by the EYE, not by the gap. Scaling the 22 um design
# pitch with the gap (x4.5 on this glass) gave 99 um: 49.5 um lines that
# subtend 1.13 arcmin at 300 mm, which the eye resolves as a hatch — the
# garland read as chunks, not as shimmer. The rule is that the carrier PERIOD
# subtends CARRIER_ARCMIN at the viewing distance, so the lines themselves are
# invisible in hand at any distance and only the beat is seen. The near field
# then sets the cost: at 65.5 um (Fresnel N 1.26) the fringes keep 51% of
# their zero-gap contrast across the ply (tools/dev/nearfield_ladder.py: 44 um
# 7%, 55 um 34%, 66 um 52%, 77 um 65%, 99 um 75%).
VIEW_DISTANCE_UM = 300_000.0
CARRIER_ARCMIN = 0.75


def _eye_pitch(arcmin: float, view_um: float = VIEW_DISTANCE_UM) -> float:
    import math
    return view_um * math.tan(math.radians(arcmin / 60.0))


BOX_CARRIER_UM = max(4.0, _snap_half(_eye_pitch(CARRIER_ARCMIN)))
"""The back-layer carrier every lid moire beats against: 0.75 arcmin at 300 mm."""
BOX_FRONT_LEAF_UM = BOX_CARRIER_UM * 1.09
"""The garland's front grating (plates.FRONT_GRATING_RATIO)."""
BOX_COMB_UM = _snap_half(60.0 * gap_scale())
"""The parallax-barrier comb (plates.fab_center_period_um)."""
BOX_BEAT_UM = 1635.0
BOX_MONO_UM = BOX_CARRIER_UM + beat_delta(BOX_CARRIER_UM, BOX_BEAT_UM)
"""The monogram's front carrier: beats the back carrier at BOX_BEAT_UM."""


def parallax_um_per_deg(t_um: float = PLY_UM, n: float = GLASS_N) -> float:
    import math
    return t_um * math.tan(math.asin(math.sin(math.radians(1.0)) / n))


PARALLAX_UM_PER_DEG = parallax_um_per_deg()


def fresnel_number(p_um: float) -> float:
    """N = p^2 n / (4 lambda z) across one ply. >= 1 intact, < 1/4 gone."""
    return p_um * p_um * GLASS_N / (4.0 * LAMBDA_UM * PLY_UM)


P_MIN_UM = (2.0 * LAMBDA_UM * PLY_UM / GLASS_N) ** 0.5
"""The pitch at the N = 1/2 null: sqrt(2 lambda z / n)."""


def swap_deg(comb_um: float) -> float:
    """Exterior tilt at which a straddle-registered barrier of pitch ``comb_um``
    has walked p/4 across the ply."""
    import math
    return math.degrees(math.asin(GLASS_N * math.sin(math.atan((comb_um / 4.0) / PLY_UM))))


REF_SCREEN_UM = 44.0
REF_TONE_STEPS = 22
REF_BASE_PERIOD_UM = 5.0
REF_DUTY = 0.5
REF_SPREAD = 1.45
REF_COARSEN_PX = 7

SOURCE_PHOTO = "PXL_20250920_201250581.jpg"
PORTRAIT_CROP = (0.040, 0.165, 0.825)
"""Crop of the source photo, as source-WIDTH fractions.

Chosen to hold all three colour subjects at once: the flower carpet fills the
left half, the sage sweater the right, and the teal spectacle frames sit high
enough in the frame to stay above the sweater's bbox. A tighter portrait crop
loses the carpet, which is the only region with enough distinct hues to show
what a period ladder actually does.
"""


# --- cells ------------------------------------------------------------------


@dataclass
class CellArt:
    """What one cell contributes, in plate-centred micrometres."""

    front: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    polys: list[np.ndarray] = field(default_factory=list)
    """METAL geometry that is not axis-aligned: a list of ``(k, 2)`` vertex
    arrays. A production die's rotated placement produces these; clipping them
    to the cell makes the vertex count variable, so this cannot be one array."""
    free_polys: list[np.ndarray] = field(default_factory=list)
    """Variable-vertex polygons, each ``(k, 2)``. Only the boolean clear-field
    inverter produces these: a box minus a grating is not a rectangle."""
    arrays: list[dict[str, Any]] = field(default_factory=list)
    """Periodic sub-gratings, deferred as array references rather than
    polygons. Each entry is one band: ``x0/x1/y0/y1`` of the parent band,
    ``period_um``, ``line_um``. See :func:`_emit_arrays`."""
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class Cell:
    cid: str
    title: str
    group: str
    w_um: float
    h_um: float
    build: Callable[[float, float, float, float], CellArt]
    """``build(cx, cy, w, h) -> CellArt``."""
    note: str = ""
    label: str = ""
    """What is etched in gold beside the cell. Defaults to ``cid``.

    A witness plate is read under a microscope at 50x, where every sweep cell
    looks like every other sweep cell. An id like "K3" is unreadable there
    without the map, and maps get separated from plates — so the label carries
    the VALUE, not just the index: "CP 6.5um" tells you what you are looking at
    and a stack of them tells you which way the ladder runs."""
    axis: str = ""
    """Which DoE axis this cell is a rung of, empty for a one-off."""
    block: str = ""
    """Which physics block: metrology, diffraction, moire, parallax, halftone."""
    takes_polarity: bool = False
    """True if ``build`` accepts ``polarity=`` and knows its own clear-field
    inverse analytically. False means the plate inverts it with a per-cell
    boolean instead — exact and trivial while the cell is small, which every
    cell that sets this False is."""
    level: str = ""
    """This cell's value on that axis."""


# --- small rect utilities ---------------------------------------------------


def _rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, x1, y0, y1]], dtype=np.float64)


def _cat(*parts: np.ndarray) -> np.ndarray:
    """Concatenate rect arrays. Never a GEOS union — see CLAUDE.md geometry perf."""
    live = [p for p in parts if p is not None and len(p)]
    return np.concatenate(live, axis=0) if live else np.empty((0, 4), dtype=np.float64)


def _grating_rects(
    cx: float, cy: float, w: float, h: float, period_um: float, duty: float = 0.5,
    *, phase_um: float = 0.0, vertical: bool = True,
) -> np.ndarray:
    """A lamellar grating filling a box, as one rect per line.

    Analytic, so a 2 um grating over a 12 mm patch costs 6000 rectangles and no
    raster at all.
    """
    if period_um <= 0:
        raise ValueError("period_um must be > 0")
    span = w if vertical else h
    a0 = (cx - w / 2.0) if vertical else (cy - h / 2.0)
    n = int(math.ceil(span / period_um)) + 1
    k = np.arange(n, dtype=np.float64)
    s0 = a0 + phase_um + k * period_um
    s1 = s0 + period_um * duty
    s0 = np.clip(s0, a0, a0 + span)
    s1 = np.clip(s1, a0, a0 + span)
    keep = s1 - s0 > 1e-9
    s0, s1 = s0[keep], s1[keep]
    out = np.empty((s0.size, 4), dtype=np.float64)
    if vertical:
        out[:, 0], out[:, 1] = s0, s1
        out[:, 2], out[:, 3] = cy - h / 2.0, cy + h / 2.0
    else:
        out[:, 0], out[:, 1] = cx - w / 2.0, cx + w / 2.0
        out[:, 2], out[:, 3] = s0, s1
    return out


def _text_rects(
    text: str, cx: float, cy: float, height_um: float, *,
    cell_um: float | None = None, anchor: str = "center",
) -> np.ndarray:
    """Gold cell-ID text, rasterized coarsely and run-merged into rectangles.

    A witness plate is read under a microscope at 50x, where every cell looks
    like every other cell; without a written ID beside each one the map is the
    only way to know what you are looking at, and maps get separated from
    plates. Deliberately coarse — this is signage, not a feature.
    """
    from PIL import Image, ImageDraw, ImageFont

    cell = cell_um if cell_um is not None else max(4.0, height_um / 14.0)
    px_h = max(7, int(round(height_um / cell)))
    try:
        font = ImageFont.truetype("arialbd.ttf", px_h)
    except Exception:
        font = ImageFont.load_default()
    tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
    box = tmp.textbbox((0, 0), text, font=font)
    w_px, h_px = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    img = Image.new("L", (w_px + 4, h_px + 4), 0)
    ImageDraw.Draw(img).text((2 - box[0], 2 - box[1]), text, fill=255, font=font)
    g = np.asarray(img) > 127
    if not g.any():
        return np.empty((0, 4), dtype=np.float64)

    h, w = g.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = g
    d = np.diff(padded, axis=1)
    rows, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)
    x0 = (cx if anchor == "left" else cx - w * cell / 2.0)
    y1 = cy + h * cell / 2.0
    out = np.empty((rows.size, 4), dtype=np.float64)
    out[:, 0] = x0 + starts * cell
    out[:, 1] = x0 + ends * cell
    out[:, 2] = y1 - (rows + 1) * cell
    out[:, 3] = y1 - rows * cell
    return out


# --- polarity ---------------------------------------------------------------
#
# The plate is written as a DARKFIELD mask with positive resist: the write
# defines where the chrome comes OFF, so the file must contain the CLEAR
# regions, not the metal ones. Every builder therefore has to be able to hand
# back the complement of its own features within its own cell.
#
# The complement is always computed ANALYTICALLY, never as a boolean. A
# whole-plate `box - features` over twelve million polygons is precisely the
# GEOS-style operation CLAUDE.md forbids on this host — an eight-minute run that
# had to be killed proved the point. Each structure here knows its own inverse
# in closed form: a lamellar grating's complement is the grating at duty 1-c
# shifted by c*d, a centred band's is the two strips either side of it, and a
# pinhole field's is the pinholes.

CLEAR = "clear"
METAL = "metal"


def invert_grating(
    cx: float, cy: float, w: float, h: float, period_um: float, duty: float,
    *, phase_um: float = 0.0, vertical: bool = True,
) -> np.ndarray:
    """The clear complement of :func:`_grating_rects` with the same arguments.

    Metal lines sit at ``[k*d, k*d + c*d]``, so the gaps are the grating at duty
    ``1 - c`` shifted by ``c*d``. Both are clipped to the same box, so their
    union is the box exactly.
    """
    return _grating_rects(
        cx, cy, w, h, period_um, 1.0 - duty,
        phase_um=phase_um + period_um * duty, vertical=vertical,
    )


def grating_array(cx: float, cy: float, w: float, h: float, period_um: float,
                  duty: float, *, phase_um: float = 0.0,
                  vertical: bool = True) -> tuple[dict[str, Any], np.ndarray]:
    """A lamellar grating as ONE array entry plus its two clipped edge stripes.

    Same lattice as :func:`_grating_rects` (lines at ``phase + k*period``). The
    interior — every stripe that lies wholly inside the band — is a single array
    reference, which is how the writer already emits the colour sub-gratings; a
    36 mm ladder of 2 um lines becomes one record instead of 18,000.

    The two PARTIAL stripes at the band's edges are returned as explicit
    rectangles, clipped to the band. Leaving them to the array's centre-in
    selection was fine at 5 um (the documented half-period slop) and wrong at
    173 um: a whole comb tooth protruded 86 um past the cell, and the metal and
    clear versions of eight parallax cells overlapped by exactly one stripe.
    With the band trimmed to whole stripes, centre-in and fully-inside coincide,
    and metal + clear tile the box to the nanometre.
    """
    if not vertical:
        raise ValueError("grating_array steps in x; use _grating_rects for horizontal")
    d, L = float(period_um), float(period_um) * float(duty)
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    k_first = int(math.ceil((x0 - phase_um) / d - 1e-9))
    k_last = int(math.floor((x1 - phase_um - L) / d + 1e-9))
    edges: list[np.ndarray] = []
    # partial stripe to the left of the first whole one
    s0, s1 = phase_um + (k_first - 1) * d, phase_um + (k_first - 1) * d + L
    if s1 > x0 + 1e-9:
        edges.append(_rect(max(s0, x0), y0, min(s1, x1), y1))
    # partial stripe to the right of the last whole one
    s0, s1 = phase_um + (k_last + 1) * d, phase_um + (k_last + 1) * d + L
    if s0 < x1 - 1e-9:
        edges.append(_rect(max(s0, x0), min(s1, x1), y0, y1)[:, [0, 1, 2, 3]] if False
                     else _rect(max(s0, x0), y0, min(s1, x1), y1))
    if k_last < k_first:
        # no whole stripe fits: the band is all edges
        return ({"rects": np.empty((0, 4)), "period_um": np.empty(0),
                 "line_um": np.empty(0), "phase_um": np.empty(0)}, _cat(*edges))
    bx0, bx1 = phase_um + k_first * d, phase_um + k_last * d + L
    band = _rect(bx0, y0, bx1, y1)
    return ({"rects": band, "period_um": np.array([d]), "line_um": np.array([L]),
             "phase_um": np.array([phase_um])}, _cat(*edges))


def grating_array_inverse(cx: float, cy: float, w: float, h: float,
                          period_um: float, duty: float, *,
                          phase_um: float = 0.0) -> tuple[dict[str, Any], np.ndarray]:
    """The clear complement of :func:`grating_array`: duty ``1-c`` at phase
    ``c*d``. Together with the original it tiles the band exactly."""
    return grating_array(cx, cy, w, h, period_um, 1.0 - duty,
                         phase_um=phase_um + period_um * duty)


def _frame_rects(cx: float, cy: float, w: float, h: float, t: float = 60.0) -> np.ndarray:
    """A hairline box around a cell, so the dice/inspect boundary is visible."""
    hw, hh = w / 2.0, h / 2.0
    return _cat(
        _rect(cx - hw, cy + hh - t, cx + hw, cy + hh),
        _rect(cx - hw, cy - hh, cx + hw, cy - hh + t),
        _rect(cx - hw, cy - hh, cx - hw + t, cy + hh),
        _rect(cx + hw - t, cy - hh, cx + hw, cy + hh),
    )
