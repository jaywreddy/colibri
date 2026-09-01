"""From a colour photograph to a per-pixel PERIOD field — the authoring pipeline.

``colourzone`` says what a colour zone costs and why a period RATIO is the only
inter-zone relationship that survives a change of view. This module is how you
get zones in the first place: it turns a colour source image into an integer
``period_id`` map plus a table of periods, which is the one input
``screenrects`` needs to sub-grate a halftone.

Everything is a period field
----------------------------
The three variants the box asks for are not three mechanisms, they are three
ways of filling in one array:

  plain   every pixel id 0 — no sub-grating, solid gold bands.
  hue     id from the pixel's own HUE, quantised onto the ladder. A red petal
          gets a long period and stays red-ish; a blue one gets a short period.
  zones   id from a RULE — named regions (the sweater, the glasses) take a
          fixed rung, and a region may delegate to hue so that a carpet of
          flowers picks up roughly one rung per flower without anyone
          segmenting individual flowers.

Collapsing them into one representation is what makes "plain" free to test
against: it is the same code path with an empty rule list, so a difference
between the plain and coloured cells on the plate is the colour and nothing
else.

Why the field is COARSENED
--------------------------
Quantising hue per pixel produces salt-and-pepper: neighbouring petals differ
by a rung, runs break every few cells, and the rectangle count (see
``screenrects``) goes up by an order of magnitude for detail no eye can resolve
at 87 um. ``coarsen_px`` mode-filters the id map at roughly one flower, which
both restores long runs and makes the result look INTENTIONAL — patches of
colour rather than chromatic noise. It is the same reason the earlier
random-period experiment read as a warm cast: colour needs to be organised at a
scale the eye integrates over, or it averages away.

The litho floor gates the blue end
----------------------------------
A rung's gold line is ``base * scale * duty``. The ladder's blue end is the
finest, so the whole plan is printable only if
``base * min(ladder) * duty >= MIN_FEATURE_UM`` — which is exactly
``colourzone.min_base_period_um``. :func:`build_period_field` reports it per
rung rather than silently writing sub-floor lines, because on this build the
blue end is genuinely the thing that gets cut.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .colourzone import MIN_FEATURE_UM, hue_ladder, min_base_period_um

# Hue angle that maps to the SHORT (blue) end of the ladder.
BLUE_HUE_DEG = 240.0

# Where the hue circle is CUT before mapping onto the ladder. Everything above
# this folds to negative and clamps to the red end.
#
# This matters more than it looks. Hue is a circle and the spectrum is a line:
# 0 and 360 are both red, and the magentas in between (240-360) are not
# spectral colours at all — no single grating period produces them. Cutting at
# 240, the obvious choice, sends every pink petal (hue 330-350, and the 90th
# percentile of this carpet is 347) to the BLUE end of the ladder, which is the
# opposite of its colour. Cutting at 300 puts the magentas on the red side of
# the fold where they belong and leaves only true violets at the blue end.
HUE_WRAP_DEG = 300.0


@dataclass(frozen=True)
class Rule:
    """One named region and the rung it takes.

    Rules are tested in order and the FIRST match wins, so put the specific
    ones (glasses) before the general ones (anything saturated). ``bbox`` is
    ``(x0, y0, x1, y1)`` in fractions of the image, and is what keeps a colour
    rule from firing on the far side of the frame — the sweater's sage green
    and a green leaf in the carpet are the same hue, and only position tells
    them apart.
    """

    name: str
    hue_deg: tuple[float, float] | None = None
    sat: tuple[float, float] = (0.0, 1.0)
    val: tuple[float, float] = (0.0, 1.0)
    bbox: tuple[float, float, float, float] | None = None
    mode: str = "fixed"
    """``fixed`` pins the rung, ``hue`` takes it from each pixel's own hue, and
    ``plain`` CLAIMS the region without colouring it. The third is what keeps a
    later, looser rule off a region you want left as gold — the crowd on the
    pavement above the carpet is as saturated as the petals are and no colour
    rule can tell them apart, but a bbox can."""
    period_scale: float = 1.0
    min_blob_px: int = 0
    """Drop connected components smaller than this — speckle costs rectangles
    and reads as noise."""
    close_px: int = 0
    """Morphological closing, in pixels, to knit a broken region together."""
    mask_png: str = ""
    """Path to a painted 8-bit mask (>127 is inside), ANDed with the colour and
    bbox criteria. Colour thresholding cannot select a hairline that is not a
    distinct colour along its whole length — the shadowed half of a spectacle
    frame measures the same as hair — so an authored feature that must look
    deliberate is authored, not inferred. Relative paths resolve against
    ``ASSET_DIR``."""
    dilate_px: int = 0
    """Grow the region after cleaning. A zone thinner than a couple of halftone
    line periods has no bands to sub-grate, so a hairline feature that matters
    (the spectacle frames) has to be thickened deliberately or it will simply
    not be coloured on the plate."""

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "hue", "plain"):
            raise ValueError(
                f"mode must be 'fixed', 'hue' or 'plain' (got {self.mode!r})"
            )
        if self.period_scale <= 0.0:
            raise ValueError(f"period_scale must be > 0 (got {self.period_scale})")


@dataclass(frozen=True)
class ColourPlan:
    """A complete recipe for turning one image into a period field."""

    name: str = ""
    mode: str = "zones"
    """``plain`` | ``hue`` | ``zones``."""
    base_period_um: float = 5.0
    ladder_steps: int = 12
    spread: float = 1.45
    duty: float = 0.5
    coarsen_px: int = 6
    hue_min_value: float = 0.14
    hue_min_sat: float = 0.10
    """Below these, a pixel's HUE is not a measurement and is left plain gold.

    Hue is an angle on a cone: as value or saturation goes to zero the angle is
    determined by a few least-significant bits and lands anywhere. Dark hair on
    this portrait computes to teal often enough that a spectacle-frame rule
    picked it up, and in ``hue`` mode it takes a random rung — so the fix is not
    a better hue window, it is refusing to read hue where there is none. Only
    hue-DERIVED assignment is gated; an authored region still colours its own
    shadows, which is what you want for a sweater."""
    hue_equalize: bool = False
    """Spread the region's OWN hue distribution across the whole ladder.

    A direct hue map is faithful but wasteful whenever a subject's colours are
    clustered, and most are: 90% of this flower carpet's pixels fall in 33% of
    the ladder because the petals are almost all warm. Equalising ranks the
    hues and spends the full ladder on the range actually present, so
    neighbouring petals separate by several rungs instead of one. It trades
    absolute fidelity — the hues are no longer even approximately the
    photograph's — for DISTINCTNESS, which is what "one colour per flower"
    is really asking for. Ordering is preserved either way."""
    rules: tuple[Rule, ...] = ()
    sub_angle_deg: float = 90.0
    """Sub-grating orientation. 90 crosses the horizontal screen lines; see
    ``screenrects.stripe_plan`` for why parallel is a no-op."""

    def __post_init__(self) -> None:
        if self.mode not in ("plain", "hue", "zones"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.ladder_steps < 1:
            raise ValueError("ladder_steps must be >= 1")

    def ladder(self) -> list[float]:
        return hue_ladder(self.ladder_steps, self.spread)

    def periods_um(self) -> dict[int, float]:
        """Period per id. Id 0 is reserved for plain gold and has no period."""
        return {i + 1: self.base_period_um * s for i, s in enumerate(self.ladder())}

    def min_printable_base_um(self) -> float:
        return min_base_period_um(self.ladder_steps, self.spread, self.duty)


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hue_deg, sat, val) from a float RGB image in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    c = mx - mn
    h = np.zeros_like(mx)
    nz = c > 1e-6
    # Standard piecewise hue; np.where keeps it branchless over the image.
    rm, gm, bm = (mx == r) & nz, (mx == g) & nz, (mx == b) & nz
    h = np.where(rm, ((g - b) / np.where(nz, c, 1.0)) % 6.0, h)
    h = np.where(gm, (b - r) / np.where(nz, c, 1.0) + 2.0, h)
    h = np.where(bm, (r - g) / np.where(nz, c, 1.0) + 4.0, h)
    hue = (h * 60.0) % 360.0
    sat = np.where(mx > 1e-6, c / np.where(mx > 1e-6, mx, 1.0), 0.0)
    return hue.astype(np.float32), sat.astype(np.float32), mx.astype(np.float32)


def hue_to_scale(hue_deg: np.ndarray, spread: float = 1.45) -> np.ndarray:
    """Hue angle -> period scale. Red is the LONG period, blue the short one.

    ``lambda = d * k`` at a fixed view, so a longer period diffracts a longer
    wavelength. Mapping red to the top of the ladder therefore keeps the
    plate's colours in the same ORDER as the photograph's, even though the
    absolute hue at any moment is set by where the viewer is standing.

    The circle is cut at :data:`HUE_WRAP_DEG` first — see there for why 240 is
    the wrong cut and turns pink petals blue.
    """
    h = np.asarray(hue_deg, dtype=np.float32)
    h = np.where(h > HUE_WRAP_DEG, h - 360.0, h)
    t = np.clip(h / BLUE_HUE_DEG, 0.0, 1.0)
    return (spread ** (0.5 - t)).astype(np.float32)


def _quantize_to_ladder(scale: np.ndarray, ladder: Sequence[float]) -> np.ndarray:
    """Nearest rung index (0-based) for each scale."""
    lad = np.asarray(ladder, dtype=np.float32)
    if lad.size == 1:
        return np.zeros(scale.shape, dtype=np.int32)
    # Ladder is geometric, so match in log space or the low rungs win too often.
    d = np.abs(np.log(np.clip(scale, 1e-6, None))[..., None] - np.log(lad)[None, None, :])
    return np.argmin(d, axis=-1).astype(np.int32)


def _equalize_scale(
    scale: np.ndarray, where: np.ndarray, ladder: Sequence[float]
) -> np.ndarray:
    """Re-rank ``scale`` inside ``where`` so it spans the ladder uniformly.

    Rank-based, not a histogram stretch: a stretch is defeated by a single
    outlier petal at the far end of the hue circle, and this carpet has a 1%
    tail at hue 215 that would otherwise absorb most of the range. Ranking is
    monotone, so the ORDER of the hues survives — a redder petal still gets a
    longer period than a yellower one — while the spacing becomes uniform.
    """
    lad = np.asarray(ladder, dtype=np.float32)
    out = np.array(scale, dtype=np.float32, copy=True)
    vals = scale[where]
    if vals.size == 0 or lad.size < 2:
        return out
    order = np.argsort(vals, kind="stable")
    t = np.empty(vals.size, dtype=np.float32)
    # DESCENDING: argsort is ascending, so the smallest scale (the bluest pixel)
    # lands last and must get t = 1, the SHORT period. Ranking it 0 instead maps
    # blue to the long end and silently reverses every colour in the picture —
    # which is what this did until test_equalising_preserves_the_ORDER_of_the_hues
    # caught it.
    t[order] = np.linspace(1.0, 0.0, vals.size, dtype=np.float32)
    # t = 0 is the LONGEST period (reddest), matching hue_to_scale's direction.
    out[where] = np.exp(np.log(lad.max()) + t * (np.log(lad.min()) - np.log(lad.max())))
    return out


def _mode_filter(ids: np.ndarray, size: int) -> np.ndarray:
    """Majority filter — the coarsener. See the module docstring.

    Implemented as a per-id maximum-count vote rather than scipy's rank filter
    because ids are nominal, not ordinal: a median of rung 1 and rung 11 would
    invent rung 6, which is a colour neither region asked for.
    """
    if size < 2:
        return ids
    from scipy import ndimage

    n = int(ids.max()) + 1
    if n <= 1:
        return ids
    best = np.zeros(ids.shape, dtype=np.float32)
    out = np.zeros(ids.shape, dtype=np.int32)
    for k in range(n):
        cnt = ndimage.uniform_filter((ids == k).astype(np.float32), size=size)
        win = cnt > best
        best = np.where(win, cnt, best)
        out = np.where(win, k, out)
    return out


def _clean(
    mask: np.ndarray, close_px: int, min_blob_px: int, dilate_px: int = 0
) -> np.ndarray:
    if close_px <= 0 and min_blob_px <= 0 and dilate_px <= 0:
        return mask
    from scipy import ndimage

    if close_px > 0:
        st = np.ones((close_px, close_px), dtype=bool)
        mask = ndimage.binary_closing(mask, structure=st)
    if min_blob_px > 0:
        lab, n = ndimage.label(mask)
        if n:
            sizes = np.bincount(lab.ravel())
            small = np.isin(lab, np.nonzero(sizes < min_blob_px)[0])
            mask = mask & ~small
    if dilate_px > 0:
        mask = ndimage.binary_dilation(
            mask, structure=np.ones((dilate_px, dilate_px), dtype=bool)
        )
    return mask


ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "bitmaps"


def _load_mask_png(path: str, shape: tuple[int, int]) -> np.ndarray:
    p = Path(path)
    if not p.is_absolute():
        p = ASSET_DIR / path
    if not p.is_file():
        raise FileNotFoundError(f"rule mask not found: {p}")
    im = Image.open(p).convert("L")
    if im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(im) > 127


def _rule_mask(
    rule: Rule, hue: np.ndarray, sat: np.ndarray, val: np.ndarray
) -> np.ndarray:
    m = (sat >= rule.sat[0]) & (sat <= rule.sat[1])
    m &= (val >= rule.val[0]) & (val <= rule.val[1])
    if rule.hue_deg is not None:
        lo, hi = rule.hue_deg
        m &= ((hue >= lo) & (hue <= hi)) if lo <= hi else ((hue >= lo) | (hue <= hi))
    if rule.bbox is not None:
        h, w = hue.shape
        x0, y0, x1, y1 = rule.bbox
        bb = np.zeros(hue.shape, dtype=bool)
        bb[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = True
        m &= bb
    if rule.mask_png:
        m &= _load_mask_png(rule.mask_png, hue.shape)
    return _clean(m, rule.close_px, rule.min_blob_px, rule.dilate_px)


def build_period_field(
    rgb: np.ndarray, plan: ColourPlan
) -> tuple[np.ndarray, dict[int, float], dict[str, Any]]:
    """Colour image -> ``(period_id, periods_um, report)``.

    ``rgb`` is float in [0, 1], shape (H, W, 3). ``period_id`` is int32 at the
    same resolution, 0 meaning plain gold. The report carries per-rung coverage
    and printability, and per-rule coverage, so a plan can be judged before it
    becomes 4 million rectangles.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")
    rgb = rgb[..., :3]
    ladder = plan.ladder()
    periods = plan.periods_um()
    ids = np.zeros(rgb.shape[:2], dtype=np.int32)
    rule_report: list[dict[str, Any]] = []

    if plan.mode != "plain":
        hue, sat, val = _hsv(rgb)
        scale = hue_to_scale(hue, plan.spread)

        if plan.mode == "hue":
            ok = (val >= plan.hue_min_value) & (sat >= plan.hue_min_sat)
            if plan.hue_equalize:
                scale = _equalize_scale(scale, ok, ladder)
            ids = np.where(ok, _quantize_to_ladder(scale, ladder) + 1, 0)
            if plan.coarsen_px > 1:
                ids = _mode_filter(ids, plan.coarsen_px)
            rule_report.append({"name": "whole frame", "mode": "hue", "frac": 1.0})
        else:
            # TWO passes, and the order is the whole reason the glasses survive.
            # Coarsening is there to organise HUE-derived speckle; run it over an
            # authored region too and any stroke thinner than the filter window
            # is voted out of existence. The teal frames are 4 px of 900 — a
            # 7 px majority filter erased them completely. So: hue rules first,
            # coarsen, then paint the fixed rules on top untouched.
            claimed = np.zeros(rgb.shape[:2], dtype=bool)
            masks: list[tuple[Rule, np.ndarray]] = []
            for rule in plan.rules:
                m = _rule_mask(rule, hue, sat, val) & ~claimed
                claimed |= m
                masks.append((rule, m))
                rule_report.append(
                    {
                        "name": rule.name,
                        "mode": rule.mode,
                        "frac": round(float(m.mean()), 4),
                        "period_scale": rule.period_scale if rule.mode == "fixed" else None,
                    }
                )

            hue_any = np.zeros(rgb.shape[:2], dtype=bool)
            for rule, m in masks:
                if rule.mode == "hue":
                    hue_any |= m
            hue_any &= (val >= plan.hue_min_value) & (sat >= plan.hue_min_sat)
            if plan.hue_equalize and hue_any.any():
                scale = _equalize_scale(scale, hue_any, ladder)
            hue_rung = _quantize_to_ladder(scale, ladder) + 1
            ids = np.where(hue_any, hue_rung, ids)
            if plan.coarsen_px > 1:
                ids = _mode_filter(ids, plan.coarsen_px)
            for rule, m in masks:
                if rule.mode == "fixed":
                    rung = int(np.argmin(np.abs(np.log(np.asarray(ladder))
                                                - math.log(rule.period_scale)))) + 1
                    ids = np.where(m, rung, ids)

    floor_ok = True
    rungs = []
    for pid, d in sorted(periods.items()):
        line = d * plan.duty
        gap = d * (1.0 - plan.duty)
        ok = line >= MIN_FEATURE_UM - 1e-9 and gap >= MIN_FEATURE_UM - 1e-9
        floor_ok &= ok or not bool((ids == pid).any())
        rungs.append(
            {
                "id": pid,
                "period_um": round(d, 4),
                "line_um": round(line, 4),
                "frac": round(float((ids == pid).mean()), 4),
                "clears_litho_floor": bool(ok),
            }
        )

    report = {
        "mode": plan.mode,
        "base_period_um": plan.base_period_um,
        "min_printable_base_um": round(plan.min_printable_base_um(), 3),
        "base_is_printable": plan.base_period_um >= plan.min_printable_base_um() - 1e-9,
        "spread": plan.spread,
        "duty": plan.duty,
        "coarsen_px": plan.coarsen_px,
        "hue_equalize": plan.hue_equalize,
        "frac_coloured": round(float((ids > 0).mean()), 4),
        "rungs_used": int(np.unique(ids[ids > 0]).size),
        "rungs": rungs,
        "rules": rule_report,
        "all_used_rungs_printable": bool(floor_ok),
    }
    return ids, periods, report


def load_rgb(
    path: str | Path,
    *,
    crop: tuple[float, float, float] | None = None,
    size: int = 1400,
) -> np.ndarray:
    """Colour twin of ``imageprep.load_gray`` — same crop convention, so the
    darkness map and the period field are guaranteed to be in register."""
    im = Image.open(Path(path)).convert("RGB")
    if crop is not None:
        w, h = im.size
        fx, fy, fs = crop
        x0, y0 = int(fx * w), int(fy * h)
        side = int(fs * w)
        im = im.crop((x0, y0, min(w, x0 + side), min(h, y0 + side)))
    im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def save_field(ids: np.ndarray, path: str | Path) -> Path:
    """Write a period-id map as an 8-bit PNG sidecar.

    Ids are small integers, so the file is a paletteless L image and stays
    lossless — reading it back must give the same field or the plate would not
    match the plan it was reviewed from.
    """
    ids = np.asarray(ids)
    if ids.max() > 255:
        raise ValueError(f"too many rungs for an 8-bit sidecar ({ids.max()})")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ids.astype(np.uint8), "L").save(p)
    return p


def load_field(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(Path(path)).convert("L"), dtype=np.int32)


def plan_to_json(plan: ColourPlan) -> str:
    return json.dumps(asdict(plan), indent=2, sort_keys=True)


def plan_from_json(text: str) -> ColourPlan:
    raw = json.loads(text)
    rules = tuple(
        Rule(
            **{
                k: (tuple(v) if isinstance(v, list) else v)
                for k, v in r.items()
            }
        )
        for r in raw.pop("rules", [])
    )
    return ColourPlan(rules=rules, **raw)


# --- presets ----------------------------------------------------------------
# The reference plan, and the worked example of the rule format. Order matters:
# the glasses are inside the face bbox and share their hue band with nothing
# else on the frame, so they are claimed first; the sweater's sage green is the
# same hue as a carpet leaf and only its bbox separates them; the carpet is last
# and takes whatever saturated pixels are left, by hue.

PAULA_ZONES = ColourPlan(
    name="portrait-paula",
    mode="zones",
    base_period_um=5.0,
    ladder_steps=12,
    spread=1.45,
    duty=0.5,
    coarsen_px=7,
    hue_equalize=True,
    rules=(
        Rule(
            name="crowd",
            # The pavement and the people above the carpet are as saturated as
            # the petals; only position separates them. Claimed first and left
            # as plain gold so the flower rule cannot reach them.
            bbox=(0.0, 0.0, 0.34, 0.17),
            mode="plain",
        ),
        Rule(
            name="glasses",
            # The teal window alone is not enough: dark hair computes to teal
            # (val 0.04-0.11, and plenty of it), and the SHADOWED left frame
            # sits at val 0.09 — indistinguishable by brightness. So the bbox
            # is cropped to the eye band between the two falls of hair, which
            # is the only thing that actually separates them.
            # Painted, not thresholded. The colour rule catches the two LIT
            # top arcs and nothing else, because the shadowed rim measures the
            # same as hair; the mask is fitted to those arcs by
            # tools/dev/paint_glasses_mask.py and closes the shape.
            mask_png="portrait-paula.glasses.png",
            bbox=(0.62, 0.06, 0.90, 0.30),
            mode="fixed",
            period_scale=0.83,          # blue end of the ladder — teal frames
        ),
        Rule(
            name="sweater",
            # Sage knit measures sat 0.08 median, not the 0.3 a "green sweater"
            # suggests, and hue is unstable that close to grey (p10..p90 spans
            # 19..331 on the shoulder). So the hue window is deliberately wide
            # and it is the BBOX plus a low saturation CAP that separate the
            # sweater from the jeans below it and the tan bag beside it.
            hue_deg=(45.0, 200.0),
            sat=(0.03, 0.42),
            val=(0.12, 0.78),
            bbox=(0.44, 0.24, 1.0, 0.74),
            mode="fixed",
            period_scale=1.0,           # mid ladder — reads as the base gold
            close_px=5,
            min_blob_px=600,
        ),
        Rule(
            name="flowers",
            sat=(0.30, 1.0),
            val=(0.20, 1.0),
            bbox=(0.0, 0.05, 0.57, 1.0),
            mode="hue",                 # one rung per petal colour
            min_blob_px=24,
        ),
    ),
)

PAULA_HUE = ColourPlan(
    name="portrait-paula-hue",
    mode="hue",
    base_period_um=5.0,
    ladder_steps=12,
    spread=1.45,
    duty=0.5,
    coarsen_px=7,
    # NOT equalised: over the whole frame the hues are already spread (carpet
    # warm, sweater green, jeans blue), and equalising would push the jeans and
    # the sky apart at the expense of separating one petal from the next.
    hue_equalize=False,
)

PAULA_PLAIN = ColourPlan(name="portrait-paula-plain", mode="plain")

PRESETS: dict[str, ColourPlan] = {
    "paula-zones": PAULA_ZONES,
    "paula-hue": PAULA_HUE,
    "paula-plain": PAULA_PLAIN,
}
