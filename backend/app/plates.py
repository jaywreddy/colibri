"""Plate composition — central optical pattern + decorative gold frame.

A ``PlateSpec`` is the user-level recipe for one glass plate:
  * a central pattern slug + params (any existing Pattern subclass), and
  * a frame spec (theme, algorithm, dials, seed) wrapping it.

The runtime path is ``materialize_plate`` -> ``_raster_compose_plate``: the
cached central-pattern PNGs and a RasterPen-rendered frame composite straight
into PIL images (no Shapely on the hot path). ``compose_plate`` is the
polygon-space equivalent, kept for tests; it CONCATENATES the frame polygons
onto the front gold layer (never ``unary_union`` — see ``_concat_polygons``).
The back layer is untouched — frames are decorative metallization on the
viewer-facing side.

SVGs are NOT built at materialize time: ``ensure_plate_svg`` builds the fab
pair lazily on first export request (the polygon/SVG path is ~40× slower
than raster and the interactive UI never needs it).

Plate layout: the ``weld_margin_um`` rim around the plate edge stays blank
glass (reserved for copper foil + solder); the frame band sits inside that
on the *active rect*; the square central aperture sits inside the band at
``min(active) - 2·band``. The central pattern always generates square so
existing patterns (all of which take a single ``extent_um`` scalar) drop in
without changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon

from .export_svg import to_svg
from .patterns.base import GeneratedPattern, Substrate, ensure_multipolygon, registry
from .patterns.frames import (
    RectFrame,
    Scene,
    generate_frame,
    render_scene_to_image,
    render_scene_to_svg,
    scene_to_multipolygon,
)
from .patterns.effects.gratings import beat_delta_um
from .patterns.frames.api import FrameParams
from .rasterize import (
    DEFAULT_THUMBNAIL_METAL,
    THUMBNAIL_PALETTES,
    make_thumbnail,
)
from .service import (
    DATA_ROOT,
    cache_lock,
    extra_layer_url,
    heavy_compute_gate,
    read_json_cache,
    save_png_atomic,
    validate_params,
    variant_key,
    write_json_atomic,
    write_text_atomic,
)


def _concat_polygons(*sources: MultiPolygon) -> MultiPolygon:
    """Combine MultiPolygons by concatenation, not union.

    For the lithography raster we don't care that overlapping front-layer
    polygons appear twice — ``ImageDraw.polygon`` paints them with the same
    gold color and the result is visually identical. Skipping ``unary_union``
    on tens of thousands of small polygons (a dense frame's stroke buffer
    output) is what keeps compose_plate inside GEOS memory budgets.
    """
    geoms: list[Polygon] = []
    for src in sources:
        if src is None or src.is_empty:
            continue
        if isinstance(src, MultiPolygon):
            geoms.extend(g for g in src.geoms if not g.is_empty)
        elif isinstance(src, Polygon):
            geoms.append(src)
    return MultiPolygon(geoms)


_log = logging.getLogger("optics.plates")

PLATES_ROOT = DATA_ROOT / "plates"


@dataclass
class FrameSpec:
    """Frame generation knobs, serializable across the API.

    The four band-composition dials (``edge_gradient``, ``understory``,
    ``border_vine``, ``corner_fans``) pass straight through to the colonize
    band grower (see frames/api.py::FrameParams). They're surfaced here so a
    box can hand each face its OWN frame recipe — a distinct edge gradient +
    understory density per face makes every side read as a deliberately
    different engraved border even before theme assignment.
    """

    algorithm: str = "wreath"
    theme: str = "esmeralda"
    density: float = 1.0
    bloom: float = 0.6
    foliage: float = 0.6
    seed: int = 1
    band_um: float | None = None  # default: 12% of shorter plate side
    edge_gradient: float = 0.8   # outer-edge density bias (0 flat .. 1 strong)
    understory: float = 0.85     # density of the outer-band small-leaf infill
    border_vine: float = 1.15    # continuous running-ornament border line
    corner_fans: float = 1.0     # size/reach of the corner fan compositions
    # Wreath composition preset (wreath algorithm only): the ordered garland
    # generator's character. "garland2" = lush mixed-tropical garland — the
    # chosen default: keeps the laurel vine's ordered outward flow + corner
    # medallions but restores species variety, full band depth, and the
    # multi-direction moiré shimmer a single-species rank flattened. "laurel" =
    # dense single-species shingled rank (austere classic); "garland" = looser
    # mixed foliage; "clusters" = spaced rosette swag. Ignored by colonize.
    # See frames/algorithms/wreath.py::STYLES (DEFAULT_STYLE == garland2).
    wreath_style: str = "garland2"
    # MOTIF size dial (wreath only; colonize ignores it). Shrinks the leaves,
    # blooms, understory sprigs and corner sprig — and their spacing along the
    # vine with them, so finer motifs simply come more often and the band stays
    # filled. The band WIDTH and the vine gauge are untouched: use this to make
    # the foliage read finer/lacier without narrowing the frame. 1.0 = the
    # tuned production look (bit-identical to before this knob existed).
    motif_scale: float = 1.0

    def to_frame_params(self) -> FrameParams:
        return FrameParams(
            algorithm=self.algorithm,
            theme_slug=self.theme,
            density=self.density,
            bloom=self.bloom,
            foliage=self.foliage,
            seed=self.seed,
            frame_width_um=self.band_um,
            edge_gradient=self.edge_gradient,
            understory=self.understory,
            border_vine=self.border_vine,
            corner_fans=self.corner_fans,
            wreath_style=self.wreath_style,
            motif_scale=self.motif_scale,
        )


@dataclass
class GlassSpec:
    thickness_um: float = 500.0
    material: str = "fused silica"
    n: float = 1.46


@dataclass
class PlateSpec:
    """User-level recipe for one glass plate."""

    pattern_slug: str
    pattern_params: dict[str, Any] = field(default_factory=dict)
    frame: FrameSpec = field(default_factory=FrameSpec)
    glass: GlassSpec = field(default_factory=GlassSpec)
    width_um: float = 30000.0
    height_um: float = 30000.0
    # Blank rim around the plate edges reserved for assembly welds — no
    # FRONT gold (foliage art) patterned anywhere inside this border. Default
    # 1 mm matches a typical UV-cure / solder joint allowance. Equals the box
    # keep-out (foil overlap + safety) when stamped from a box.
    weld_margin_um: float = 1000.0
    # Blank rim for the BACK carrier grating — foil overlap ONLY, so the
    # uniform moiré carrier spans the whole *exposed* face (wider than the
    # front weld margin). Mirrors assembly.back_window_um; stamped from the
    # box. ``None`` (standalone plate) falls back to weld_margin_um.
    back_margin_um: float | None = None
    # User-facing GRATING PITCH (μm) of the REAL fabricated back carrier + leaf
    # louvre family. Drives ``_carrier_recipe_data`` (fab back period = this;
    # front louvre = this × FRONT_GRATING_RATIO) and, through it, both the
    # preview shader uniforms and the fab SVG/GDS bake. Default 22 µm (the
    # historical BACK_CARRIER_PERIOD_UM). Litho floor 4 µm (2 µm line + 2 µm
    # gap at 50% duty). The switch/comb barrier faces keep their own 60 µm
    # architecture — this field does NOT touch them. Stamped from the box level
    # (see boxes.normalize_face_dims); part of the plate hash so a pitch change
    # regenerates the cached plate.
    carrier_pitch_um: float = 22.0
    # PER-FACE carrier scaling policy against the plate's real glass:
    #   "gap"   (default) — the fabricated carrier family scales with the
    #           paraxial gap t/n (see parallax_period_scale), preserving the
    #           designed reveal/beat TILT behavior on any stock. A no-op at
    #           the 500 µm fused-silica baseline (scale = 1), so every
    #           existing plate keeps byte-identical geometry.
    #   "fixed" — keep carrier_pitch_um as the literal fabricated pitch. On
    #           thick stock the reveals compress into sub-degree "refraction
    #           shimmer" — a deliberate aesthetic choice per face.
    # The switch/scanimation barrier periods scale with the gap regardless
    # (fab_center_period_um) — their crossing angle is the effect.
    carrier_scale_mode: str = "gap"
    # SINGLE-PLY face: this side of the box is ONE sheet of glass, not a bonded
    # pair, so there is no inner ply to write a back layer on. Everything moves
    # to the FRONT (outer) layer — the frame leaves AND the uniform carrier
    # grating, the carrier filling the back window MINUS the centerpiece art box
    # — and the BACK layer is left empty. The garland's shimmer then comes from
    # each leaf's grating beating against the carrier IN THE PLANE (the two-ply
    # moiré at zero gap: static fringes rather than travelling ones), which is
    # the honest thing a single ply can do. Reference geometry: the single-ply
    # block in ``export_fine.build_plate_fine``, which is what the production
    # witness dies (and every other single-ply face) are actually written from.
    single_ply: bool = False
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_slug": self.pattern_slug,
            "pattern_params": dict(self.pattern_params),
            "frame": asdict(self.frame),
            "glass": asdict(self.glass),
            "width_um": self.width_um,
            "height_um": self.height_um,
            "weld_margin_um": self.weld_margin_um,
            "back_margin_um": self.back_margin_um,
            "carrier_pitch_um": self.carrier_pitch_um,
            "carrier_scale_mode": self.carrier_scale_mode,
            "single_ply": self.single_ply,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlateSpec":
        frame_data = data.get("frame", {}) or {}
        glass_data = data.get("glass", {}) or {}
        return cls(
            pattern_slug=data["pattern_slug"],
            pattern_params=dict(data.get("pattern_params", {})),
            frame=FrameSpec(**frame_data),
            glass=GlassSpec(**glass_data),
            width_um=float(data.get("width_um", 30000.0)),
            height_um=float(data.get("height_um", 30000.0)),
            weld_margin_um=float(data.get("weld_margin_um", 1000.0)),
            back_margin_um=(
                float(data["back_margin_um"])
                if data.get("back_margin_um") is not None
                else None
            ),
            carrier_pitch_um=float(data.get("carrier_pitch_um", 22.0)),
            carrier_scale_mode=str(data.get("carrier_scale_mode", "gap")),
            single_ply=bool(data.get("single_ply", False)),
            label=data.get("label", ""),
        )

    def active_dims(self) -> tuple[float, float]:
        """(W, H) of the FRONT foliage area inside the weld border, in μm."""
        return (
            max(0.0, self.width_um - 2.0 * self.weld_margin_um),
            max(0.0, self.height_um - 2.0 * self.weld_margin_um),
        )

    def back_dims(self) -> tuple[float, float]:
        """(W, H) of the BACK carrier window inside the foil-overlap rim, in μm.

        Wider than ``active_dims`` — the back carrier grating spans the whole
        exposed face (plate minus foil overlap only). Falls back to the weld
        margin for a standalone plate that was never stamped from a box.
        """
        m = self.weld_margin_um if self.back_margin_um is None else self.back_margin_um
        return (
            max(0.0, self.width_um - 2.0 * m),
            max(0.0, self.height_um - 2.0 * m),
        )


def plate_hash(spec: PlateSpec) -> str:
    raw = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _band_um(spec: PlateSpec) -> float:
    if spec.frame.band_um is not None:
        return spec.frame.band_um
    aw, ah = spec.active_dims()
    return 0.12 * max(0.0, min(aw, ah))


# --- two-region mask encoding -----------------------------------------------
# The front/back plate masks are single-channel L images that carry TWO regions
# distinguished by intensity, so the one foliage_moire shader recipe can run the
# perimeter shimmer AND the centerpiece tilt-switch in a single pass:
#   FRAME_LEVEL  — the perimeter foliage frame (front) / carrier window (back).
#   ART_LEVEL    — the colibrí (front) / globe (back) centerpiece silhouette.
# Thresholds in plate.frag: art if r > ART_MIN, frame if r in (FRAME_MIN, ART_MIN).
# Keep these in sync with the shader's ART_MIN / FRAME_MIN constants.
#
# The frame band is NOT a single flat level any more: it carries a small PALETTE
# of graylevels that encode a PER-MOTIF grating-angle bucket, so every leaf and
# flower shimmers with its own fringe direction and beats differently against
# the uniform back carrier (the "cool effects" the art brief asked for). See
# FRAME_BUCKET0 / FRAME_BUCKET_STEP / N_FRAME_BUCKETS below and the matching
# decode in plate.frag (frameAngleFromLevel) + the fab bake in ensure_plate_svg.
FRAME_LEVEL = 110   # legacy alias / vine + fallback frame level (bucket 1)
ART_LEVEL = 255     # 1.0 in the shader
# Diffraction rainbow accent graylevel. Slots into the free gap between the
# frame-bucket top (166 = 0.651) and ART_LEVEL (255): the shader carves a third
# window RAINBOW = [0.72, 0.86) (RAINBOW_MIN/ART_MIN in plate.frag) and, when
# `rainbow_level` is present in recipe_data, treats these pixels as gold grating
# PLUS a travelling spectral sheen (preview stand-in for a real sub-5 µm fan).
# In fab (ensure_plate_svg) the same pixels are OR-ed with a 4.4 µm 45° grating
# from effects.gratings.diffraction_accent_grating. Small designed accent zones
# only (steam-curl tips, gear hub, monogram flourish tips) —
# keep them tiny so the sub-5 µm feature count stays inside the lattice budget.
RAINBOW_LEVEL = 200
# Fab accent-grating parameters, imported (never copied) so recipe_data and the
# baked mask always quote the same grating — see the rainbow_* keys below.
from .patterns.effects.gratings import (  # noqa: E402
    DIFFRACTION_ACCENT_ANGLE_DEG as _DIFFRACTION_ACCENT_ANGLE_DEG,
    DIFFRACTION_ACCENT_DUTY as _DIFFRACTION_ACCENT_DUTY,
    DIFFRACTION_ACCENT_PERIOD_UM as _DIFFRACTION_ACCENT_PERIOD_UM,
    INTERLEAVE_BAND_PITCH_UM as _INTERLEAVE_BAND_PITCH_UM,
)

# Frame-band angle-bucket palette. Levels live strictly inside the shader's
# frame window (FRAME_MIN·255 = 51 .. RAINBOW_MIN·255 ≈ 183.6) so they still classify
# as "frame" (not centerpiece art) while carrying an angle index. Bucket b sits
# at L = FRAME_BUCKET0 + b·FRAME_BUCKET_STEP; the shader recovers b by rounding
# and maps it to an angle offset spanning ±(N/2)·FRAME_ANGLE_SPAN_DEG.
FRAME_BUCKET0 = 96          # L of bucket 0 (~0.376)
FRAME_BUCKET_STEP = 14      # L per bucket (~0.055); 6 buckets top out at 166
N_FRAME_BUCKETS = 6
# Degrees between adjacent buckets. Kept SMALL on purpose: the moiré beat between
# the front louvre (rotated by base_off + (b-2.5)·span) and the back carrier is
# only visible when the net dθ stays within a few degrees of the carrier — with
# the fine preview carrier below, span·(N/2) must land the beat at a ~0.4-1 mm
# spacing. At 3.5° the six buckets span dθ ≈ -6.75°..+10.75° (relative to the
# base 2° offset), giving beats 0.35-0.85 mm each in its OWN orientation, so
# different motifs shimmer in visibly different fringe directions. (Was 11° —
# that fanned the buckets ±27° where the beats stayed >0.6 mm but their
# ORIENTATIONS barely differed and, combined with the coarse 340 µm carrier,
# the fringes did not travel with tilt. See CARRIER_ANGLE_OFFSET_DEG /
# PREVIEW_BACK_PERIOD_UM notes for the travel fix.)
# Retuned from 3.5 deg. The per-species fan is added ON TOP of
# CARRIER_ANGLE_OFFSET_DEG, so 3.0 + (b-2.5)*3.5 spanned -5.75..+11.75 deg and
# the outer buckets (fern, heliconia, wax palm - 17% of frame gold) beat at
# 103-137 um = 1.2-1.6 arcmin, BELOW the ~2 arcmin an eye resolves at 300 mm:
# those species were fabricated with a shimmer nobody would ever see. Because
# the beat depends on |crossing|, the fix is to centre the fan low rather than
# merely narrow it - 2.5 +/- 2.5*1.0 gives crossings 0..5 deg, keeping all six
# species in the 187-266 um (2.15-3.05 arcmin) band with six DISTINCT
# treatments. Pinned by tests/test_readability.py.
FRAME_ANGLE_SPAN_DEG = 1.0


def frame_level(bucket: int) -> int:
    """L-value for frame-band angle ``bucket`` (0..N_FRAME_BUCKETS-1)."""
    b = max(0, min(N_FRAME_BUCKETS - 1, int(bucket)))
    return FRAME_BUCKET0 + b * FRAME_BUCKET_STEP


def _frame_level_for(motif_key: str) -> int:
    """Graylevel for a motif family in the frame band (its angle bucket)."""
    return frame_level(MOTIF_ANGLE_BUCKET.get(motif_key, MOTIF_ANGLE_BUCKET["vine"]))


# Per-motif angle bucket assignment. Each species (and the vine) picks a bucket
# so neighbouring motifs of different kinds get visibly different fringe
# directions. Leaves/flowers spread across the palette; the vine sits mid-range.
# Keep in sync with the SVG bake (ensure_plate_svg) which reads the same table.
MOTIF_ANGLE_BUCKET: dict[str, int] = {
    "vine": 2,
    # flowers
    "heliconia": 4,
    "orchid": 0,
    "anthurium": 3,
    "coffee": 1,
    # leaves
    "wax_palm": 5,
    "plantain": 1,
    "fern": 4,
    "philodendron": 2,
}
# Fraction of the aperture the centerpiece silhouette fills (breathing gap to
# the frame band). The motif is scale-free; we render it into this sub-square.
CENTERPIECE_FILL = 0.86


# --- moiré carrier grating parameters ---------------------------------------
# The two layers each carry a fine line grating; the moiré shimmer is the beat
# between them. We give them BOTH a small period ratio and a small relative
# angle so the fringes read even head-on and travel with tilt (parallax slides
# the back grating under the front). Preview renders the gratings PROCEDURALLY
# in the shader (see plate.frag runFoliageMoire) so the plate PNGs stay clean
# silhouette masks with no baked ripple/aliasing. The FAB SVG path bakes true
# fine gratings at these micrometer periods (clipped to the silhouettes).
#
# Fab period: 22 µm lines/space (well inside the 400k-lattice budget for a
# 50 mm plate: ~50000/22 ≈ 2270 lines × a few spans each). Preview period is a
# derived, coarser value handled in-shader and is resolution-independent.
BACK_CARRIER_PERIOD_UM = 22.0     # back uniform grating period (fab)
# front period = back × ratio. The period MISMATCH is one of the two moiré-beat
# terms (the other is the angle offset below); together they set the fringe
# spacing. With the 220 µm preview carrier the 1.09 ratio beat is ~2.4 mm — big
# and mobile against the 3° angle offset (see PREVIEW_BACK_PERIOD_UM); on the fine
# 22 µm fab carrier the 1.06→1.09 change is a sub-µm shift in the baked front
# period — negligible for fab.
FRONT_GRATING_RATIO = 1.09
CARRIER_ANGLE_OFFSET_DEG = 2.5    # fan centre; see FRAME_ANGLE_SPAN_DEG
GRATING_DUTY = 0.5                # gold-line fraction of a period
# SINGLE-PLY faces (the photo sides) carry no carrier, so their leaves cannot
# beat against anything; instead every leaf (connected leaf cluster) is written
# as a FINE 50% grating at its own orientation. Zero-order reflection is the
# same for every orientation (flat 50% gold), but the first diffraction order
# leaves at lambda/p = 5.3 deg (green) in the plane perpendicular to the lines,
# so under a lamp each leaf flashes spectral colour at its own tilt and azimuth
# — a real single-layer, view-dependent shimmer (the physics of the colour
# zones in the photographs). 10 um keeps the lines at 5 um, well above the
# litho floor even at the acute tips where an angled line meets a leaf edge;
# 0.11 arcmin at 300 mm, so the leaf reads as smooth gold, never as a hatch.
#
# This is the period of the fills that HAVE one period ("lines", "crossed",
# "dots"). The shipping fill is "hue", whose period is per-family — see
# SINGLE_PLY_LEAF_HUE_PERIODS_UM and ``single_ply_leaf_period_um`` below, which
# is what the manifest and the preview period map both read.
SINGLE_PLY_LEAF_PERIOD_UM = 10.0

# HOW each family's leaf is filled. See ``app/leaf_fills.py`` for the physics and
# the file-size arithmetic of each mode; ``"lines"`` is the shipping default and
# reproduces the geometry this branch emitted before the knob existed.
#
#   "lines"    one period, one ANGLE per family (azimuth selects the family)
#   "hue"      one ANGLE (0°), one PERIOD per family (colour selects the family)
#   "hue2"     as "hue" but families alternate 0°/90° for two azimuths
#   "crossed"  "lines" plus a +90° set: 75 % gold, darker leaf, lattice clear
#   "dots"     square pads on a square lattice (measurement only — the written
#              polygon count is the leaf AREA over p², which no line fill pays)
SINGLE_PLY_LEAF_FILL = "hue"

# The per-family period ladder for the "hue"/"hue2" fills — one rung per frame
# bucket (N_FRAME_BUCKETS = 6). Every family flashes at the SAME tilt but in its
# own colour: at a fixed first-order angle θ the wavelength that comes back is
# λ = p·sin θ, so a ladder of periods is a ladder of hues at one view.
#
# This IS the photographs' own ladder — ``colourplan.PlanSpec(base_period_um=5.0,
# ladder_steps=6, spread=1.45)``, i.e. ``5.0 * colourzone.hue_ladder(6, 1.45)`` —
# so a garland family and a colour zone in the picture beside it flash the same
# hue at the same tilt, one optical vocabulary per die. 4.15 µm is the finest
# rung either uses; the 2 µm floor caps it at p ≥ 4.
#
# COARSER VARIANT: multiply every rung by 1.4 → (5.81, 6.26, 6.74, 7.27, 7.83,
# 8.43). Same hue ORDER, ~15 % fewer written polygons than the 6 µm default, and
# the first order still leaves at 4.4° (a 2.2° tilt) in green — the practical
# floor for telling the flash apart from the specular glint.
SINGLE_PLY_LEAF_HUE_PERIODS_UM = (4.15, 4.47, 4.82, 5.19, 5.59, 6.02)

# Gold fraction of a "dots" leaf. The pad is p·√c and the clear gap between pads
# is p(1 − √c), so at 50 % coverage the 2 µm floor needs p ≥ 6.83 µm.
SINGLE_PLY_LEAF_DOT_COVERAGE = 0.5


def single_ply_leaf_period_um(spec: "PlateSpec") -> float:
    """The leaf grating period a single-ply face's garland is WRITTEN at (0 on a
    two-ply face, whose leaves are moiré louvres rather than diffractive).

    ONE answer for both consumers — the manifest key ``single_ply_leaf_period_um``
    and the preview period map ``_single_ply_leaf_period_raster`` — because they
    describe the same gold. Under the ``"hue"``/``"hue2"`` fills the ladder IS the
    period (one rung per motif family, ``SINGLE_PLY_LEAF_HUE_PERIODS_UM``), so a
    single scalar can only be its MEAN; ``SINGLE_PLY_LEAF_PERIOD_UM`` is the
    period only for the fills that use one ("lines", "crossed", "dots"). The two
    used to disagree — the manifest advertised the 10 µm ``"lines"`` constant
    while the plate was written on the 4.15–6.02 µm ladder — which is how a
    renderer ends up drawing a sheen the mask does not have.
    """
    if not getattr(spec, "single_ply", False):
        return 0.0
    if SINGLE_PLY_LEAF_FILL in ("hue", "hue2") and SINGLE_PLY_LEAF_HUE_PERIODS_UM:
        return float(sum(SINGLE_PLY_LEAF_HUE_PERIODS_UM)
                     / len(SINGLE_PLY_LEAF_HUE_PERIODS_UM))
    return float(SINGLE_PLY_LEAF_PERIOD_UM)

# Preview (shader) grating period, in µm of PLATE surface. The shader draws the
# gratings ANALYTICALLY (fwidth-AA), so this is a resolution-independent visual
# scale, not a rastered feature — it can be far finer than the plate raster
# without aliasing.
#
# CRITICAL — RESOLUTION, not the old travel story: the box preview now draws the
# two layers as TWO REAL PlaneGeometry surfaces separated by the physical slab
# (see frontend BoxScene buildPlate / plate.frag runFoliageMoireLayer). The leaf
# moiré and its travel EMERGE from the perspective projection of the two real
# gratings — the front louvre and back carrier are each drawn on their own plane
# with NO in-shader parallax. On real planes a grating must resolve at
# >= ~2-3 px/period at the default box zoom or it aliases into a shimmering dot
# lattice (the two-plane rig matrix measured px/period: 70 µm → 0.88 px/period,
# sub-pixel → checkerboard aliasing; 160 µm → 2.0 px/period, mild sparkle only in
# the most foreshortened corner; 220 µm → 2.8 px/period, clean). The OLD 70 µm
# carrier was fine for the previous single-plane analytic shader (which
# supersampled the two-layer product) but is sub-pixel — and therefore unusable —
# for real planes. 220 µm resolves cleanly (2.8 px/period) while the beat stays
# big and mobile: with FRONT_GRATING_RATIO 1.09 the ratio beat is ~2.4 mm and the
# 3° angle offset adds a diagonal beat, and the beat still TRAVELS az6→az14 (rig-
# verified) because the two real planes shear against each other with tilt. Do
# NOT drop this below ~200 µm or the real-plane grating aliases again.
PREVIEW_BACK_PERIOD_UM = 220.0

# Preview magnification for the user-tunable carrier/louvre (grating pitch).
# The fab carrier can be as fine as the 4 µm litho floor; on the two REAL preview
# planes a period that small is deeply sub-pixel even at Pattern Scale 4× (4 µm ×
# 4 = 16 µm), so the analytic grating's anti-alias `collapse` fades it to flat
# gold and the pitch-driven fringes vanish. The preview therefore draws the
# carrier/louvre at ``pitch × PREVIEW_PITCH_MAGNIFY`` — PROPORTIONAL to the real
# pitch, so finer pitch still reads as finer, livelier fringes — chosen so the
# finest 4 µm preset lands at 20 µm preview (the same on-screen period the 20 µm
# default already resolves cleanly at Pattern Scale 4×). The advertised
# ``carrier_period_um`` and the ``fab_*`` fields stay the TRUE pitch; only the
# ``preview_*`` fields (bound by BoxScene) carry this magnified value. Pattern
# Scale multiplies on top. (The centerpiece 60 µm switch/comb is NOT magnified —
# it is not part of the pitch family; see uCenterPeriodUm binding.)
PREVIEW_PITCH_MAGNIFY = 5.0


# Centerpiece tilt-switch carrier. The colibrí (front) and globe (back) are
# filled with a VERTICAL stripe carrier drawn analytically in the shader; the
# back stripes ride half a period out of phase, and the substrate parallax
# biases which phase the eye samples — so one tilt reveals the bird, the other
# the globe. Vertical stripes → horizontal switch axis (+X). The centerpiece
# carrier is finer than the frame louvre so the switch is crisp and the art
# still reads as a solid silhouette when the stripes collapse (shader averages).
CENTER_CARRIER_PERIOD_UM = 170.0   # preview centerpiece stripe period (shader)
CENTER_SWITCH_AXIS_DEG = 0.0       # vertical stripes -> +X switch axis

# --- centerpiece SHADING MOIRE on the front-only shimmer faces --------------
# These four faces carry one figure and no second image to switch to. They used
# to be described as "front-only glimmer", which quietly conceded that their
# back plate did nothing — and on a two-ply build that is a plate's worth of
# glass and litho earning nothing.
#
# They were never short of a second grating. ``back_dims`` spans the whole
# exposed face, so the uniform carrier already runs underneath the figure. What
# they were short of was any reason for the two to BEAT: the centerpiece fill
# ran at a fixed CENTER_SWITCH_AXIS_DEG (0 deg) while the back carrier runs at a
# per-face angle ((seed*17) mod 180). Crossing two ~20 um gratings at 114 deg
# puts the beat at 18 um -- 0.21 arcmin, against the ~2 arcmin the eye needs,
# and FINER than either grating, which is the signature of crossing too steeply.
#
# So the fill now runs PARALLEL to the back carrier and takes its beat from a
# pitch mismatch. That is a fabrication choice before it is an aesthetic one:
# this build has no backside alignment, so front-to-back ROTATION is the one
# thing it cannot hold, and a beat derived from a crossing angle is at its mercy
# (0.45 deg designed -> 2801 um; 869 um if the flip lands a degree off; INFINITE
# if it lands square). A beat from delta-p is anchored in the mask geometry:
# 1635 um designed, 1372 um at half a degree of misregistration, and it cannot
# collapse to "no fringes" for any rotation error at all.
#
# The knob is the BEAT, not the pitch mismatch that produces it. 1.64 mm puts
# about a dozen bands across a 20 mm lid. Coarser is bolder AND more
# rotation-sensitive, because the delta it implies shrinks toward the rotation
# term. Delta is solved from it per face (gratings.beat_delta_um) because the
# carrier pitch is NOT fixed -- it is gap-scaled with the glass, so it is 22 um
# at the 500 um baseline and 63.5 um on the 1.5 mm stock this box is built from.
# Pinning delta instead would let the beat follow the carrier up by that same
# 2.9x and put one and a half bands on the lid.
CENTERPIECE_BEAT_UM = 1635.0
SHIMMER_MOIRE_SLUGS = frozenset(
    {"monogram-jp", "inscription-line", "jamon-tray", "food-pair-chirp"}
)
# FAB centerpiece stripe period. The A<->B (colibrí<->globe) switch happens
# after parallax walks HALF a period; at t=500 µm, n=1.46 the parallax is
# ~5.98 µm/deg, so a 60 µm period switches at ~5° tilt — squarely inside the
# comfortable 3-8° hand-tilt sweet spot the effects toolkit recommends (vs the
# ~14° "stiff" switch a 170 µm fab period would need). The preview stripe scale
# stays coarse (170 µm) for a visible louvre; the physical switch timing lives
# in this separate fab period, exactly like the frame preview/fab period split.
FAB_CENTER_PERIOD_UM = 60.0

# Water SCANIMATION (capybara back face). The centerpiece back-art region (the
# water band) is animated by the shader's N-phase water-scan path instead of the
# 2-phase colibrí/globe switch: travelling wavelets whose phase advances with
# the substrate parallax so the crests flow sideways as the box rocks. N phases
# and the ripple wavelength are surfaced to the shader; a full N-phase cycle
# maps to the same comfortable hand-tilt sweep the standalone pattern targets.
WATER_SCAN_N_PHASES = 4
# Crest spacing as a FRACTION of the ART-BOX width (the 0.86·aperture square the
# capybara/water live in), NOT the whole plate. The shader normalizes the μm
# wavelength against the art-box width so preview spacing matches the baked fab.
# 0.14 → ~7 legible streamlines across the band (the r3 vision default; keep in
# lock-step with capybara_scanimation.RIPPLE_WAVELENGTH_NORM).
WATER_RIPPLE_WAVELENGTH_FRAC = 0.14
# There is deliberately NO preview-only phase-advance pitch here. A
# WATER_PHASE_PITCH_PREVIEW_UM = 97 µm constant used to be published as
# ``water_phase_pitch_preview_um`` and documented (with a "Verified:" claim) as
# the divisor that walks the ripple phase in the shader — but the two-plane
# rewrite had already made the phase emerge GEOMETRICALLY, no line of GLSL ever
# read the uniform, and the shader has since dropped it (see plate.frag's
# water-scanimation FLOW RATE note). The flow rate is set by geometry: the outer
# comb reveals the next 1/N lane after center_period/N µm of substrate parallax
# (15 µm at the fab 60 µm pitch, N=4 → ~2.5°/phase through the T/n gap), which is
# also the fab timing. Retune it via the barrier pitch or N — a preview-only
# timing constant is exactly the procedural fake the renderer-honesty rule bans.
WATER_SCAN_SLUG = "capybara-scanimation"
# Barrier-interlace tilt-switch slugs (Task 3). These faces abandon the
# phase-offset construction (A on front, B on back, half-period shift) for a TRUE
# lenticular/Poemotion BARRIER INTERLACE: both silhouettes live on the BACK layer
# in alternating lanes (A even, B odd, lane pitch = half the barrier pitch) and
# the FRONT layer is a neutral slit comb over the FULL centerpiece art box —
# never clipped to the silhouettes (a union-gated comb is itself a static front
# image: its envelope is the union, which can never vanish under parallax — the
# measured ~0.31 front residual of the half rebuild). With the image-free comb,
# tilting one way shows ONLY A and the other ONLY B — a hard swap, not a
# redistribution. The barrier pitch is FAB_CENTER_PERIOD_UM (60 µm), so the swap
# crosses at the same comfortable ~5° tilt the old switch targeted. The preview
# shader draws the lanes/barrier procedurally (uSwitchInterlace); the fab SVG +
# fine GDS bake the same architecture at fab pitches (see ensure_plate_svg /
# export_fine), and the standalone generators mirror it (full-field comb there).
SWITCH_INTERLACE_SLUGS = frozenset(
    {"globe-duo-phase", "gear-quill-switch", "colibri-flap-phase"}
)
# BARE GLASS. A face whose slug is this gets NO frame, NO carrier and NO
# centerpiece on either layer — the production box's back and bottom are plain
# quartz on purpose (see patterns/blank.py). Every writer tests the slug once:
# the composed preview (``_raster_compose_plate``), the fab SVG (``_bake_plate_svg``)
# and the fine GDS (``export_fine.build_plate_fine``, via an all-empty ZoneMasks).
BLANK_SLUG = "blank"
# PHOTOGRAPH. The centerpiece is not a silhouette but a LINE SCREEN: horizontal
# bands whose height tracks the picture's coverage, front layer only, with the
# coloured bands carrying a diffraction sub-grating. ``_centerpiece_masks``
# returns None for it (there is no silhouette to place); ``_paste_centerpiece``
# stamps the bands, and ``photo_band_rects`` is the exact vector form both fab
# writers bake. See patterns/bitmap/photo.py for the tone model.
PHOTO_SLUG = "photo-halftone"
# Fab (SVG/GDS) barrier-grid parameters — the standalone builder's defaults
# (60 µm slit pitch, N=4 → 15 µm slot, 24 µm body-shimmer carrier). These bake
# the REAL slit barrier + interleaved ripple frames into the box back plate.
WATER_SCAN_FAB_PITCH_UM = 60.0
WATER_SCAN_FAB_CARRIER_UM = 24.0

# --- diffraction / moiré interleave in the accent zone ------------------------
# The accent used to be PURE 4.4 µm diffraction, so it could flash a rainbow but
# never shimmer, while the moiré louvre lived only in the frame band — two
# effects at two scales in two places that never met. The accent zone now
# carries BOTH, spatially interleaved in sub-acuity bands (see
# effects.gratings.INTERLEAVE_BAND_PITCH_UM for why interleaving beats nesting).
#
# Both front gratings are written at the SAME 45° accent axis so they share one
# grating-local frame and the band lattice is a scalar test in it; the moiré
# then comes from the PITCH difference against a dedicated back patch laid under
# the accent at a small crossing, rather than from the frame's per-species fan.
ACCENT_MOIRE_OFFSET_DEG = 2.5   # accent back patch vs its 45° front bands
# ---------------------------------------------------------------------------

# --- glass-derived fab periods ------------------------------------------------
# The 60 µm barrier periods above are TILT targets, not absolute pitches: the
# A↔B switch crosses after the back layer walks half a period, and the parallax
# rate is set by the paraxial gap t/n (~5.98 µm/deg at the 500 µm / n=1.46
# baseline → half of 60 µm ≈ 5° of tilt, the comfortable hand sweep). On other
# stock — e.g. the bonded build's 1.5 mm soda-lime ply (gap 987 µm, ~2.9× the
# baseline) — keeping the RAW 60 µm would triple the flicker rate, so the fab
# bake scales the barrier/scanimation periods with the spec's real gap and the
# ~5° crossing is preserved for ANY glass. Snapped to 0.5 µm; exactly the
# legacy constants at the baseline, so every existing 500 µm plate hash keeps
# byte-identical geometry. The FINE-pitch families (22 µm louvre carrier, 24 µm
# body shimmer, 4.4 µm rainbow) deliberately do NOT scale: on thick stock their
# reveals tighten into sub-degree refraction shimmer — a chosen look, and the
# 2 µm litho floor is the binding constraint in the other direction.
BASE_PARALLAX_GAP_UM = 500.0 / 1.46


def parallax_gap_um(spec: "PlateSpec") -> float:
    """Paraxial back-layer gap t/n for this plate's glass (um)."""
    n = spec.glass.n if spec.glass.n > 1.0 else 1.46
    return spec.glass.thickness_um / n


def parallax_period_scale(spec: "PlateSpec") -> float:
    """How much slower/faster this glass walks parallax vs the baseline."""
    return parallax_gap_um(spec) / BASE_PARALLAX_GAP_UM


def _snap_half_um(v: float) -> float:
    return round(v * 2.0) / 2.0


def fab_center_period_um(spec: "PlateSpec") -> float:
    """The switch-barrier fab period for THIS spec's glass (~5° crossing)."""
    return _snap_half_um(FAB_CENTER_PERIOD_UM * parallax_period_scale(spec))


def water_scan_fab_pitch_um(spec: "PlateSpec") -> float:
    """The scanimation frame pitch for THIS spec's glass (~2.5°/phase)."""
    return _snap_half_um(WATER_SCAN_FAB_PITCH_UM * parallax_period_scale(spec))


def _water_waterline_y(params: dict[str, Any] | None) -> float:
    """Effective capybara waterline for a face — art-box normalized y, 0 = top.

    The generator exposes it as the tunable ``waterline`` ParamSpec (0.4–0.85,
    default ``WATERLINE_Y``); every plate-side consumer used to hardwire the
    constant instead, so a face that set the param got its water band, calm patch
    and depth shear baked at 0.66 anyway — and the preview shader, freezing the
    same constant, had nothing to read. ONE resolver now feeds all three (the
    centerpiece mask, the fab SVG bake, and the ``water_waterline_y`` the shader
    binds), so mask and preview cannot drift.

    Out-of-range/non-numeric values fall back to the default rather than raising:
    the API validates against the ParamSpec (``service.validate_params``) but
    direct callers do not, and the flow field divides by ``1 - waterline_y``.
    """
    from .patterns.artistic.capybara_scanimation import WATERLINE_Y

    try:
        value = float((params or {}).get("waterline", WATERLINE_Y))
    except (TypeError, ValueError):
        return WATERLINE_Y
    # NaN-safe by construction (every comparison against NaN is False).
    if not 0.0 < value < 1.0:
        return WATERLINE_Y
    return value


def _carrier_recipe_data(spec: PlateSpec) -> dict[str, Any]:
    """Per-plate moiré-carrier knobs the shader + fab path both read.

    The FRAME grating pair (perimeter foliage shimmer): the front louvre is
    rotated by ``CARRIER_ANGLE_OFFSET_DEG`` relative to the back and scaled by
    ``FRONT_GRATING_RATIO``; a per-face base angle keyed off the frame seed
    keeps each face's carrier visually distinct without changing the physics.

    The CENTERPIECE carrier: a finer vertical stripe carrier with a ±X switch
    axis. On the barrier-interlace faces (SWITCH_INTERLACE_SLUGS) the shader
    draws the neutral comb + interleaved lanes at this pitch instead; on the
    front-only shimmer faces (monogram/inscription/food/jamón) it is the
    procedural glimmer carrier.
    """
    base_angle = (spec.frame.seed * 17.0) % 180.0
    # User-tunable grating pitch drives the REAL back carrier + leaf louvre
    # family (the switch/comb barrier faces keep their own 60 µm architecture,
    # untouched below). The louvre stays pitch × FRONT_GRATING_RATIO. Both the
    # preview period fields AND the fab period fields track this one value, so
    # the preview shader draws the true fabricated pitch (× uPatternScale) and
    # finer pitch reads as finer, livelier fringes. Floor 4 µm at the litho
    # limit. Falls back to the historical 22 µm default for a spec that predates
    # the field. See PlateSpec.carrier_pitch_um / BACK_CARRIER_PERIOD_UM.
    carrier_pitch = float(getattr(spec, "carrier_pitch_um", BACK_CARRIER_PERIOD_UM))
    if carrier_pitch <= 0:
        carrier_pitch = BACK_CARRIER_PERIOD_UM
    # PER-FACE carrier scaling (PlateSpec.carrier_scale_mode): in "gap" mode the
    # FABRICATED carrier family follows the paraxial gap t/n, so the reveal and
    # frame-louvre TILT behavior designed at the 500 µm baseline carries to any
    # stock (a no-op at the baseline itself, scale = 1). "fixed" keeps the
    # literal pitch — on thick stock the reveals compress into sub-degree
    # refraction shimmer (a per-face aesthetic choice). Floored at the 4 µm
    # carrier litho limit for very thin stock. The capybara BODY shimmer
    # (WATER_SCAN_FAB_CARRIER_UM below) deliberately stays FIXED in both modes:
    # it is a sparkle accent, not a phase-coded reveal.
    if getattr(spec, "carrier_scale_mode", "gap") != "fixed":
        carrier_pitch = max(4.0, _snap_half_um(carrier_pitch * parallax_period_scale(spec)))
    front_pitch = carrier_pitch * FRONT_GRATING_RATIO
    # Preview periods (what the shader actually draws) — magnified so even the
    # 4 µm litho-floor pitch resolves on the two real planes; see
    # PREVIEW_PITCH_MAGNIFY. Proportional to the real pitch, so finer pitch reads
    # as finer fringes.
    preview_carrier = carrier_pitch * PREVIEW_PITCH_MAGNIFY
    preview_front = preview_carrier * FRONT_GRATING_RATIO
    # Water scanimation: only the capybara back face turns it on (N>0). Every
    # other face emits N=0, so the shader's water-scan branch is a no-op and the
    # centerpiece keeps its own path (barrier interlace or legacy 2-phase
    # shimmer) — pixel-identical.
    is_water = spec.pattern_slug == WATER_SCAN_SLUG
    water_n = WATER_SCAN_N_PHASES if is_water else 0
    # Art-box registration for the flow wake. The centerpiece is a SQUARE of side
    # (CENTERPIECE_FILL·aperture) centered on the plate; on a non-square plate it
    # maps to different uv half-extents per axis. The shader transforms face-uv →
    # art-box uv with these so the wake (body center, calm patch) lands on the
    # capybara, and normalizes the μm crest wavelength against the art-box width.
    art_side_um = CENTERPIECE_FILL * _aperture(spec)
    art_half_w_uv = (0.5 * art_side_um / spec.width_um) if spec.width_um > 0 else 0.0
    art_half_h_uv = (0.5 * art_side_um / spec.height_um) if spec.height_um > 0 else 0.0
    water_ripple_um = WATER_RIPPLE_WAVELENGTH_FRAC * art_side_um if is_water else 0.0
    waterline_y = _water_waterline_y(spec.pattern_params)

    # Centerpiece fill. All three consumers -- the preview shader, the SVG bake
    # (ensure_plate_svg) and the fine export -- read exactly these two fields,
    # so setting them here is the whole change for a face's centerpiece grating.
    #
    # A shimmer face gets the shading-moire pair: parallel to the back carrier,
    # mismatched by CENTERPIECE_BEAT_DELTA_UM. Every other face keeps the
    # glass-derived barrier/comb pitch on the +X axis, because there the pitch
    # and axis set a SWITCH angle rather than a beat and must not be retuned for
    # fringe aesthetics.
    is_shimmer = spec.pattern_slug in SHIMMER_MOIRE_SLUGS
    center_period = (
        carrier_pitch + beat_delta_um(carrier_pitch, CENTERPIECE_BEAT_UM)
        if is_shimmer
        else fab_center_period_um(spec)
    )
    center_axis = base_angle if is_shimmer else CENTER_SWITCH_AXIS_DEG

    return {
        # Frame shader gratings — the REAL fabricated pitch (the preview shader
        # draws these exact μm values × uPatternScale; see BoxScene binding and
        # the Task 1b "1× = exact fab" note). ``carrier_period_um`` is the
        # canonical carrier pitch the manifest advertises + the assertion the
        # e2e harness checks; ``slit_period_um`` is the leaf louvre (pitch ×
        # ratio). Both equal the fab_* fields below — one source of truth.
        "carrier_period_um": carrier_pitch,
        "slit_period_um": front_pitch,
        # Magnified preview periods the shader actually draws (× uPatternScale).
        # Keep the fringes resolvable at the finest pitch; the advertised carrier/
        # slit above stay the TRUE pitch. See PREVIEW_PITCH_MAGNIFY.
        "preview_carrier_period_um": preview_carrier,
        "preview_slit_period_um": preview_front,
        "carrier_angle_deg": base_angle,
        "slit_axis_deg": base_angle + CARRIER_ANGLE_OFFSET_DEG,
        "grating_duty": GRATING_DUTY,
        # Per-motif frame-band angle encoding (shader decodes the mask graylevel
        # into a bucket, then adds bucket·span to the base slit angle so every
        # motif shimmers in its own direction). See plates.FRAME_BUCKET0 etc.
        "frame_bucket0": float(FRAME_BUCKET0),
        "frame_bucket_step": float(FRAME_BUCKET_STEP),
        "frame_bucket_count": float(N_FRAME_BUCKETS),
        "frame_angle_span_deg": FRAME_ANGLE_SPAN_DEG,
        # Centerpiece tilt-switch carrier (preview).
        "center_period_um": CENTER_CARRIER_PERIOD_UM,
        "switch_axis_deg": center_axis,
        # Barrier-interlace tilt switch (Task 3): the shader draws the neutral
        # barrier (outer) + interleaved A/B lanes (inner) at the fab barrier pitch
        # instead of the phase-offset carrier. True only on the hard-swap faces
        # (SWITCH_INTERLACE_SLUGS); every other face keeps the legacy 2-phase
        # centerpiece (which post-rebuild means the front-only shimmer faces).
        "switch_interlace": spec.pattern_slug in SWITCH_INTERLACE_SLUGS,
        # BARE GLASS: this face carries no gold at all, so the renderer can skip
        # its passes rather than sample two empty masks. Emitted unconditionally
        # (like switch_barrier_phase_um): a consumer that reads the flag on every
        # face cannot drift from it, and a key that appears on some faces and not
        # others is how a stale manifest quietly changes meaning.
        "blank": spec.pattern_slug == BLANK_SLUG,
        # SOLID ART: the centerpiece is a line-screen picture whose bands are
        # already the tone, so the shader must fill ART_LEVEL with plain gold
        # instead of running the procedural switch carrier over it — a second
        # grating on top would multiply the halftone's duty and wash the picture
        # out. (RAINBOW_LEVEL bands keep their diffraction sheen; those ARE the
        # colour sub-grating.)
        "art_solid": spec.pattern_slug == PHOTO_SLUG,
        # SINGLE PLY: one sheet of glass, so there is no second plane for a
        # carrier to beat against — CARRIER_COV = 0 for every photo-halftone
        # face, so the picture dissolves straight to bare glass instead. The
        # garland's leaves (single_ply_leaf_period_um below) are the only
        # grating this face writes; the shader draws them at zero gap, static
        # under tilt because there is no gap to shear across, which is the
        # truth about this face and not a limitation of the preview.
        "single_ply": bool(getattr(spec, "single_ply", False)),
        # the per-leaf diffractive grating a single-ply face writes its
        # garland at (0 on a two-ply face, whose leaves are moire louvres).
        # Under the "hue" fill this is the ladder's mean — see the helper.
        "single_ply_leaf_period_um": single_ply_leaf_period_um(spec),
        # RENDER IT LITERALLY: this face publishes ``files.literal_front`` /
        # ``files.literal_back`` — coverage rasters of the ACTUAL DRC-healed
        # chrome geometry the mask writer emits (``export_fine.build_plate_fine``),
        # not the graylevel zone codes in front.png/back.png. The renderer samples
        # those on the two pattern planes instead of synthesising gratings, so the
        # preview is a picture of the plate rather than an impression of it.
        # Unconditionally True on every composed face (an empty layer is written
        # as an all-zero raster, never as a missing file) for the same reason
        # ``blank`` is emitted unconditionally: a key that appears on some faces
        # and not others is how a stale manifest quietly changes meaning.
        "literal": True,
        # Solved barrier REGISTRATION, published rather than rediscovered. One
        # convention for all three consumers — the fab SVG bake
        # (``_barrier_masks``), the preview shader, and the six standalone
        # generators (``patterns/_helpers.barrier_lattice``): x in µm from the
        # FACE CENTRE, open-slit centres at ``k·switch_interlace_period_um +
        # switch_barrier_phase_um``, back channel A starting at that same
        # boundary on its +x side — so a +x back shift (+tilt) reveals channel
        # B, and the slit STRADDLES an A|B boundary (sharp at ±p/4, mud at
        # ±p/2). The comb and the interlace share ONE period by construction,
        # which is the whole registration.
        #
        # The phase is 0 because both lattices are anchored on the face centre,
        # which is also the centerpiece art-box centre at every aperture — the
        # extent-dependent phase ``barrier_lattice`` has to solve (its raster
        # starts at the tile EDGE) does not arise on this path. Emitted anyway,
        # unconditionally: a consumer that reads the number cannot drift from it,
        # and a hardcoded phase constant in a shader is exactly how the generator
        # side lost its registration. ``svg_bake_barrier_*`` carries the ACHIEVED
        # lattice when the budget raster forces a coarser one.
        "switch_interlace_period_um": fab_center_period_um(spec),
        "switch_barrier_phase_um": 0.0,
        # Water scanimation (capybara back face). N>0 makes the shader render an
        # N-phase travelling-ripple flow over the back-art (water) region of the
        # centerpiece; N=0 leaves the 2-phase colibrí/globe switch untouched.
        "water_scan_n": water_n,
        "water_ripple_wavelength_um": water_ripple_um,
        # (No ``water_phase_pitch_preview_um``: the preview-only phase-advance
        # pitch it carried was never read by any shader line and the uniform is
        # gone — the ripple phase advances geometrically off center_period/N. See
        # the note where WATER_PHASE_PITCH_PREVIEW_UM used to be defined.)
        #
        # Capybara BODY shimmer carrier: the period the fab path actually bakes
        # over the dry animal (WATER_SCAN_FAB_CARRIER_UM, 24 µm) — its own field
        # because it is NOT the frame carrier the preview had been reusing
        # (carrier_period_um / its magnified twin), so the two disagreed by the
        # 22-vs-24 µm gap on the one region the user looks at. True period plus a
        # magnified preview twin, same split as the frame pair. 0 elsewhere.
        "water_body_carrier_period_um": WATER_SCAN_FAB_CARRIER_UM if is_water else 0.0,
        "water_body_carrier_preview_um": (
            WATER_SCAN_FAB_CARRIER_UM * PREVIEW_PITCH_MAGNIFY if is_water else 0.0
        ),
        # EFFECTIVE waterline (art-box normalized y, 0 = top) — the split the
        # composed mask and the fab bake actually use, resolved from the face's
        # ``waterline`` pattern param (see _water_waterline_y). The shader had
        # frozen it as a const (FLOW_WATERLINE 0.66) while the param moved the
        # baked band, calm patch and depth shear, so the preview stopped
        # depicting its own litho mask at the range ends. Emitted
        # UNCONDITIONALLY, like switch_barrier_phase_um: it is a pure number, a
        # consumer that reads it cannot drift from it, and 0.0 on a non-water
        # face would be a division trap for anything reading it before the
        # uWaterScanN>0 gate.
        "water_waterline_y": waterline_y,
        # Art-box uv rect (see above): the shader maps face-uv → art-box uv so the
        # flow wake registers to the capybara, and sizes the crest wavelength off
        # the art-box width. (halfW, halfH, centerX, centerY); center is the plate
        # center because the centerpiece paste is plate_w//2 / plate_h//2.
        "water_art_half_uv": [float(art_half_w_uv), float(art_half_h_uv)],
        "water_art_center_uv": [0.5, 0.5],
        # Diffraction rainbow accent level. The shader lights ONLY pixels at
        # exactly this graylevel, so faces whose centerpiece has no accent zone
        # render identically — safe to emit unconditionally. Drives the spectral
        # sheen in preview + the 4.4 µm 45° grating OR-in during fab bake.
        "rainbow_level": float(RAINBOW_LEVEL),
        # The FABRICATED accent grating, published so the preview's spectral
        # sheen is computed from the same numbers the mask is written with.
        # Until this existed the shader was given only `rainbow_level` and
        # hardcoded its own 45 deg + a tuned hue ramp, so changing the fab
        # period produced a BIT-IDENTICAL preview — the on-screen rainbow could
        # not report anything true about the plate. Sourced from
        # effects.gratings (the same constants diffraction_accent_grating bakes
        # with), never re-declared here, so the two cannot drift.
        "rainbow_period_um": float(_DIFFRACTION_ACCENT_PERIOD_UM),
        "rainbow_angle_deg": float(_DIFFRACTION_ACCENT_ANGLE_DEG),
        "rainbow_duty": float(_DIFFRACTION_ACCENT_DUTY),
        # Zeroth-order share of the accent grating's return, eta_0 = duty^2 for
        # a lamellar amplitude grating. Inside the accent zone the metal does
        # NOT behave like plain gold: only this fraction stays in the specular
        # (zeroth) order, and the remainder is what becomes the diffracted
        # rainbow. The renderer scales its ordinary body/specular/env terms by
        # this in the accent zone so the spectrum REPLACES that energy instead
        # of being painted on top of a full-strength metal.
        "rainbow_zero_order": float(_DIFFRACTION_ACCENT_DUTY ** 2),
        # Interleave parameters — the SAME numbers the fab bake uses, so the
        # preview draws the construction that is actually written.
        "accent_interleave_pitch_um": float(_INTERLEAVE_BAND_PITCH_UM),
        "accent_moire_period_um": float(front_pitch),
        "accent_moire_offset_deg": float(ACCENT_MOIRE_OFFSET_DEG),
        # Fab (SVG/GDS) — true fine gratings baked as clipped rect arrays. These
        # are the user-tunable pitch (== carrier_period_um / slit_period_um): the
        # fab path (ensure_plate_svg, export_fine.build_plate_fine) reads these,
        # so a pitch change flows straight to the baked gold. The fab raster
        # coarsens the period only if the lattice budget forces it (see
        # ensure_plate_svg) — the design pitch here is exact.
        "fab_back_period_um": carrier_pitch,
        "fab_front_period_um": front_pitch,
        "fab_angle_offset_deg": CARRIER_ANGLE_OFFSET_DEG,
        # Fab centerpiece stripe period — GLASS-DERIVED so the ~5° switch
        # crossing holds on any stock (see fab_center_period_um).
        "fab_center_period_um": center_period,
    }


# In-process cache of (slug, params-hash) -> GeneratedPattern. The full
# materialize() path persists PNG/SVG/JSON to disk for the standalone
# /patterns endpoints; here we just need the shapely polygons quickly. A
# composed plate that re-uses the same central pattern across frame edits
# is the common case, so even a tiny LRU keeps the dev loop responsive.
_central_cache: dict[str, GeneratedPattern] = {}
_CENTRAL_CACHE_MAX = 24


def _generate_central_cached(cls: type, params: dict[str, Any]) -> GeneratedPattern:
    key = cls.slug + ":" + hashlib.sha1(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    hit = _central_cache.get(key)
    if hit is not None:
        return hit
    if len(_central_cache) >= _CENTRAL_CACHE_MAX:
        # Drop one arbitrary entry — exact LRU isn't worth the bookkeeping
        # at this cache size.
        _central_cache.pop(next(iter(_central_cache)))
    result = cls.generate(**params)
    _central_cache[key] = result
    return result


def _aperture(spec: PlateSpec) -> float:
    """Side length (μm) of the square central aperture inside the frame band.

    Sits inside *both* the weld margin and the frame band, so the central
    optical pattern has clean glass on every side.
    """
    band = _band_um(spec)
    aw, ah = spec.active_dims()
    return max(0.0, min(aw, ah) - 2.0 * band)


def _aperture_width_um(spec: PlateSpec) -> float:
    """Horizontal width (μm) of the clean window inside the frame band.

    ``_aperture`` is the SQUARE side (``min(W, H) − band``) that sizes the
    centerpiece motif; on a non-square face its width is height-limited. The
    WATER FULL WIDTH band must instead span the whole window edge-to-edge
    horizontally, so it uses this WIDTH-axis aperture (``active_w − 2·band``)
    — wider than ``_aperture`` on a wide face, equal on a square one. Preview
    (`_paste_centerpiece`), the fab bake (`ensure_plate_svg`), and the fine
    export (`export_fine._water_box_um`) all size the water band off this so
    the three stay in lock-step.
    """
    band = _band_um(spec)
    aw, _ah = spec.active_dims()
    return max(0.0, aw - 2.0 * band)


def _centerpiece_masks(
    slug: str, n_px: int, params: dict | None = None
) -> tuple["np.ndarray", "np.ndarray"] | None:
    """(front_art, back_art) bool silhouettes for the tilt-switch centerpiece.

    Rendered directly from the motif silhouettes (clean masks, NO baked stripe
    carrier — the shader draws the glimmer procedurally) at ``n_px`` resolution.
    Returns ``None`` for patterns that don't define a paired centerpiece, so the
    plate is frame-only. Driven by the pattern slug so the composition stays
    data-directed rather than hard-wiring one pattern into the plate compositor.

    ``params`` carries the face's ``pattern_params`` for slugs whose silhouette
    depends on them (the inscription's editable date/text); slugs with a fixed
    silhouette ignore it.
    """
    params = params or {}
    if slug in (BLANK_SLUG, PHOTO_SLUG):
        # Neither face has a silhouette centerpiece, and saying so HERE is what
        # keeps every consumer consistent: the composed preview, the fab SVG's
        # ``_center_masks`` and ``export_fine._build_zone_masks`` all read this
        # one function, so a blank face gets no art on either layer and a photo
        # face gets no ``front_art``/``back_art`` for the switch carrier to fill.
        # The photo's own geometry is a LINE SCREEN, not a silhouette — it comes
        # from ``_photo_band_stamp`` (preview) / ``photo_band_rects`` (fab).
        return None
    if slug == "colibri-flap-phase":
        # Wing-flap A/B tilt switch: the SAME hummingbird in two wing poses —
        # A = pose "up" (hover V), B = pose "down" (mid-downstroke + trailing
        # speed slivers). Body, head, beak, neck, tail and feet are
        # pixel-registered between poses (colibri._draw_colibri only swaps the
        # wing set). The slug is in SWITCH_INTERLACE_SLUGS, so this (A, B)
        # pair feeds the BARRIER INTERLACE: both poses interleaved in
        # alternating back lanes under the neutral front comb, reading as ONE
        # bird beating its wings — not two birds.
        from .patterns.motifs import colibri

        n = max(64, int(n_px))
        return (
            colibri.colibri_silhouette((1.0, 1.0), n_grid=n, pose="up"),
            colibri.colibri_silhouette((1.0, 1.0), n_grid=n, pose="down"),
        )
    if slug == "globe-duo-phase":
        # Duo-globe A/B tilt switch: front = orthographic globe centered on
        # CALIFORNIA (phase 0), back = orthographic globe centered on COLOMBIA
        # (phase π). Same disc size/position so the switch reads as one globe
        # ROTATING between the couple's homes. Clean silhouettes (no baked stripe
        # carrier) — the shader's centerpiece phase-switch supplies the glimmer,
        # exactly like the colibrí/globe branch. Centers/star markers mirror
        # patterns.artistic.globe_duo_phase so preview == the standalone pattern.
        from .patterns.artistic.globe_duo_phase import (
            CAL_LAT0,
            CAL_LON0,
            CAL_STAR_LONLAT,
            COL_LAT0,
            COL_LON0,
            COL_STAR_LONLAT,
        )
        from .patterns.geo.render import render_globe

        n = max(64, int(n_px))
        front = render_globe(
            n,
            lat0=CAL_LAT0,
            lon0=CAL_LON0,
            highlight_country=None,
            highlight_star=True,
            star_lonlat=CAL_STAR_LONLAT,
        )
        back = render_globe(
            n,
            lat0=COL_LAT0,
            lon0=COL_LON0,
            highlight_country="Colombia",
            highlight_star=True,
            star_lonlat=COL_STAR_LONLAT,
        )
        return (front, back)
    if slug == "gear-quill-switch":
        # Engineer↔historian A/B tilt switch: front = gear (phase 0), back =
        # open book + quill (phase π). Mirrors the colibrí/globe branch exactly
        # — the shader's centerpiece phase-switch reveals one motif per tilt.
        from .patterns.motifs.lab.gear_quill import (
            gear_silhouette,
            quill_book_silhouette,
        )

        n = max(64, int(n_px))
        return (
            gear_silhouette((1.0, 1.0), n_grid=n),
            quill_book_silhouette((1.0, 1.0), n_grid=n),
        )
    if slug == "food-pair-chirp":
        # Coffee cup + arepa, front-only glimmer (back empty, like monogram-jp).
        # The plate compositor fills the whole silhouette with the shared shader
        # carrier; the chirped-steam wave lives in the standalone pattern's
        # baked front polygons (fab path) — the shared preview shader renders a
        # uniform shimmer over the scene, which reads as gently steaming metal.
        import numpy as np

        from .patterns.motifs.lab.food_pair import (
            food_bodies_silhouette,
            food_steam_silhouette,
        )

        n = max(64, int(n_px))
        front = food_bodies_silhouette((1.0, 1.0), n_grid=n) | food_steam_silhouette(
            (1.0, 1.0), n_grid=n
        )
        back = np.zeros_like(front)
        return (front, back)
    if slug == "capybara-scanimation":
        # Serene capybara half-submerged on a waterline + a water SCANIMATION
        # that FLOWS on tilt, delivered right on the box plate:
        #   FRONT centerpiece art = the capybara ABOVE the waterline (the dry,
        #     still animal) — reads as a solid gold silhouette that shimmers.
        #   BACK  centerpiece art = the WATER BAND (below the waterline, spanning
        #     the aperture width) MINUS the submerged body — this marks the
        #     region where the shader's N-phase water scanimation draws
        #     travelling ripples. The foliage_moire shader runs the water-scan
        #     path (uWaterScanN>0) over the back-art region, advancing the ripple
        #     phase by the substrate parallax so the crests visibly travel as the
        #     box rocks. The fab SVG path (ensure_plate_svg) bakes the true slit
        #     barrier + interleaved ripple frames from the standalone builder.
        import numpy as np

        from .patterns.motifs.lab import capybara

        n = max(64, int(n_px))
        capy = capybara.capybara_silhouette((1.0, 1.0), n_grid=n)
        rows = (np.arange(n)[:, None] / n)  # y in 0..1, y-down (Pillow order)
        # Honour the face's tunable waterline — the mask, the fab bake and the
        # recipe_data the shader reads all resolve it here (_water_waterline_y).
        below = np.broadcast_to(rows >= _water_waterline_y(params), capy.shape)
        # Dry capybara (above waterline) on the FRONT; water band (below the
        # line, not covered by the submerged body) on the BACK. This carve is the
        # convention: the ripple band is the water AROUND the animal, and the
        # submerged body keeps the plain carrier. Both fab writers now build their
        # band with the same algebra at the same waterline —
        # ``capybara_scanimation._capybara_and_water`` returns ``below & ~capy``,
        # which is what ``ensure_plate_svg`` and ``export_fine`` consume.
        front = capy & ~below
        water_band = below & ~capy
        return (front, water_band)
    if slug == "jamon-tray":
        # Jamón ibérico on its jamonero, FRONT-only glimmer (back empty, exactly
        # like monogram-jp / food-pair-chirp). The whole silhouette carries the
        # shared shader carrier and shimmers on tilt; there is no second image to
        # switch to, so the back art is all-False. (The standalone pattern bakes
        # the real uniform front carrier for the fab path.)
        import numpy as np

        from .patterns.motifs.lab.jamon import jamon_silhouette

        n = max(64, int(n_px))
        front = jamon_silhouette((1.0, 1.0), n_grid=n)
        back = np.zeros_like(front)
        return (front, back)
    if slug == "monogram-jp":
        # Interlocked cursive J+P monogram on the FRONT face; the BACK face gets
        # no centerpiece art (all-False), so the back reads as plain carrier and
        # the monogram *shimmers* on tilt (front-only glimmer). A tilt-SWITCH
        # J<->P variant is feasible: render the two glyphs into separate masks
        # (front=J, back=P) instead of unioning — see monogram.py for the split.
        import numpy as np

        from .patterns.motifs import monogram

        n = max(64, int(n_px))
        front = monogram.monogram_silhouette((1.0, 1.0), n_grid=n)
        back = np.zeros_like(front)
        return (front, back)
    if slug == "inscription-line":
        # Hidden bottom inscription: cursive line on the FRONT face, empty BACK
        # (front-only shimmer, like the monogram). The line's text is composed
        # from the face's editable params (initials/separator/year); the drawn
        # heart flourish lands in the ♥-separator gap. Falls back to the class
        # defaults for any param the face didn't override.
        import numpy as np

        from .patterns.artistic.inscription_line import InscriptionLine
        from .patterns.motifs import inscription

        d = {**InscriptionLine.defaults(), **params}
        text = InscriptionLine.compose_text(
            d.get("initials", "J & P"),
            d.get("separator", "·"),
            int(d.get("year", 2026)),
        )
        want_heart = bool(d.get("heart", True)) or str(
            d.get("separator", "")
        ).strip() == InscriptionLine.HEART_SEP

        n = max(64, int(n_px))
        front = inscription.inscription_silhouette(
            (1.0, 1.0), n_grid=n, text=text, heart=want_heart
        )
        back = np.zeros_like(front)
        return (front, back)
    return None


# --- diffraction rainbow accent zones ---------------------------------------
# Each accent zone is a SMALL patch of the centerpiece the confirmed plan calls
# out (steam-curl tips, gear hub, monogram flourish tips). The
# zone is defined in the same normalized 0..1 art box the motif silhouette uses
# (y-DOWN, matching the Pillow motifs), as a centered ellipse / rect, then AND-ed
# with the art silhouette so only gold pixels inside the shape become accent.
# Painting these at RAINBOW_LEVEL (instead of ART_LEVEL) makes the shader add a
# spectral sheen there and the fab bake OR-in a real 4.4 µm 45° grating.
#
# Returns (side, side) bool for the FRONT centerpiece art or None if the slug
# has no designed accent (so pre-accent behaviour is byte-identical). Kept
# front-only: every plan accent lives on the front silhouette — which is exactly
# why the barrier-interlace faces get NO accent at all (see the guard below).
def _front_accent_zone(slug: str, art: "np.ndarray") -> "np.ndarray | None":
    import numpy as np

    if slug in SWITCH_INTERLACE_SLUGS:
        # Diffraction accents are incompatible with a barrier-interlace face by
        # construction. The accent is cut from the FRONT art, and on these faces
        # the front art is switch channel A (gear-quill-switch's hub ellipse is
        # a piece of the gear); the front mask does NOT move with tilt, so any
        # feature shaped from A is a static residual of image A visible at every
        # angle — so A could never fully vanish when B is gated in. That is the
        # precise residual SWITCH_INTERLACE_SLUGS and CLAUDE.md's image-switch
        # rule exist to eliminate. One guard for every consumer: the preview
        # stamp, ensure_plate_svg, and export_fine all read this helper.
        return None
    h, w = art.shape
    ys = (np.arange(h)[:, None] + 0.5) / h  # 0..1, y-down
    xs = (np.arange(w)[None, :] + 0.5) / w

    def _ellipse(cx: float, cy: float, rx: float, ry: float) -> "np.ndarray":
        return ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0

    if slug == "food-pair-chirp":
        # Steam-curl TIPS: a thin band across the top of the steam column above
        # the cup (the cup sits left; steam rises to ~y<0.30).
        zone = (xs > 0.18) & (xs < 0.42) & (ys < 0.30)
    elif slug == "gear-quill-switch":
        # Gear HUB: the central boss/bore of the centered gear.
        zone = _ellipse(0.5, 0.5, 0.11, 0.11)
    elif slug == "monogram-jp":
        # No accent. The monogram is ONE shading moiré: its front carrier
        # (witness_geom.BOX_MONO_UM, 68.23 um) beats against the back carrier
        # (BOX_CARRIER_UM, 65.5 um) at BOX_BEAT_UM over the whole silhouette.
        # (The 105.38/99 um pair this note used to quote was the gap-SCALED
        # carrier; the faces run carrier_scale_mode "fixed" since 2026-09-10.) The
        # old "flourish tips" patches (two ellipses at the swash ends) cut a
        # 4.4 um rainbow grating into the letters with their own back patch,
        # and read as two arbitrary grey strips on the lid rather than as part
        # of the letters — and the back patch eroded away under the thin
        # swashes, leaving holes in the back carrier. One effect, uniformly.
        return None
    else:
        return None
    z = zone & art
    return z if z.any() else None


# --- photo-halftone centerpiece ---------------------------------------------
# A photo face's centerpiece is a LINE SCREEN, so it has no silhouette and no
# back layer: the picture is written entirely in the height of the front bands.
# Three writers need it — the composed preview PNG, the fab SVG and the fine GDS
# — and all three go through this pair, so the picture on screen and the picture
# in the mask are the same screen at the same tone model.


def _photo_screen_inputs(spec: PlateSpec) -> dict[str, Any] | None:
    """Resolved photo params + the art-box side, or None if the face has none."""
    from .patterns.bitmap import photo as ph

    side_um = CENTERPIECE_FILL * _aperture(spec)
    if side_um <= 0:
        return None
    p = {**registry[PHOTO_SLUG].defaults(), **spec.pattern_params}
    period = float(p["line_period_um"])
    steps = ph.resolve_steps(period, int(p["tone_steps"]))
    return {
        "side_um": side_um,
        "image": str(p["image"]),
        "colour_mode": str(p["colour_mode"]),
        "fade_start": float(p["fade_start"]),
        "fade_gate": float(p["fade_gate"]),
        "line_period_um": period,
        "tone_steps": steps,
        # Sized off the ART BOX, not the plate, and identically in all three
        # writers — so the preview stamp and the two fab bakes prep the SAME
        # source array and cannot disagree about a pixel of coverage.
        "asset_px": ph.asset_px_for(side_um, period),
    }


def _photo_band_stamp(
    spec: PlateSpec, side_px: int
) -> tuple["np.ndarray", "np.ndarray"] | None:
    """``(gold, coloured)`` bool grids of ``side_px``², y-DOWN, over the art box.

    This is ``screenrects.screen_bands``' geometry evaluated ON THE PLATE RASTER
    instead of emitted as rectangles: line ``j`` occupies ``[j·P, (j+1)·P)``
    measured down from the art box's top edge, the band is CENTRED in its line,
    and its height is ``floor(coverage·steps)/steps · P`` — the screen's real
    tone ladder, not a rounding artefact. The coverage and colour fields are
    sampled through ``screenrects._sample_rows`` at the same ``(n_lines,
    n_cols)`` grid the rectangle path uses, so the preview is a rasterization of
    the fab geometry rather than a second, similar-looking screen.

    On a 29 mm face the plate raster is ~19 um, i.e. only ~2.3 px per 44 um
    line, so the preview's bands are coarse. That is accepted (the preview is
    what it is); the exact bands are ``photo_band_rects``.
    """
    import numpy as np

    from .patterns.bitmap import screenrects as sr
    from .patterns.bitmap import photo as ph

    inp = _photo_screen_inputs(spec)
    if inp is None or side_px <= 0:
        return None
    side_um = inp["side_um"]
    period = inp["line_period_um"]
    steps = inp["tone_steps"]

    cov, ids, periods = ph.photo_coverage(
        inp["image"],
        inp["fade_start"],
        inp["fade_gate"],
        steps,
        inp["asset_px"],
        colour_mode=inp["colour_mode"],
    )

    # The screen's own sampling grid (screen_bands): one line per period, one
    # column per tone cell, capped against the asset so we never sample finer
    # than the source has.
    n_lines = max(1, int(round(side_um / period)))
    want_cols = int(round(side_um / (period / steps)))
    n_cols = max(8, min(want_cols, cov.shape[1] * 4))
    tone = sr._sample_rows(cov, n_lines, n_cols)
    level = np.floor(np.clip(tone, 0.0, 1.0) * steps).astype(np.int32)
    if inp["colour_mode"] == "plain":
        pid = np.zeros(level.shape, dtype=np.int32)
    else:
        pid = sr._sample_rows(ids.astype(np.float32), n_lines, n_cols).astype(np.int32)

    # Art-box pixel centres, in um from the top-left of the box.
    cell = side_um / side_px
    ys = (np.arange(side_px) + 0.5) * cell
    xs = (np.arange(side_px) + 0.5) * cell
    li = np.minimum((ys / period).astype(np.int64), n_lines - 1)
    col_um = side_um / n_cols
    ci = np.minimum((xs / col_um).astype(np.int64), n_cols - 1)

    lvl = level[li[:, None], ci[None, :]]
    band_h = (lvl.astype(np.float64) / steps) * period
    line_centre = (li[:, None] + 0.5) * period
    gold = (np.abs(ys[:, None] - line_centre) <= band_h / 2.0) & (lvl > 0)

    # A band is "coloured" when its period id maps to a real sub-grating period;
    # id 0 (and any id absent from the table) stays plain gold — the same rule
    # ``screenrects.split_by_colour`` applies to the rectangles.
    lut = np.zeros(int(pid.max()) + 1 if pid.size else 1, dtype=np.float64)
    for k, v in periods.items():
        if 0 <= int(k) < lut.size:
            lut[int(k)] = float(v)
    coloured = gold & (lut[pid[li[:, None], ci[None, :]]] > 0.0)
    return gold, coloured


def _photo_halftone_art(spec: PlateSpec, *, defer_arrays: bool):
    """The face's screened halftone (``witness_cells.CellArt``), or None.

    One preparation of the picture — coverage, colour plan, band ladder —
    behind both photo consumers, so the exact rectangles the fab writes and the
    period map the renderer samples cannot be screened from two different
    coverage maps. ``defer_arrays`` picks which form the coloured bands come
    back in (see ``build_halftone_bands``): False concatenates every whole
    sub-grating stripe into ``art.front``; True leaves them as one
    ``art.arrays`` band-plus-period reference.
    """
    from . import witness_cells as wc
    from .patterns.bitmap import photo as ph

    inp = _photo_screen_inputs(spec)
    if inp is None:
        return None
    cov, ids, periods = ph.photo_coverage(
        inp["image"],
        inp["fade_start"],
        inp["fade_gate"],
        inp["tone_steps"],
        inp["asset_px"],
        colour_mode=inp["colour_mode"],
    )
    plan = ph.colour_plan(inp["colour_mode"])
    side = inp["side_um"]
    art, _report = wc.build_halftone_bands(
        0.0,
        0.0,
        side,
        side,
        coverage=cov,
        period_id=None if plan.mode == "plain" else ids,
        periods=periods,
        duty=plan.duty,
        line_period_um=inp["line_period_um"],
        tone_steps=inp["tone_steps"],
        defer_arrays=defer_arrays,
        polarity="metal",
    )
    return art


def photo_band_rects(spec: PlateSpec) -> "np.ndarray":
    """EXACT front-layer geometry of a photo face: ``(N, 4)`` ``[x0,x1,y0,y1]`` um.

    The line-screen bands plus, inside every coloured band, the vertical
    diffraction sub-grating (``screenrects.stripe_plan``'s whole stripes). Plate
    coords, origin at the plate centre, y UP — the same frame
    ``_helpers.raster_to_polygons`` and ``export_fine``'s rect sets use, so this
    drops into the fab SVG and the fine GDS without a transform.

    Front layer ONLY. A halftone's tone IS its band height; putting anything on
    the inner ply under it would show through the gaps and lift the shadows.
    """
    import numpy as np

    # defer_arrays=False: the SVG/GDS writers here want real rectangles. The
    # array-reference path (`stripe_plan`) is the witness plate's, where a
    # periodic sub-grating becomes one GDS array instead of 100k boxes.
    art = _photo_halftone_art(spec, defer_arrays=False)
    if art is None:
        return np.empty((0, 4), dtype=float)
    return np.asarray(art.front, dtype=float)


def photo_colour_band_periods(spec: PlateSpec) -> tuple["np.ndarray", "np.ndarray"]:
    """The COLOURED halftone bands and their sub-grating periods.

    ``((N,4) [x0,x1,y0,y1] µm bands, (N,) period µm)`` in plate coords, y UP —
    the same band rectangles ``photo_band_rects`` fills with whole stripes, but
    kept whole so the period behind each one is still a number rather than
    geometry. This is what ``period_front.png`` is painted from: at 2048 px the
    stripes themselves are far below a pixel, so the raster carries their
    COVERAGE while this carries the period that sets their diffracted colour.

    Empty on a plain (uncoloured) screen and on any non-photo face.
    """
    import numpy as np

    empty = (np.empty((0, 4), dtype=float), np.empty((0,), dtype=float))
    if spec.pattern_slug != PHOTO_SLUG:
        return empty
    # defer_arrays=True: we want the band + period reference, NOT the expanded
    # stripes (which is also why this is cheaper than photo_band_rects).
    art = _photo_halftone_art(spec, defer_arrays=True)
    if art is None or not art.arrays:
        return empty
    plan = art.arrays[0]
    rects = np.asarray(plan["rects"], dtype=float)
    periods = np.asarray(plan["period_um"], dtype=float)
    if rects.shape[0] == 0 or periods.shape[0] != rects.shape[0]:
        return empty
    return rects, periods


def _paste_centerpiece(
    spec: PlateSpec,
    front: Image.Image,
    back: Image.Image,
    pitch_um: float,
    plate_w: int,
    plate_h: int,
) -> None:
    """Paint the colibrí (front) + globe (back) centerpiece into the aperture.

    The centerpiece is a square of side ``CENTERPIECE_FILL × aperture`` centered
    on the plate. Front gets the colibrí at ART_LEVEL; back gets the globe at
    ART_LEVEL. Both are clamped to their layer's keep-out later by the rim zero.
    """
    import numpy as np

    aperture = _aperture(spec)
    if aperture <= 0:
        return
    side_um = CENTERPIECE_FILL * aperture
    side_px = max(8, int(round(side_um / pitch_um)))

    cx = plate_w // 2
    cy = plate_h // 2
    x0 = cx - side_px // 2
    y0 = cy - side_px // 2

    if spec.pattern_slug == PHOTO_SLUG:
        # LINE SCREEN, front only. Bands go in at ART_LEVEL — with
        # ``recipe_data['art_solid']`` the shader fills that level with solid
        # gold instead of the procedural switch carrier, because the band height
        # IS the tone and a second grating over it would halve every duty. The
        # bands that carry a colour period go in at RAINBOW_LEVEL instead, where
        # the shader already adds the travelling spectral sheen that stands in
        # for their sub-grating (which the fab bakes for real).
        stamp = _photo_band_stamp(spec, side_px)
        if stamp is None:
            return
        gold, coloured = stamp
        lvl = np.where(coloured, RAINBOW_LEVEL, np.where(gold, ART_LEVEL, 0)).astype(
            np.uint8
        )
        # Binary paste mask, not the graylevel — an 'L' mask alpha-blends and
        # would drag RAINBOW_LEVEL 200 down to ~157 (same trap as _paste below).
        front.paste(
            Image.fromarray(lvl, "L"),
            box=(x0, y0),
            mask=Image.fromarray((gold.astype(np.uint8) * 255), "L"),
        )
        return

    masks = _centerpiece_masks(spec.pattern_slug, side_px, spec.pattern_params)
    if masks is None:
        return
    front_art, back_art = masks

    def _resize_to_side(art: "np.ndarray") -> "np.ndarray":
        if art.shape[0] != side_px or art.shape[1] != side_px:
            m = Image.fromarray((art.astype(np.uint8) * 255), "L").resize(
                (side_px, side_px), Image.NEAREST
            )
            return np.asarray(m) > 127
        return art

    def _paste(
        dst: Image.Image,
        art: "np.ndarray",
        with_accent: bool,
        box: tuple[int, int] | None = None,
    ) -> None:
        # Resize the motif to exactly side_px if the motif grid differs (unless a
        # pre-sized mask + explicit box is supplied, e.g. the full-width water).
        a = art if box is not None else _resize_to_side(art)
        # Paint ART_LEVEL where the silhouette covers, RAINBOW_LEVEL in any
        # designed accent zone (a subset of the art, computed at THIS grid so it
        # aligns with `a`). A per-pixel graylevel stamp (not a flat paste) so the
        # two levels survive; where the stamp is 0 the frame/window underneath
        # stays intact (paste mask = the stamp).
        lvl = np.where(a, ART_LEVEL, 0).astype(np.uint8)
        if with_accent:
            accent = _front_accent_zone(spec.pattern_slug, a)
            if accent is not None:
                lvl[accent] = RAINBOW_LEVEL
        stamp = Image.fromarray(lvl, "L")
        # Binary paste mask (NOT the graylevel itself — an "L" mask alpha-blends,
        # which would drag RAINBOW_LEVEL 200 down to ~157). We want a hard stamp
        # of the graylevel wherever the art covers, frame/window intact elsewhere.
        pmask = Image.fromarray((a.astype(np.uint8) * 255), "L")
        dst.paste(stamp, box=(box if box is not None else (x0, y0)), mask=pmask)

    # Accent stamp only on the non-interlace faces: on a barrier-switch face the
    # FRONT plane must carry no image-shaped feature at all, so the preview must
    # not paint RAINBOW_LEVEL over channel-A geometry either (the shader reads
    # that level as gold + spectral sheen on the front plane). _front_accent_zone
    # refuses these slugs too — this makes the intent visible at the paste.
    _paste(
        front,
        front_art,
        with_accent=spec.pattern_slug not in SWITCH_INTERLACE_SLUGS,
    )

    if spec.pattern_slug == WATER_SCAN_SLUG:
        # WATER FULL WIDTH: the capybara body stays in the centered art square,
        # but the water band must span the whole window edge-to-edge (the full
        # WIDTH-axis aperture, `_aperture_width_um`, NOT the square `_aperture`
        # that sizes the body) — open current flanking the animal on both sides.
        # Build a full-width water band at the square's y-registration (so the
        # waterline lines up with the body) and paste it as the BACK art. The
        # shader's flow-field art box stays keyed to the square (see
        # _carrier_recipe_data art_half_*_uv), so the body wake stays registered
        # and the widened wings just continue the periodic streamline current.
        back_sq = _resize_to_side(back_art)  # square water = (below & ~capy)
        aperture_px = max(side_px, int(round(_aperture_width_um(spec) / pitch_um)))
        col_off = (aperture_px - side_px) // 2
        rows = np.arange(side_px)[:, None] / side_px  # 0..1 y-down over the square
        _wl = _water_waterline_y(spec.pattern_params)
        below = np.broadcast_to(rows >= _wl, (side_px, aperture_px))
        water_full = np.array(below, dtype=bool)  # open water everywhere below line
        # Carve the submerged body out of the central square columns (reuse the
        # square back-art, which is already `below & ~capy`).
        water_full[:, col_off : col_off + side_px] = back_sq
        wx0 = cx - aperture_px // 2
        _paste(back, water_full, with_accent=False, box=(wx0, y0))
        # TASK 1a: also mark the water band on the FRONT mask at ART_LEVEL. The
        # front (outer) plane physically carries the SLIT BARRIER comb over the
        # water; the preview shader draws that comb procedurally at exact fab size
        # but needs the water-band ZONE from its own mask level. The dry capybara
        # (above the waterline) and the water band (below) are disjoint, so the
        # shader tells them apart by the art-box waterline: body → shimmer carrier,
        # water → the 60/15/45 slit comb. Same square y-registration as the body.
        _paste(front, water_full, with_accent=False, box=(wx0, y0))
    else:
        _paste(back, back_art, with_accent=False)


def compose_plate(spec: PlateSpec) -> GeneratedPattern:
    """Run the central pattern + frame and return the union as polygons.

    NOTE: kept around for tests and the SVG export path. The runtime
    materialize_plate uses the raster compositor (`_compose_plate_raster`)
    which is far faster for the dense-frame common case.
    """
    if spec.pattern_slug not in registry:
        raise KeyError(f"Unknown pattern: {spec.pattern_slug}")

    central_cls = registry[spec.pattern_slug]
    merged_params = {**central_cls.defaults(), **spec.pattern_params}

    aperture = _aperture(spec)
    if aperture <= 0:
        raise ValueError(
            f"Aperture is non-positive ({aperture}μm) for plate "
            f"{spec.width_um}×{spec.height_um}μm with band {_band_um(spec)}μm — "
            "shrink the frame band or grow the plate."
        )

    central = _generate_central_cached(central_cls, merged_params)

    # Frame is generated on the inset active rect so vines stop at the
    # weld boundary. The polygons come out in *active-rect coords* (origin
    # at active-rect center, which happens to be the same as plate center
    # since the inset is symmetric), so they drop into the plate frame
    # without any translation.
    active_w, active_h = spec.active_dims()
    rect = RectFrame(width_um=active_w, height_um=active_h)
    frame_params = spec.frame.to_frame_params()
    scene = generate_frame(rect, frame_params)
    frame_mask = scene_to_multipolygon(scene, rect, frame_params)

    central_front = ensure_multipolygon(central.front)
    central_back = ensure_multipolygon(central.back)
    combined_front = _concat_polygons(central_front, frame_mask)

    return GeneratedPattern(
        front=combined_front,
        back=central_back,
        extent_um=(spec.width_um, spec.height_um),
        pixel_pitch_um=central.pixel_pitch_um,
        min_feature_um=central.min_feature_um,
        substrate=Substrate(
            thickness_um=spec.glass.thickness_um,
            material=spec.glass.material,
            n=spec.glass.n,
        ),
        extra={
            **central.extra,
            "central_pattern": spec.pattern_slug,
            "aperture_um": aperture,
        },
        recipe_data={**central.recipe_data, "frame_scene": scene.to_dict()},
        extra_layers=central.extra_layers,
    )


def _raster_compose_plate(spec: PlateSpec, out_dir: Path) -> dict[str, Any]:
    """Raster-space compose: foliage FRAME at the edges + tilt-switch CENTERPIECE.

    The single-channel ``L`` masks carry TWO regions each, separated by
    intensity so the shader's ``foliage_moire`` recipe can run both effects at
    once (see plate.frag runFoliageMoire):

    FRONT  = a perimeter FOLIAGE FRAME (colonize band, value ``FRAME_LEVEL``)
             wrapping the four edges, PLUS the COLIBRÍ silhouette centerpiece
             (value ``ART_LEVEL``) filling the aperture inside the band. Solid
             silhouettes only — the fine gold grating that glimmers inside them
             is drawn procedurally in the shader / baked in the SVG, so the PNG
             carries no grating (no raster aliasing rings).
    BACK   = a uniform CARRIER window (value ``FRAME_LEVEL``) spanning the whole
             EXPOSED face (``back_dims`` — foil overlap only), PLUS the GLOBE
             silhouette centerpiece (value ``ART_LEVEL``) in the same aperture,
             half-carrier-period phase-shifted from the colibrí by the shader.

    Tilt one way → the colibrí phase emerges in the centerpiece; tilt the other
    → the globe; head-on they interlace. Everywhere the front grating beats
    against the back carrier for the moiré shimmer, so the frame foliage
    glimmers too.

    Returns the *partial* manifest dict (files + extras + recipe data); the
    caller stamps the spec/labels and writes manifest.json.
    """
    central_cls = registry[spec.pattern_slug]
    merged_params = {**central_cls.defaults(), **spec.pattern_params}
    # The ParamSpec bounds check used to ride in for free on the materialize call
    # this replaced; keep it explicit so an out-of-range face param still raises
    # (the /plates and /boxes routes surface it as a 400/422) instead of sizing a
    # lattice off it.
    validate_params(spec.pattern_slug, merged_params)

    # METADATA ONLY. The centerpiece is drawn from the motif silhouettes
    # (_paste_centerpiece / _centerpiece_masks) and the frame from generate_frame,
    # so the central pattern's POLYGONS are never used on this path — the compose
    # consumes exactly four things: the pitch below, the min feature, and the
    # ``extra`` / ``recipe_data`` blocks the plate manifest carries through.
    # Materializing the whole variant for those cost 1-4 s per face (8-20 s per
    # cold six-face box). Each generator overrides the metadata accessors with the
    # same arithmetic its generate() uses; the two that MEASURE a field off their
    # own emitted raster (capybara-scanimation, bitmap-halftone) still fall back
    # to generate() there. See Pattern.metadata.
    central_meta = central_cls.metadata(**merged_params)
    central_pitch = central_meta.pixel_pitch_um

    # Plate canvas — pitch chosen so the longest plate side caps at ~1500 px.
    # We deliberately stay below the rasterize cap (16M px) by a wide margin
    # so the *six* face textures uploaded together to the WebGL preview don't
    # exhaust VRAM (3 mm cube was fine at 4000 px; 30 mm cube needs the cap).
    # The fab SVG export goes through the polygon path at full precision, so
    # this only bounds the live raster, not the actual mask.
    plate_pitch = max(central_pitch, max(spec.width_um, spec.height_um) / 1500.0)
    plate_w = max(1, int(round(spec.width_um / plate_pitch)))
    plate_h = max(1, int(round(spec.height_um / plate_pitch)))

    front = Image.new("L", (plate_w, plate_h), 0)
    back = Image.new("L", (plate_w, plate_h), 0)

    # BARE GLASS: no frame, no carrier, no centerpiece, on either layer. The
    # plate still exists — the box needs its cut dims, glass and assembly entry
    # — it just describes a rectangle of quartz. See BLANK_SLUG.
    is_blank = spec.pattern_slug == BLANK_SLUG
    # SINGLE PLY: there is no inner ply to write on, and (2026-09-10) the one
    # ply carries NO carrier: the photograph and the leaf frame on bare glass.
    # The back layer stays empty. Reference: export_fine's single-ply note.
    single_ply = bool(getattr(spec, "single_ply", False)) and not is_blank
    back_margin = spec.weld_margin_um if spec.back_margin_um is None else spec.back_margin_um

    # --- FRONT: perimeter foliage frame band ---------------------------------
    # The colonize band hugs the four edges (fill_interior=False), leaving the
    # center open for the art centerpiece. Painted at FRAME_LEVEL so the shader
    # tells frame foliage apart from the ART_LEVEL centerpiece.
    active_w, active_h = spec.active_dims()
    if is_blank:
        scene = None
    elif active_w > 0 and active_h > 0:
        rect = RectFrame(width_um=active_w, height_um=active_h)
        frame_params = spec.frame.to_frame_params()
        frame_params.fill_interior = False  # perimeter band, center stays open
        scene = generate_frame(rect, frame_params)
        # The frame PNG now carries a PER-MOTIF graylevel (angle bucket) rather
        # than a flat silhouette; paste the graylevels straight through so the
        # shader can decode each motif's fringe direction. The mask (frame_png
        # itself, nonzero) restricts the paste to the foliage pixels so the
        # open center + weld rim stay untouched.
        frame_png = render_scene_to_image(
            scene, rect, frame_params, plate_pitch, level_fn=_frame_level_for
        )
        active_w_px = max(1, int(round(active_w / plate_pitch)))
        active_h_px = max(1, int(round(active_h / plate_pitch)))
        weld_x_px = (plate_w - active_w_px) // 2
        weld_y_px = (plate_h - active_h_px) // 2
        # Binarize the paste mask. Using frame_png itself as an 'L' mask
        # alpha-blends the graylevels onto the black plate (result = L*L/255),
        # squaring every angle-bucket level so orchid/coffee fall below the
        # shader's FRAME_MIN and the six fringe directions collapse to two.
        # A 0/255 mask keeps each foliage pixel's graylevel intact — same trap
        # documented for the centerpiece _paste above.
        bin_mask = frame_png.point(lambda v: 255 if v else 0)
        front.paste(frame_png, box=(weld_x_px, weld_y_px), mask=bin_mask)
    else:
        scene = None  # weld swallowed the active area — degenerate but legal

    # --- BACK: solid uniform carrier window across the whole exposed face ----
    # Skipped on a single-ply face (the carrier is already on the front, above)
    # and on a blank one (no gold anywhere).
    back_w, back_h = spec.back_dims()
    if not is_blank and not single_ply and back_w > 0 and back_h > 0:
        back_w_px = max(1, int(round(back_w / plate_pitch)))
        back_h_px = max(1, int(round(back_h / plate_pitch)))
        bx0 = (plate_w - back_w_px) // 2
        by0 = (plate_h - back_h_px) // 2
        ImageDraw.Draw(back).rectangle(
            (bx0, by0, bx0 + back_w_px - 1, by0 + back_h_px - 1), fill=FRAME_LEVEL
        )

    # --- CENTERPIECE: colibrí (front) + globe (back) tilt-switch art ---------
    # Fills the aperture inside the frame band, generously, with a breathing gap
    # to the frame. The two silhouettes are painted at ART_LEVEL so the shader
    # runs the phase-shift switch inside the centerpiece while the frame band
    # around it runs the foliage-moiré shimmer.
    if not is_blank:
        _paste_centerpiece(spec, front, back, plate_pitch, plate_w, plate_h)

    # Belt-and-suspenders: clear each layer's OWN keep-out rim. The front art
    # must stay inside the weld margin (foil overlap + safety); the back
    # carrier only has to clear the foil overlap (its wider window), so each
    # gets its own border zeroed. Rounding overshoot can't leak under the foil.
    def _zero_rim(img: Image.Image, margin_um: float) -> None:
        if margin_um <= 0:
            return
        m = max(1, int(round(margin_um / plate_pitch)))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, plate_w, m), fill=0)
        d.rectangle((0, plate_h - m, plate_w, plate_h), fill=0)
        d.rectangle((0, 0, m, plate_h), fill=0)
        d.rectangle((plate_w - m, 0, plate_w, plate_h), fill=0)

    # On a SINGLE-PLY face the carrier is the back layer moved onto the outer
    # ply, so its keep-out is the BACK window (foil overlap only), not the front
    # weld margin — the same split the single-ply block in
    # ``export_fine.build_plate_fine`` makes. The frame band is bounded by the
    # active rect and the centerpiece by the art box, so nothing else can
    # reach into the ring between the two rims; taking
    # the smaller of them keeps the frame's own keep-out honest whichever way
    # round the two margins happen to fall for a given foil/glass pair.
    _zero_rim(
        front,
        min(spec.weld_margin_um, back_margin) if single_ply else spec.weld_margin_um,
    )
    _zero_rim(back, back_margin)

    save_png_atomic(front, out_dir / "front.png")
    save_png_atomic(back, out_dir / "back.png")
    # One chip per litho metal. Written at compose time (three tiny pastes)
    # rather than keyed into the plate hash, so switching metal in the UI is an
    # instant swap instead of a full six-plate recompose — the METAL never
    # changes the mask geometry, only how the chip is tinted.
    thumb = make_thumbnail(front, back, size=320)
    save_png_atomic(thumb, out_dir / "thumbnail.png")
    for _metal in THUMBNAIL_PALETTES:
        if _metal == DEFAULT_THUMBNAIL_METAL:
            continue
        save_png_atomic(
            make_thumbnail(front, back, size=320, metal=_metal),
            out_dir / f"thumbnail_{_metal}.png",
        )

    # The frame scene (potentially MBs of segment dicts) goes to a sidecar,
    # NOT into manifest.json: embedding it made every plate manifest 0.4-2.3
    # MB and dominated both the warm box regen (6 × json decode) and the cold
    # compose's manifest encode. The frontend never reads it and export strips
    # it from the fab bundle, but it is NOT debug-only: the fab SVG bake and
    # export_fine's zone masks load it back (``frame_scene_for_plate``) instead
    # of regrowing the band, which is what keeps all three renders of a face
    # pinned to one scene. Keyed to PLATE_COMPOSE_VERSION by that reader.
    frame_scene = scene.to_dict() if scene is not None else {"segments": [], "flowers": [], "leaves": [], "max_t": 0.0}
    write_json_atomic(out_dir / "scene.json", frame_scene, indent=None)

    # The standalone variant's recipe_data, reproduced without materializing it:
    # ``_materialize_locked`` stamps one ``<name>_png`` URL per extra layer after
    # the generator's own keys, and the plate manifest carries the merged block
    # through. Same variant id and same URL builder as that path, so the strings
    # are identical — the layer PNGs themselves only exist once someone actually
    # materializes the standalone variant (nothing on the plate path reads them;
    # the composed plate binds foliage_moire against its own front/back masks).
    central_variant = variant_key(merged_params)
    central_recipe_data = {
        **central_meta.recipe_data,
        **{
            f"{name}_png": extra_layer_url(spec.pattern_slug, central_variant, name)
            for name in central_meta.extra_layer_names
        },
    }

    return {
        "pixel_pitch_um": plate_pitch,
        "min_feature_um": central_meta.min_feature_um,
        "central_extra": central_meta.extra,
        "central_recipe_data": central_recipe_data,
    }


# --- LITERAL rasters: the fabricated chrome, sampled as coverage -------------
#
# ``front.png`` / ``back.png`` are SHADER masks: graylevel ZONE CODES
# (FRAME_LEVEL, ART_LEVEL, RAINBOW_LEVEL, the FRAME_BUCKET angle ladder) that
# tell plate.frag which procedural effect to run where, at a pitch far too
# coarse to carry a grating without aliasing rings. The literal rasters are the
# opposite kind of image: no codes and no zones, just the actual DRC-healed
# litho polygons ``export_fine.build_plate_fine`` hands the GDS writer, sampled
# as area coverage. 255 = chrome/metal present, 0 = bare glass.
#
# The renderer samples these on the outer and inner pattern planes instead of
# synthesising gratings, which is what makes the preview a picture of the plate
# rather than an impression of it — and it is the same geometry the mask writer
# prints, so a moiré on screen is a moiré the fab will make.
#
# Same extent, orientation and centre as front.png/back.png (whole plate
# INCLUDING the weld margin, unmirrored, plate µm with the origin at the plate
# centre and y UP mapped to row 0 = top) so the two overlay pixel for pixel
# after a uniform scale. An EMPTY layer is a valid literal raster and is
# written as all zeros — a blank face, or the back of a single-ply one — never
# left as a missing file.
LITERAL_RASTER_PX = 2048
# ``period_front.png`` packs the sub-grating period as period_um × 25, so the
# whole diffractive family fits an 8-bit channel (0-10.2 µm at 0.04 µm steps)
# and 0 keeps its meaning: no sub-grating here.
LITERAL_PERIOD_SCALE = 25.0


def _literal_raster_dims(spec: PlateSpec) -> tuple[int, int]:
    """``(w_px, h_px)`` for the literal rasters: plate aspect, 2048 on the long side."""
    w_um = max(1e-6, float(spec.width_um))
    h_um = max(1e-6, float(spec.height_um))
    if w_um >= h_um:
        return LITERAL_RASTER_PX, max(1, int(round(LITERAL_RASTER_PX * h_um / w_um)))
    return max(1, int(round(LITERAL_RASTER_PX * w_um / h_um))), LITERAL_RASTER_PX


def _literal_layer_raster(
    polys: list["np.ndarray"], spec: PlateSpec, w_px: int, h_px: int
) -> Image.Image:
    """One layer's plate-frame polygon rings → an ``L`` coverage raster.

    ``polys`` are ``build_plate_fine``'s healed EXTERIOR rings (holes already
    dropped by ``drc_clean_region`` — a fill-only gold mask), so filling every
    ring is the whole rasterization. Plate µm (origin centre, y up) map to
    pixels exactly as ``_raster_compose_plate`` maps them: x + W/2 scaled
    across the width, H/2 - y scaled down the height.

    The coverage is computed ANALYTICALLY (``literal_raster.layer_coverage``),
    not by supersampling a fill. Every feature here is far below a texel — a
    27.5 mm face is ~13.4 µm/texel while the colour sub-gratings are 2.5 µm
    lines — so a boundary-inclusive fill biases the whole raster bright
    (measured +6% on a 50%-duty carrier, and the 5 µm colour bands read 0.71
    with 46% of their texels pinned at 255). See that module for why no
    supersample/shrink combination fixes it and how the exact accumulation
    works. 255 = chrome over the whole texel, 0 = bare glass.
    """
    from .literal_raster import layer_coverage

    import numpy as np

    cov = layer_coverage(polys or [], spec.width_um, spec.height_um, w_px, h_px)
    return Image.fromarray(np.round(cov * 255.0).astype(np.uint8), mode="L")


def _literal_period_raster(spec: PlateSpec, w_px: int, h_px: int) -> Image.Image | None:
    """``period_front.png``: the sub-grating PERIOD field, or None if there is none.

    Only a photo-halftone face with a colour plan has one. Its diffraction
    sub-grating is ~5 µm lines on a few-µm period — three orders of magnitude
    below a 2048 px plate raster — so ``literal_front`` can only carry the
    stripes' COVERAGE (a grey inside each coloured band). The colour they
    diffract comes from their PERIOD, which is what this map publishes:
    ``period_um × LITERAL_PERIOD_SCALE`` inside every band that carries one,
    0 everywhere else.

    Bands are painted at final resolution and outset to whole pixels: a band is
    thinner than a pixel at this scale, and a label field that rounds its thin
    bands away publishes "no sub-grating here" — a worse lie than a band one
    pixel too wide.

    A texel therefore lands inside several stacked bands at once, so the one
    that WINS it is the one whose exact overlap area is largest — not whichever
    was painted last. Bands with no sub-grating (period 0) never enter the
    contest, so the map stays 0 wherever there is nothing to diffract.
    """
    import numpy as np

    from .literal_raster import rect_texel_overlaps

    rects, periods = photo_colour_band_periods(spec)
    if rects.shape[0] == 0:
        return None
    values = np.minimum(
        255, np.round(np.asarray(periods, dtype=float) * LITERAL_PERIOD_SCALE)
    ).astype(np.int64)
    has = values > 0
    if not has.any():
        return None
    rects, values = rects[has], values[has]

    w_um = max(1e-6, float(spec.width_um))
    h_um = max(1e-6, float(spec.height_um))
    sx, sy = w_px / w_um, h_px / h_um
    hx, hy = 0.5 * w_um, 0.5 * h_um
    flat, band, area = rect_texel_overlaps(
        (rects[:, 0] + hx) * sx,
        (rects[:, 1] + hx) * sx,
        (hy - rects[:, 3]) * sy,
        (hy - rects[:, 2]) * sy,
        w_px,
        h_px,
    )
    if flat.size == 0:
        return None
    # Largest overlap wins each texel: sort by (texel, area) and keep the last
    # entry of every texel's run.
    order = np.lexsort((area, flat))
    flat, band = flat[order], band[order]
    last = np.nonzero(np.diff(flat, append=flat[-1] + 1))[0]
    out = np.zeros(h_px * w_px, dtype=np.uint8)
    out[flat[last]] = values[band[last]].astype(np.uint8)
    if not out.any():
        return None
    return Image.fromarray(out.reshape(h_px, w_px), mode="L")


def _single_ply_leaf_period_raster(
    spec: PlateSpec, front: Image.Image, period: Image.Image | None
) -> Image.Image | None:
    """Add the leaf grating's period to the period map of a single-ply face.

    On one ply the garland is fine diffractive gratings
    (``single_ply_leaf_period_um`` — the same value the manifest advertises)
    that a 2048 px raster cannot resolve — ``literal_front`` carries their 50%
    coverage. Every front texel OUTSIDE the centerpiece art box that carries
    metal is a leaf (nothing else is written there on a single ply), so it gets
    the leaf period; texels the photo bands already claimed keep theirs. The
    per-leaf ORIENTATION is not in this map — the preview's sheen is the same
    for every leaf, which is the one approximation the flag in the shader names.
    """
    import numpy as np

    cov = np.asarray(front.convert("L"), dtype=np.uint8)
    h_px, w_px = cov.shape
    out = np.zeros((h_px, w_px), dtype=np.uint8) if period is None else np.asarray(period.convert("L"), dtype=np.uint8).copy()
    side = CENTERPIECE_FILL * _aperture(spec)
    sx, sy = w_px / max(1e-6, spec.width_um), h_px / max(1e-6, spec.height_um)
    x0 = int(np.floor((spec.width_um / 2 - side / 2) * sx)); x1 = int(np.ceil((spec.width_um / 2 + side / 2) * sx))
    y0 = int(np.floor((spec.height_um / 2 - side / 2) * sy)); y1 = int(np.ceil((spec.height_um / 2 + side / 2) * sy))
    outside = np.ones((h_px, w_px), dtype=bool)
    outside[max(0, y0):min(h_px, y1), max(0, x0):min(w_px, x1)] = False
    leaf = outside & (cov > 5) & (out == 0)
    value = int(min(255, round(single_ply_leaf_period_um(spec) * LITERAL_PERIOD_SCALE)))
    out[leaf] = value
    if not out.any():
        return None
    return Image.fromarray(out, mode="L")


def _write_literal_rasters(spec: PlateSpec, out_dir: Path, pid: str) -> dict[str, str]:
    """Publish the plate's literal rasters; return the ``files`` entries for them.

    ``literal_front`` / ``literal_back`` always, ``period_front`` only when the
    face actually has a sub-grating period field (an all-zero period map is
    noise — the absent key IS "no sub-gratings on this face"). Any period map
    left by an earlier compose of this slot is removed, so the manifest and the
    directory cannot disagree.

    ONE ``build_plate_fine`` per face: it is the expensive call on this path
    (seconds to ~40 s), and it already reuses this plate's frame-scene sidecar
    rather than regrowing the band — see ``frame_scene_for_plate`` and the
    fresh-compose window ``_materialize_plate_locked`` opens around it.
    """
    from .export_fine import build_plate_fine

    w_px, h_px = _literal_raster_dims(spec)
    # ``face`` is informational on this path (it labels PlateFine.face for the
    # wafer report's per-face stats and picks nothing geometric); a plate
    # materialized on its own has no box face id, so it is labelled by what it
    # actually is.
    t_fine = time.perf_counter()
    fine = build_plate_fine(spec, spec.label or spec.pattern_slug)
    fine_ms = int((time.perf_counter() - t_fine) * 1000)

    t_raster = time.perf_counter()
    files: dict[str, str] = {}
    front_img: Image.Image | None = None
    for name, layer in (("literal_front", fine.front_polys), ("literal_back", fine.back_polys)):
        img = _literal_layer_raster(layer, spec, w_px, h_px)
        if name == "literal_front":
            front_img = img
        save_png_atomic(img, out_dir / f"{name}.png")
        files[name] = f"/data/plates/{pid}/{name}.png"

    period = _literal_period_raster(spec, w_px, h_px)
    if getattr(spec, "single_ply", False) and front_img is not None:
        period = _single_ply_leaf_period_raster(spec, front_img, period)
    period_path = out_dir / "period_front.png"
    if period is None:
        period_path.unlink(missing_ok=True)
    else:
        save_png_atomic(period, period_path)
        files["period_front"] = f"/data/plates/{pid}/period_front.png"
    _log.info(
        "literal_rasters id=%s slug=%s %dx%d fine=%dms raster=%dms polys=%d/%d period=%s",
        pid,
        spec.pattern_slug,
        w_px,
        h_px,
        fine_ms,
        int((time.perf_counter() - t_raster) * 1000),
        len(fine.front_polys),
        len(fine.back_polys),
        period is not None,
    )
    return files


# Bump when the composed-plate output changes under an unchanged spec hash:
# ``_raster_compose_plate``, ``_paste_centerpiece``, ``_centerpiece_masks``, the
# mask level palette (FRAME_LEVEL / ART_LEVEL / FRAME_BUCKET* / RAINBOW_LEVEL),
# or ``_carrier_recipe_data`` (including the render_recipe the manifest forces
# and every shader knob it emits). ``plate_hash`` covers spec fields only, so a
# cached manifest+PNG pair is otherwise served forever after a code or constant
# change — the same trap PLATE_SVG_VERSION already closes on the fab SVGs. See
# CLAUDE.md; a mismatch on the hit path is treated as a miss.
# v2: first versioned compose — wave 1 changed compose geometry and recipe_data
#     (front water band, barrier registration, litho floor) with no key to
#     invalidate the caches it had already written.
# v3: recipe_data publishes the solved barrier registration
#     (switch_interlace_period_um / switch_barrier_phase_um) for the preview
#     shader; the mask PNGs are unchanged (the barrier lives in the fab bake, see
#     PLATE_SVG_VERSION v6).
# v4: recipe_data publishes the EFFECTIVE capybara waterline
#     (``water_waterline_y``) and the composed centerpiece mask honours the
#     ``waterline`` pattern param instead of the module constant, so the shader's
#     water/body split follows the mask it is drawing. Mask PNGs move only on a
#     face that actually sets the param (default-param output is unchanged).
# v5: recipe_data drops the dead ``water_phase_pitch_preview_um`` key (and its
#     WATER_PHASE_PITCH_PREVIEW_UM constant) — no shader line ever read it and the
#     uniform is gone. recipe_data is part of the cached manifest, so a warm cache
#     would otherwise keep serving the key forever; a consumer added later against
#     a stale manifest would find it on some faces and not others. Mask PNGs are
#     unchanged by this bump.
# v6: the capybara water band drops the submerged body in the PATTERN too
#     (``_capybara_and_water``), so capybara-scanimation's generated masks — and
#     with them the ``min_feature_um`` / ``central_extra`` (``min_back_gold_um``,
#     ``min_front_gold_um``, …) this manifest copies out of ``central_cls.metadata``
#     — move. The composed PNGs do NOT: ``_paste_centerpiece`` /
#     ``_centerpiece_masks`` already carved the band and the plate pitch is keyed to
#     the pattern's unchanged ``pixel_pitch_um``. Without the bump a warm plate slot
#     keeps advertising the pre-carve measured minimums next to a re-baked SVG.
# v13: FrameSpec grows ``motif_scale`` — a motif-only size dial threaded into
#     the wreath grower (leaf/bloom/understory/corner-sprig sizes AND their
#     station spacing along the vine; band width and vine gauge untouched). The
#     default 1.0 is bit-identical to v12's geometry, but the field is part of
#     the frame recipe the compose path bakes, so warm slots must re-derive
#     rather than serve a manifest that predates the knob.
# v14: the PRODUCTION box lands — three new compose behaviours and three new
#     recipe_data keys. BLANK faces (``BLANK_SLUG``) emit nothing on either
#     layer; SINGLE-PLY faces (``PlateSpec.single_ply``) move the uniform
#     carrier onto the FRONT mask over the back window minus the art box and
#     leave the back empty; PHOTO faces (``PHOTO_SLUG``) stamp line-screen bands
#     at ART_LEVEL / RAINBOW_LEVEL instead of a silhouette. ``_carrier_recipe_data``
#     grows ``blank`` / ``art_solid`` / ``single_ply`` unconditionally, so even a
#     face whose PNGs are unchanged must re-derive rather than serve a manifest
#     that predates the keys.
# v15: RENDER IT LITERALLY. Every composed face publishes coverage rasters of
#     its actual DRC-healed chrome (``literal_front``/``literal_back``, plus
#     ``period_front`` where a halftone carries colour sub-gratings) and stamps
#     ``recipe_data['literal']``. The manifest SHAPE changed — new ``files``
#     keys and a new recipe_data key — and the payload PNGs do not exist beside
#     a v14 manifest at all, so a warm slot must re-derive rather than serve a
#     manifest whose renderer contract it cannot satisfy.
# v16: the literal rasters become EXACT coverage. v15 filled the rings with a 2×
#     supersampled PIL polygon draw, whose boundary-inclusive, phase-QUANTISED
#     fill fattened every line by a whole sample: a 50%-duty carrier rastered at
#     0.532 and the 5 µm colour bands at 0.71 with 46% of their texels pinned to
#     255, i.e. the preview showed the colour zones as near-solid gold and the
#     whole plate too heavy. ``literal_raster.layer_coverage`` now accumulates
#     the analytic area instead (see that module). ``period_front`` picks each
#     texel's band by overlap AREA rather than by paint order. Every
#     ``literal_*``/``period_front`` PNG on disk is therefore wrong under a v15
#     manifest and must be re-derived.
# v17-v22: the PLATE FOR WRITING (2026-09-10) — one entry, because the six bumps
#     were one design landing in stages and no cache survived any of them.
#     (a) literal rasters exact, continued: the literal bake is the ACTUAL healed
#     chrome for every face, so a composed face's PNGs and its ``files`` entries
#     both move. (b) The lid's monogram carries NO diffraction accent — one
#     shading moiré over the whole silhouette instead of two flourish-tip
#     patches — so ``monogram-jp`` masks change on both layers. (c) SINGLE-PLY
#     faces drop the carrier entirely (``photo.CARRIER_COV`` = 0): the front mask
#     is the picture and the leaf garland on bare glass, the back stays empty,
#     and the picture's edge fade now dissolves to glass rather than to a 50%
#     field. (d) The leaf fill knob (``SINGLE_PLY_LEAF_FILL`` = "hue") and the
#     period ladder arrive with the recipe key ``single_ply_leaf_period_um`` and
#     a ``period_front`` map over the leaves. (e) The box's own dials move under
#     every face: the art rim starts at the inner ply's window
#     (``assembly.bonded_art_keepout_um``), the band is a fixed 2.4 mm, and the
#     faces run ``carrier_scale_mode`` "fixed" at the eye-sized 65.5 µm carrier —
#     all of which land in the frame geometry and in ``_carrier_recipe_data``.
# v23: ``single_ply_leaf_period_um`` (recipe key AND ``period_front``) comes from
#     ONE helper, ``single_ply_leaf_period_um(spec)``, and under the shipping
#     "hue" fill that is the LADDER'S MEAN. v17-v22 advertised the 10 µm "lines"
#     constant in the manifest while writing the 4.15-6.02 µm ladder, so the
#     renderer drew a sheen at twice the period of the gold in front of it. The
#     PNG masks are unchanged; recipe_data is not, and it is cached.
PLATE_COMPOSE_VERSION = 24


# Plate ids whose ``scene.json`` was written by the compose CURRENTLY running on
# this thread. The literal-raster bake runs INSIDE ``_materialize_plate_locked``,
# before the manifest is published — so ``get_plate`` still reads the old
# manifest (or none at all) and the compose_version gate below would reject a
# sidecar this very compose just wrote, regrowing the band a second time in the
# same call. Thread-local rather than module-global: the gate exists to stop a
# sidecar from an OLDER compose leaking in, and only the thread inside the
# compose knows the file is current. Set/cleared in _materialize_plate_locked.
_fresh_scene = threading.local()


def _scene_sidecar_is_fresh(plate_id: str) -> bool:
    return plate_id in getattr(_fresh_scene, "ids", ())


def frame_scene_for_plate(
    plate_dir: Path,
    manifest: dict[str, Any] | None,
    rect: RectFrame,
    frame_params: FrameParams,
) -> Scene:
    """The composed plate's frame scene: ``scene.json`` sidecar first, regrow on miss.

    ``_raster_compose_plate`` writes the sidecar from exactly this
    ``(rect, frame_params)`` pair, and the grower is seed-deterministic, so
    reusing it is not merely equivalent — it is the SAME scene the cached
    preview PNG was painted from, which is what the "fab SVG = preview PNG"
    contract wants. Regrowing instead costs 1-3 s per face and would silently
    diverge from a cached PNG whenever the grower code changed without a
    version bump.

    Gated on the manifest's ``compose_version`` so a sidecar left by an older
    compose cannot leak into a plate that the current compose would rebuild —
    or, mid-compose, on ``_scene_sidecar_is_fresh``, which is the same
    guarantee before there is a manifest to read it from.
    """
    fresh = _scene_sidecar_is_fresh(plate_dir.name)
    if fresh or (
        manifest is not None and manifest.get("compose_version") == PLATE_COMPOSE_VERSION
    ):
        data = read_json_cache(plate_dir / "scene.json")
        if data is not None and "segments" in data:
            return Scene.from_dict(data)
    return generate_frame(rect, frame_params)


def materialize_plate(spec: PlateSpec, force: bool = False) -> dict[str, Any]:
    """Compose the plate, cache PNG/SVG/manifest under ``data/plates/<hash>/``.

    Serialized per plate hash (see ``service.cache_lock``): a box regen fans six
    faces out sequentially, and two overlapping regens of the same face would
    otherwise both compose into the same directory. A manifest that is
    unreadable or carries a stale ``compose_version`` counts as a miss.
    """
    PLATES_ROOT.mkdir(parents=True, exist_ok=True)
    pid = plate_hash(spec)
    with cache_lock(f"plate:{pid}"):
        return _materialize_plate_locked(spec, pid, force)


def _materialize_plate_locked(spec: PlateSpec, pid: str, force: bool) -> dict[str, Any]:
    out = PLATES_ROOT / pid
    manifest_path = out / "manifest.json"
    if not force:
        cached = read_json_cache(manifest_path)
        if cached is not None and cached.get("compose_version") == PLATE_COMPOSE_VERSION:
            _log.info("materialize_plate cache_hit id=%s slug=%s", pid, spec.pattern_slug)
            return cached
        if manifest_path.exists():
            _log.info(
                "materialize_plate regenerate id=%s slug=%s reason=%s",
                pid,
                spec.pattern_slug,
                "unreadable_manifest"
                if cached is None
                else f"compose_version {cached.get('compose_version')!r} != {PLATE_COMPOSE_VERSION}",
            )

    # Everything below is the heavy path, so it runs under the process-wide
    # compute gate (CLAUDE.md machine constraint; see service.heavy_compute_gate).
    # Taken HERE rather than around the whole function so the cache-hit return
    # above — six of them on a warm box regen — never queues behind an unrelated
    # cold compose.
    with heavy_compute_gate:
        t0 = time.perf_counter()
        out.mkdir(parents=True, exist_ok=True)
        raster_result = _raster_compose_plate(spec, out)
        pitch = raster_result["pixel_pitch_um"]

        # LITERAL rasters — the fabricated chrome the renderer actually samples
        # (see LITERAL_RASTER_PX). Built here, next to the shader masks, so a
        # plate slot is never half a contract: the manifest published at the
        # bottom of this block is the only thing that makes any of it visible,
        # and it advertises both. The fresh-scene window lets the fine bake read
        # the scene.json _raster_compose_plate just wrote instead of regrowing
        # the foliage band a second time in the same call.
        ids = getattr(_fresh_scene, "ids", None)
        if ids is None:
            ids = _fresh_scene.ids = set()
        ids.add(pid)
        try:
            literal_files = _write_literal_rasters(spec, out, pid)
        finally:
            ids.discard(pid)

        # SVG is built on demand by the /export endpoint (see ensure_plate_svg)
        # — the polygon path is ~40× slower than raster and the interactive UI
        # never needs it. Manifest carries empty svg paths until requested.
        # Any pair already in this slot was baked from the geometry we just
        # replaced (force, or a PLATE_COMPOSE_VERSION bump), and it still carries
        # the current PLATE_SVG_VERSION marker — so ensure_plate_svg would happily
        # serve it and break the fab-SVG = preview-PNG contract. Drop it.
        for stale_svg in (out / "front.svg", out / "back.svg"):
            stale_svg.unlink(missing_ok=True)

        central_cls = registry[spec.pattern_slug]
        manifest = {
            "kind": "plate",
            "id": pid,
            # Compose-code marker; a mismatch on the hit path is a miss (plate_hash
            # covers spec fields only). See PLATE_COMPOSE_VERSION.
            "compose_version": PLATE_COMPOSE_VERSION,
            "spec": spec.to_dict(),
            "name": spec.label or central_cls.name,
            "description": central_cls.description,
            "tags": central_cls.tags,
            "substrate": {
                "thickness_um": spec.glass.thickness_um,
                "material": spec.glass.material,
                "n": spec.glass.n,
            },
            "extent_um": [spec.width_um, spec.height_um],
            "pixel_pitch_um": pitch,
            "min_feature_um": raster_result["min_feature_um"],
            "extra": {
                **raster_result["central_extra"],
                "central_pattern": spec.pattern_slug,
                "aperture_um": _aperture(spec),
            },
            # The plate is a box-first moiré carrier: front foliage grating vs
            # back uniform grating, beat procedurally in the shader — NOT the
            # central pattern's own recipe. Force the foliage_moire recipe and
            # carry the grating knobs regardless of which central pattern seeded
            # the metadata.
            "render_recipe": "foliage_moire",
            # frame_scene deliberately NOT embedded — see _raster_compose_plate
            # (scene.json sidecar keeps the manifest ~20 KB instead of ~MBs).
            "recipe_data": {
                **raster_result["central_recipe_data"],
                **_carrier_recipe_data(spec),
            },
            "files": {
                "front_png": f"/data/plates/{pid}/front.png",
                "back_png": f"/data/plates/{pid}/back.png",
                # The literal chrome rasters the renderer samples on the two
                # pattern planes: literal_front / literal_back always (an empty
                # layer is an all-zero raster, not a missing key), period_front
                # only where a halftone carries colour sub-gratings. Same URL
                # form as front_png — the files are already on disk above.
                **literal_files,
                "front_svg": "",
                "back_svg": "",
                "thumbnail": f"/data/plates/{pid}/thumbnail.png",
                # Per-metal chips; the renderer picks by spec.metal and falls
                # back to `thumbnail` for manifests written before this existed.
                "thumbnails": {
                    _m: (
                        f"/data/plates/{pid}/thumbnail.png"
                        if _m == DEFAULT_THUMBNAIL_METAL
                        else f"/data/plates/{pid}/thumbnail_{_m}.png"
                    )
                    for _m in THUMBNAIL_PALETTES
                },
            },
        }
        # Published last and by rename — the PNGs above are already in place, so a
        # readable manifest implies a complete slot.
        write_json_atomic(manifest_path, manifest)
        dt_ms = int((time.perf_counter() - t0) * 1000)
        _log.info(
            "materialize_plate done id=%s slug=%s %dms pitch=%.3f",
            pid,
            spec.pattern_slug,
            dt_ms,
            pitch,
        )
        return manifest


def _mask_rim(grid: "np.ndarray", margin_um: float, pitch_um: float) -> None:
    """Zero the keep-out rim of a boolean grid in place (fab clip)."""
    if margin_um <= 0:
        return
    m = max(1, int(round(margin_um / pitch_um)))
    grid[:m, :] = False
    grid[-m:, :] = False
    grid[:, :m] = False
    grid[:, -m:] = False


def _back_window_grid(
    spec: PlateSpec, fw: int, fh: int, pitch_um: float
) -> "np.ndarray":
    """The uniform-carrier window as a bool grid, rimmed at the BACK keep-out.

    The window spans the whole EXPOSED face (``back_dims`` — foil overlap only,
    wider than the front weld margin). Shared by the two-ply BACK bake and the
    single-ply FRONT bake so the carrier occupies the identical rectangle
    whichever ply it ends up on.
    """
    import numpy as np

    win = np.zeros((fh, fw), dtype=bool)
    back_w, back_h = spec.back_dims()
    if back_w <= 0 or back_h <= 0:
        return win
    bw = max(1, int(round(back_w / pitch_um)))
    bh = max(1, int(round(back_h / pitch_um)))
    bx = (fw - bw) // 2
    by = (fh - bh) // 2
    win[by : by + bh, bx : bx + bw] = True
    back_margin = spec.weld_margin_um if spec.back_margin_um is None else spec.back_margin_um
    _mask_rim(win, back_margin, pitch_um)
    return win


def _photo_multipolygon(spec: PlateSpec) -> MultiPolygon:
    """A photo face's exact front geometry as polygons (bands + sub-gratings)."""
    from .patterns.bitmap.photo import rects_to_multipolygon

    return rects_to_multipolygon(photo_band_rects(spec))


def _strip_svg_body(full_svg: str) -> str:
    """Return the inner body of a drawsvg <svg>…</svg> document (no wrapper)."""
    open_idx = full_svg.find("<svg")
    if open_idx == -1:
        return full_svg
    open_end = full_svg.find(">", open_idx)
    close_idx = full_svg.rfind("</svg>")
    if open_end == -1 or close_idx == -1:
        return full_svg
    return full_svg[open_end + 1 : close_idx]


def _grating_grid(
    w_px: int,
    h_px: int,
    pitch_um: float,
    period_um: float,
    duty: float,
    angle_deg: float,
    phase: float = 0.0,
) -> "np.ndarray":
    """Boolean grid (1 = gold line) of a rotated linear grating.

    Pure numpy — a projected coordinate compared against the duty window.
    No GEOS anywhere. ``(h_px, w_px)`` row-major to match ``raster_to_polygons``.
    ``phase`` shifts the grating by that fraction of a period (0.5 = half-period
    offset, the phase-π interlace used by the centerpiece globe vs colibrí).
    """
    import numpy as np

    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Physical μm coords per pixel, origin at grid center (matches raster_to_polygons).
    xs = (np.arange(w_px) - (w_px - 1) / 2.0) * pitch_um
    ys = ((h_px - 1) / 2.0 - np.arange(h_px)) * pitch_um
    X, Y = np.meshgrid(xs, ys)
    coord = (X * ca + Y * sa) / period_um + phase
    frac = coord - np.floor(coord)
    return frac < duty


# Cells per barrier period, floor and quantum: the interlace channel is p/2 and
# each half of the open slit is p/4, so only a MULTIPLE OF FOUR cells puts the
# A|B boundary and both slit edges on cell boundaries. Four is the coarsest
# lattice that still has a switch (2-cell channel, 2-cell slit).
BARRIER_COLS_QUANTUM = 4
# Relative period deviation the raster snap is allowed to introduce silently.
# Above it the baked switch crosses at a visibly different tilt angle than the
# stamped one, so it is logged at WARNING next to the coarse-bake record.
BARRIER_SNAP_TOL = 0.02


def _barrier_plate_lattice(period_um: float, pitch_um: float) -> tuple[int, float]:
    """Snap a barrier period onto the plate raster: ``(cols_per_period, period_um)``.

    The plate compositor cannot call ``_helpers.barrier_lattice`` verbatim — that
    solver derives the back RASTER from the slit lattice, whereas here the raster
    pitch is already fixed by the plate's lattice budget — but the constraint that
    matters is the same one: the front comb and the back A|B interlace must live on
    ONE lattice, so their periods are equal by construction instead of by two float
    phases agreeing. See its docstring for the physics, and for why the straddle
    class (channel A on the +x side of every slit) is the class both paths owe the
    manifest's ``switch_half_angle_deg``.

    On a raster that means a whole number of cells per period, and a multiple of
    ``BARRIER_COLS_QUANTUM`` of them. A fractional cell count is what the two
    analytic phases used to produce: the printed slit centre lands up to half a
    cell off the printed channel boundary and the printed slit width alternates
    between floor and ceil cells — at the ~11 cells per period the default box
    bakes at, ~18 % of the ±p/4 shift that only has ~2.7 cells of margin.

    When the requested period is under four cells the raster simply cannot carry
    it; the returned period is then COARSER than asked (the caller reports the
    deviation, which scales the switch tilt angle by the same factor).
    """
    q = BARRIER_COLS_QUANTUM
    cols = max(q, q * int(round(period_um / (q * pitch_um))))
    return cols, cols * pitch_um


def _barrier_masks(
    w_px: int, cols_per_period: int, anchor_col: int
) -> tuple["np.ndarray", "np.ndarray"]:
    """1-D column masks of an exactly registered barrier: ``(comb_bar, lane_a)``.

    ``comb_bar`` is the opaque FRONT bar (its complement is the open slit),
    ``lane_a`` the BACK channel-A lane. Both are pure column-index arithmetic on
    one shared lattice, which is what makes the registration exact: with ``c``
    cells per period the A|B boundary sits on the cell edge at ``anchor_col`` and
    the open slit is the ``c/2`` cells centred on that SAME edge, so every
    open-slit centre is a B→A boundary with channel A on its +x side. That is the
    straddle class and the "+tilt reveals B" parity the six generators solve for.

    Vertical bars only (``CENTER_SWITCH_AXIS_DEG`` = 0, the same assumption
    ``_check_front_comb_pure`` makes); a rotated barrier would have to go back
    through ``_grating_grid`` and give up cell-exact registration.
    """
    import numpy as np

    c = cols_per_period
    half = c // 2
    quarter = c // 4
    rel = (np.arange(w_px) - anchor_col) % c
    lane_a = rel < half
    comb_bar = ((rel + quarter) % c) >= half
    return comb_bar, lane_a


def _check_front_comb_pure(
    comb: "np.ndarray",
    box: tuple[int, int, int, int],
    plate_id: str,
    slug: str,
) -> None:
    """Contract check: a barrier-interlace front comb carries NO image feature.

    On SWITCH_INTERLACE_SLUGS the front (outer) layer must be a pure period-p
    comb across the art box. The front mask does not move with tilt, so any
    row-to-row variation in its column support is a static silhouette residual
    that no tilt angle can gate out — the banned front-image construction. The
    comb axis is CENTER_SWITCH_AXIS_DEG = 0 (vertical bars), so "pure" is
    exactly "every non-empty art-box row has identical column support"; rows the
    keep-out rim cleared entirely are skipped. Cheap: one row-broadcast compare
    over the art-box slice, no GEOS.

    Logged, not raised — a residual is an upstream design defect to fix, not a
    reason to refuse to serve an otherwise valid fab pair.
    """
    import numpy as np

    y0, y1, x0, x1 = box
    sub = comb[y0:y1, x0:x1]
    if sub.size == 0:
        return
    rows = sub[sub.any(axis=1)]
    if rows.shape[0] == 0:
        return
    differs = (rows != rows[0]).any(axis=1)
    if bool(differs.any()):
        _log.error(
            "barrier-interlace FRONT comb is not a pure period-p comb: %d of %d "
            "art-box rows differ in column support, so the front plane carries an "
            "image-shaped feature that cannot vanish under tilt (plate=%s slug=%s)",
            int(differs.sum()),
            int(rows.shape[0]),
            plate_id,
            slug,
        )


def ensure_plate_svg(plate_id: str) -> tuple[Path, Path] | None:
    """Lazily build the SVG fab pair for an existing plate (box-first moiré).

    FRONT = the foliage silhouette FILLED with a true fine grating (front
            period @ front angle), clipped to the silhouette by a row-span /
            scanline pass (silhouette raster AND grating raster → row-run
            rectangles via ``raster_to_polygons``). No GEOS booleans.
    BACK  = a uniform fine grating (back period @ back angle) across the whole
            back-carrier window rectangle.

    The two gratings differ by the small period ratio + angle offset stamped in
    ``recipe_data`` — the same moiré the shader renders analytically, now as
    real gold lines the fab can write. Respects the 400k lattice budget by
    rasterizing at a bounded pitch, so line count stays in the thousands.

    On a real (tens-of-mm) plate that budget pitch is far coarser than the
    design periods, so the baked geometry is COARSENED (see the bake_scale block
    below) — a preview-grade mask, not a production one. When that happens the
    EFFECTIVE baked periods are stamped into the manifest's ``recipe_data``
    (``svg_bake_*`` + ``svg_bake_coarsened``) and logged at WARNING, so nothing
    downstream can mistake these files for true-pitch masks; true-pitch geometry
    comes from ``export_fine.build_plate_fine`` / a tiled GDS export.

    Returns (front, back) paths or None if the plate is unknown (or its manifest
    is unreadable — the spec to bake from lives in it).
    """
    plate_dir = PLATES_ROOT / plate_id
    manifest_path = plate_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    front_svg = plate_dir / "front.svg"
    back_svg = plate_dir / "back.svg"
    if front_svg.exists() and back_svg.exists() and _svg_is_current(front_svg):
        return front_svg, back_svg
    # Same lock key as materialize_plate: the bake is heavy AND rewrites the
    # shared manifest (svg_bake_* keys), so it must not interleave with a
    # recompose of the same slot or with a second export of the same plate.
    with cache_lock(f"plate:{plate_id}"):
        if front_svg.exists() and back_svg.exists() and _svg_is_current(front_svg):
            return front_svg, back_svg
        return _bake_plate_svg(plate_id)


def _bake_plate_svg(plate_id: str) -> tuple[Path, Path] | None:
    """Raster + write the fab SVG pair. Caller holds the plate cache lock."""
    import numpy as np

    from .patterns._helpers import MAX_LATTICE_CELLS, check_lattice_budget, raster_to_polygons

    plate_dir = PLATES_ROOT / plate_id
    manifest_path = plate_dir / "manifest.json"
    front_svg = plate_dir / "front.svg"
    back_svg = plate_dir / "back.svg"

    manifest = read_json_cache(manifest_path)
    if manifest is None or "spec" not in manifest:
        return None
    spec = PlateSpec.from_dict(manifest["spec"])
    if spec.pattern_slug == BLANK_SLUG:
        # BARE GLASS: two valid but empty documents. Written (rather than
        # skipped) so the export bundle carries a file per layer per face and a
        # fab reader sees "this face is blank", not "this face is missing".
        _publish_svg_pair(
            manifest, manifest_path, plate_id, front_svg, back_svg,
            spec.width_um, spec.height_um, "", "", {},
        )
        return front_svg, back_svg
    rd = _carrier_recipe_data(spec)
    back_period = float(rd["fab_back_period_um"])
    front_period = float(rd["fab_front_period_um"])
    angle_off = float(rd["fab_angle_offset_deg"])
    base_angle = float(rd["carrier_angle_deg"])
    duty = float(rd["grating_duty"])
    frame_span = float(rd["frame_angle_span_deg"])
    frame_count = int(round(float(rd["frame_bucket_count"])))

    W, H = spec.width_um, spec.height_um

    # Fab raster pitch: fine enough to resolve the grating lines (>= 4 samples
    # per period) but the grid MUST fit the 400k lattice budget — a run-length
    # merge still allocates one row-span rectangle per line per row, and the
    # export writes real polygons. On a large plate a true 22 µm grating over
    # the whole face would blow past the cap, so if the ideal 4-samples/period
    # pitch would overflow, we COARSEN the grating period (preserving the 1.06
    # ratio + angle offset) to the finest the budget allows. The moiré geometry
    # is scale-free, so a coarser-but-legal grating carries the identical beat;
    # a genuinely sub-30 µm production mask would come from a tiled/streamed
    # GDS export, not this single-shot lazy SVG path.
    #
    # The coarsening is a real substitution (order-of-magnitude on a 50 mm face),
    # so it is NOT silent: ``bake_scale`` is logged at WARNING and the effective
    # periods land in the manifest as ``svg_bake_*`` below. The design periods in
    # ``fab_*`` stay untouched — the pair of numbers is what tells a reader these
    # SVGs are preview-grade.
    ideal_pitch = min(back_period, front_period) / 4.0
    budget_pitch = math.sqrt(W * H / (0.92 * MAX_LATTICE_CELLS))
    pitch = max(ideal_pitch, budget_pitch)
    bake_scale = (pitch / ideal_pitch) if ideal_pitch > 0 else 1.0
    if bake_scale > 1.0:
        back_period *= bake_scale
        front_period *= bake_scale
    fw = max(1, int(round(W / pitch)))
    fh = max(1, int(round(H / pitch)))
    check_lattice_budget(fw * fh, "plate grating raster", pitch_um=pitch, w=fw, h=fh)

    # Centerpiece stripe carrier (vertical): fab period + the half-period phase
    # offset baked into the back globe. Coarsen with the frame grating if the
    # budget forced a coarser pitch (same scale factor keeps the beat).
    center_period = float(rd.get("fab_center_period_um", rd["center_period_um"]))
    if bake_scale > 1.0:
        center_period *= bake_scale
    center_axis = float(rd["switch_axis_deg"])

    # Effective-bake record. Written into the manifest's recipe_data next to the
    # (unchanged) design ``fab_*`` fields so the export bundle is self-consistent:
    # a consumer reading fab_back_period_um==22 alongside svg_bake_back_period_um
    # ==330 knows exactly what these SVGs are. Empty dict when the bake is
    # period-exact AND the face is not a barrier interlace (whose lattice snap is
    # recorded even at a period-exact pitch); the keys are STRIPPED from a stale
    # manifest below either way (a finer plate must not inherit a coarse record).
    svg_bake: dict[str, Any] = {}
    if bake_scale > 1.0:
        svg_bake = {
            "svg_bake_coarsened": True,
            "svg_bake_scale": bake_scale,
            "svg_bake_pitch_um": pitch,
            "svg_bake_back_period_um": back_period,
            "svg_bake_front_period_um": front_period,
            "svg_bake_center_period_um": center_period,
        }
        _log.warning(
            "plate SVG bake COARSENED %.2f× to fit the lattice budget "
            "(raster pitch %.3g µm): back %.4g→%.4g µm, front %.4g→%.4g µm, "
            "center %.4g→%.4g µm. front.svg/back.svg are PREVIEW-grade, not fab "
            "masks — use export_fine/tiled GDS for true-pitch geometry "
            "(plate=%s slug=%s)",
            bake_scale,
            pitch,
            back_period / bake_scale,
            back_period,
            front_period / bake_scale,
            front_period,
            center_period / bake_scale,
            center_period,
            plate_id,
            spec.pattern_slug,
        )
    is_interlace = spec.pattern_slug in SWITCH_INTERLACE_SLUGS
    is_photo = spec.pattern_slug == PHOTO_SLUG
    single_ply = bool(getattr(spec, "single_ply", False))
    aperture = _aperture(spec)
    side_px = max(8, int(round(CENTERPIECE_FILL * aperture / pitch))) if aperture > 0 else 0
    cxg = fw // 2
    cyg = fh // 2

    # Barrier REGISTRATION on the plate raster. The front comb and the back A|B
    # lanes used to be two independent analytic gratings agreeing only through a
    # pair of float phase constants (-0.25 and 0.0) written 60 lines apart in two
    # different blocks; at a raster pitch that does not divide the period a whole
    # number of times that agreement does not survive sampling. Solve ONE lattice
    # here — cell-exact period, cell-exact phase, both layers built from it below.
    #
    # Anchored on ``cxg`` — the grid centre column, which is also the centerpiece
    # art-box centre column — so the registration is independent of the aperture:
    # the extent-dependent class flip ``_helpers.barrier_lattice`` has to solve for
    # (its raster starts at the tile EDGE) cannot arise here, and the published
    # ``switch_barrier_phase_um`` = 0 stays true at every plate size.
    barrier_cols = 0
    barrier_anchor = cxg
    barrier_comb: "np.ndarray | None" = None
    barrier_lane_a: "np.ndarray | None" = None
    if is_interlace:
        if abs(center_axis % 180.0) > 1e-9:
            # Cell-exact registration is column arithmetic, so it only exists for
            # vertical bars. A rotated switch axis would have to go back through
            # _grating_grid and give the registration up — and _check_front_comb_pure
            # would stop meaning anything either. Loud, but keep baking.
            _log.error(
                "barrier-interlace face has a non-vertical switch axis (%.3f°); the "
                "cell-exact barrier lattice below assumes vertical bars, so the "
                "baked comb ignores the rotation (plate=%s slug=%s)",
                center_axis,
                plate_id,
                spec.pattern_slug,
            )
        want_period = center_period
        barrier_cols, center_period = _barrier_plate_lattice(want_period, pitch)
        barrier_comb, barrier_lane_a = _barrier_masks(fw, barrier_cols, barrier_anchor)
        # The A|B boundary is the LEFT EDGE of column ``barrier_anchor``; x = 0 is
        # the centre of column (fw-1)/2 (see ``_grating_grid``). Zero on an even
        # grid, half a cell on an odd one — reported, not hidden, because it is the
        # number a shader would need to draw the same lattice.
        barrier_phase_um = (barrier_anchor - (fw - 1) / 2.0 - 0.5) * pitch
        period_err = center_period - want_period
        svg_bake.update(
            {
                # Achieved comb == interlace period, and the witnesses that say so
                # (the generator side publishes the same three in ``extra``).
                "svg_bake_barrier_period_um": center_period,
                "svg_bake_barrier_cell_um": pitch,
                "svg_bake_barrier_cols_per_period": barrier_cols,
                "svg_bake_barrier_phase_um": barrier_phase_um,
                "svg_bake_barrier_period_err_um": period_err,
            }
        )
        if svg_bake.get("svg_bake_coarsened"):
            # The coarse-bake record above was stamped from the pre-snap period.
            svg_bake["svg_bake_center_period_um"] = center_period
        if abs(period_err) > BARRIER_SNAP_TOL * want_period:
            _log.warning(
                "barrier lattice SNAPPED to the plate raster: %d cells × %.4g µm "
                "= %.4g µm comb+interlace period against the requested %.4g µm "
                "(%+.1f %%, so the baked switch crosses at %.2f× the tilt angle "
                "the manifest stamps). Registration itself is EXACT at the snapped "
                "period — every open-slit centre sits on an A|B channel boundary — "
                "but the raster pitch is set by the lattice budget and only a whole "
                "multiple-of-%d cell count can hold that. front.svg/back.svg are "
                "preview-grade here; true-pitch barrier geometry comes from "
                "export_fine (plate=%s slug=%s)",
                barrier_cols,
                pitch,
                center_period,
                want_period,
                100.0 * period_err / want_period,
                center_period / want_period,
                BARRIER_COLS_QUANTUM,
                plate_id,
                spec.pattern_slug,
            )

    def _center_masks() -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Full-grid (front_art, back_art, front_accent) bools for the aperture.

        Data-directed by the pattern slug (colibrí/globe, gear/quill, capybara,
        food, monogram, inscription — whatever ``_centerpiece_masks`` returns),
        so the fab pair matches the preview centerpiece. ``front_accent`` is the
        subset of the FRONT art that carries the sub-5 µm diffraction grating.
        """
        empty = np.zeros((fh, fw), dtype=bool)
        if side_px <= 0:
            return empty, empty.copy(), empty.copy()
        masks = _centerpiece_masks(spec.pattern_slug, side_px, spec.pattern_params)
        if masks is None:
            return empty, empty.copy(), empty.copy()
        from PIL import Image as _Image

        def _place(art: "np.ndarray") -> "np.ndarray":
            if art.shape[0] != side_px or art.shape[1] != side_px:
                im = _Image.fromarray((art.astype(np.uint8) * 255), "L").resize(
                    (side_px, side_px), _Image.NEAREST
                )
                art = np.asarray(im) > 127
            m = np.zeros((fh, fw), dtype=bool)
            x0 = cxg - side_px // 2
            y0 = cyg - side_px // 2
            xa = max(0, x0); ya = max(0, y0)
            xb = min(fw, x0 + side_px); yb = min(fh, y0 + side_px)
            m[ya:yb, xa:xb] = art[ya - y0 : yb - y0, xa - x0 : xb - x0]
            return m, art

        front_full, front_side = _place(masks[0])
        back_full, _ = _place(masks[1])
        # Accent zone at the SAME resolution the art was placed at (front_side),
        # then re-place so it lands on the full grid exactly under the art.
        accent_zone = _front_accent_zone(spec.pattern_slug, front_side)
        if accent_zone is None:
            accent_full = np.zeros((fh, fw), dtype=bool)
        else:
            accent_full, _ = _place(accent_zone)
        return front_full, back_full, accent_full

    # Centerpiece art (front + back) and the front diffraction-accent subset,
    # placed on the full fab grid once so both layers share them.
    front_art, back_art, front_accent = _center_masks()

    # --- Water SCANIMATION fab geometry (capybara back face) ----------------
    # For the capybara face the back centerpiece is not a phase-π stripe globe:
    # it is a barrier-grid water scanimation. Bake the STANDALONE builder's real
    # geometry into the aperture so front.svg/back.svg carry (a) the capybara
    # shimmer + a slit barrier over the water, and (b) the N interleaved ripple
    # frames. Both are full-grid bool overrides consumed below (None for every
    # other slug, so their fab output is byte-identical).
    # CAVEAT: the builder is sampled at the plate's coarse budget pitch, so the
    # 60/15/45 µm barrier geometry ALIASES here and the interleave can degenerate
    # to empty (see `water_scan_empty`). This path is preview-grade like the rest
    # of the coarse bake; the period-exact scanimation rects live in export_fine.
    water_front_extra: "np.ndarray | None" = None
    water_back: "np.ndarray | None" = None
    water_band_placed: "np.ndarray | None" = None
    # True when the interleave returned NOTHING at this raster pitch (the coarse
    # budget pitch leaves <1 cell per 15 µm slot, so `_interleave_phases`'
    # atomic-slot floor rejects every slot). Consumed by the BACK block, which
    # then keeps the plain carrier across the band instead of clearing it.
    water_scan_empty = False
    if spec.pattern_slug == WATER_SCAN_SLUG and side_px > 0:
        from .patterns.artistic import capybara_scanimation as _capyscan

        # WATER FULL WIDTH: build the water band / slit comb / interleave across
        # the full WIDTH-axis aperture (`_aperture_width_um`, edge-to-edge of the
        # window) while the capybara body + flow wake stay in the centered
        # CENTERPIECE_FILL square (open current flanks the animal).
        b = _capyscan._build(
            extent_um=CENTERPIECE_FILL * aperture,
            frame_pitch_um=water_scan_fab_pitch_um(spec),
            n_phases=WATER_SCAN_N_PHASES,
            carrier_period_um=WATER_SCAN_FAB_CARRIER_UM,
            waterline_y=_water_waterline_y(spec.pattern_params),
            n_grid=side_px,
            water_extent_um=_aperture_width_um(spec),
        )
        # FRONT: capybara body shimmer + slit-barrier bars over the water band.
        # ``b["water_band"]`` is the CARVED band (``below & ~capy``, the same
        # algebra `_centerpiece_masks` uses for the preview), so the bars stop at
        # the animal's outline instead of striping across the submerged body.
        barrier_bars = (~b["barrier"]) & b["water_band"]
        front_side = b["capy_shimmer"] | barrier_bars
        # BACK: interleaved ripple frames (all N phases packed into 1/N slots).
        back_side = b["back_water"]
        if not back_side.any():
            # 60 µm frame pitch / 4 phases = a 15 µm slot, i.e. ~0.2 cells at the
            # coarse budget pitch: `_interleave_phases`' litho-floor coverage test
            # fails for every slot and the mask comes back all-False. Do NOT clear
            # the carrier for it — an empty "animated region" would print as bare
            # glass across the full-width water band (~30 mm on the default box).
            water_scan_empty = True
            _log.warning(
                "water scanimation interleave EMPTY at raster pitch %.3g µm "
                "(frame pitch %.3g µm / %d phases = %.3g µm slot); back.svg keeps "
                "the plain carrier across the water band and carries NO "
                "scanimation — true-pitch geometry comes from export_fine "
                "(plate=%s)",
                pitch,
                water_scan_fab_pitch_um(spec),
                WATER_SCAN_N_PHASES,
                water_scan_fab_pitch_um(spec) / WATER_SCAN_N_PHASES,
                plate_id,
            )

        def _place_side(side_mask: "np.ndarray") -> "np.ndarray":
            w_px = side_mask.shape[1]
            m = np.zeros((fh, fw), dtype=bool)
            x0 = cxg - w_px // 2
            y0 = cyg - side_px // 2
            xa = max(0, x0); ya = max(0, y0)
            xb = min(fw, x0 + w_px); yb = min(fh, y0 + side_px)
            m[ya:yb, xa:xb] = side_mask[ya - y0 : yb - y0, xa - x0 : xb - x0]
            return m

        water_front_extra = _place_side(front_side)
        _mask_rim(water_front_extra, spec.weld_margin_um, pitch)
        water_back = _place_side(back_side)
        # Full-width water BAND (below waterline, edge-to-edge, submerged body
        # carved out) so the back carrier can be cleared across the whole band, not
        # just the body square — and, because of the carve, NOT under the animal:
        # the submerged body keeps the plain carrier, exactly as the preview shows.
        water_band_placed = _place_side(b["water_band"])

    # --- FRONT: perimeter foliage frame (grating) + centerpiece -------------
    active_w, active_h = spec.active_dims()
    front_group = ""
    if active_w > 0 and active_h > 0:
        rect = RectFrame(width_um=active_w, height_um=active_h)
        frame_params = spec.frame.to_frame_params()
        frame_params.fill_interior = False  # perimeter band, center open for art
        scene = frame_scene_for_plate(plate_dir, manifest, rect, frame_params)
        # Per-motif graylevel silhouette — the same angle-bucket encoding the
        # preview uses, so the fab lines match the shimmer the user sees.
        sil_img = render_scene_to_image(
            scene, rect, frame_params, pitch, level_fn=_frame_level_for
        )
        # Paste the active-rect graylevel band into a full-plate grid.
        sil_lvl = np.zeros((fh, fw), dtype=np.uint8)
        aw = min(fw, sil_img.size[0])
        ah = min(fh, sil_img.size[1])
        ox = (fw - aw) // 2
        oy = (fh - ah) // 2
        sil_lvl[oy : oy + ah, ox : ox + aw] = np.asarray(sil_img)[:ah, :aw]
        sil = sil_lvl > 0
        # Enforce the front keep-out rim.
        _mask_rim(sil, spec.weld_margin_um, pitch)
        # Fill each angle bucket with a grating rotated to that bucket, so the
        # baked gold lines carry the SAME per-motif fringe directions as the
        # shader (frameAngle = base + (b - (N-1)/2)·span). Decode the graylevel
        # back to a bucket index and OR the per-bucket grating in.
        base_front = base_angle + angle_off
        frame_grating = np.zeros((fh, fw), dtype=bool)
        for b in range(frame_count):
            lvl_lo = FRAME_BUCKET0 + b * FRAME_BUCKET_STEP - FRAME_BUCKET_STEP // 2
            lvl_hi = FRAME_BUCKET0 + b * FRAME_BUCKET_STEP + FRAME_BUCKET_STEP // 2
            in_bucket = (sil_lvl > max(0, lvl_lo)) & (sil_lvl <= lvl_hi)
            if not in_bucket.any():
                continue
            ang = base_front + (b - 0.5 * (frame_count - 1)) * frame_span
            g = _grating_grid(fw, fh, pitch, front_period, duty, ang)
            frame_grating |= in_bucket & g
        # Centerpiece front layer. Two constructions:
        #   * barrier-interlace (SWITCH_INTERLACE_SLUGS): a NEUTRAL slit comb
        #     (60 µm pitch, open duty 0.5 → one 30 µm lane) over the FULL
        #     centerpiece art box — the CENTERPIECE_FILL square, NOT the union
        #     of the silhouettes. A union-clipped comb is itself a static front
        #     image (its envelope is the union, which never moves with tilt —
        #     the measured ~0.31 front residual), and physically the comb must
        #     cover every column any back lane can slide under within the first
        #     zone (≥ p/2 beyond the union), so the full art-box square is the
        #     clean choice — same treatment the capybara water band gets (bars
        #     across the whole band). Comb AND lanes come from the one lattice
        #     solved above (``_barrier_plate_lattice`` / ``_barrier_masks``), so
        #     every open slit straddles an A|B lane boundary head-on to the cell
        #     and ± tilt reveals A or B cleanly; the solved phase is published as
        #     ``switch_barrier_phase_um`` for the preview shader to draw the same
        #     lattice instead of its own constant.
        #   * water scanimation (capybara): NOTHING here — the dry body's only
        #     grating is the 24 µm shimmer OR-ed in with `water_front_extra`
        #     below. Filling `front_art` with the centerpiece carrier as well
        #     would superimpose two 50 %-duty gratings at different periods over
        #     the same body (~75 % gold), destroying both the duty and the
        #     shimmer contrast.
        #   * legacy phase-switch (front-only shimmer faces): the FRONT
        #     silhouette filled with the vertical switch carrier (phase 0).
        # Will the sub-5 µm diffraction accent actually be baked further down? The
        # physics needs pitch ≤ period/4, which the coarse budget pitch almost
        # never allows. Decided HERE because the constructions below carve the
        # accent zone out of their comb/carrier on the premise that the fine
        # grating fills it back in — when the fine bake is skipped, that carve-out
        # is BARE GLASS (a ~7 mm hole through the gear hub, ellipses through the
        # monogram flourishes) where the preview PNG paints gold. So the carve-out
        # happens only when the grating that replaces it really follows.
        from .patterns.effects.gratings import (
            DIFFRACTION_ACCENT_PERIOD_UM,
            diffraction_accent_grating,
        )

        accent_max_pitch = DIFFRACTION_ACCENT_PERIOD_UM / 4.0
        accent_fine_ok = pitch <= accent_max_pitch
        if is_interlace:
            art_box = np.zeros((fh, fw), dtype=bool)
            bx0 = by0 = bx1 = by1 = 0
            if side_px > 0:
                bx0 = max(0, cxg - side_px // 2)
                by0 = max(0, cyg - side_px // 2)
                bx1 = min(fw, bx0 + side_px)
                by1 = min(fh, by0 + side_px)
                art_box[by0:by1, bx0:bx1] = True
                _mask_rim(art_box, spec.weld_margin_um, pitch)
            # Opaque bars of the solved lattice (``barrier_comb`` is a column
            # mask; its complement is the open slit, centred on an A|B boundary).
            art_carrier = art_box & barrier_comb[None, :]
        elif water_front_extra is not None or is_photo:
            # Nothing from the raster path. The capybara's grating is the exact
            # vector comb OR-ed in below; the photo's is the exact line screen
            # concatenated after the raster (see ``photo_band_rects``). In both
            # cases filling a silhouette with the centerpiece carrier as well
            # would superimpose a second 50 %-duty grating on geometry that is
            # already carrying the effect.
            art_carrier = np.zeros((fh, fw), dtype=bool)
        else:
            center_grating = _grating_grid(fw, fh, pitch, center_period, duty, center_axis)
            art_carrier = front_art & center_grating
        if accent_fine_ok:
            art_carrier &= ~front_accent
        if is_interlace:
            # Front plane of a barrier switch: image-free comb or nothing. Checked
            # AFTER the accent carve-out so a future accent regression (or any
            # other silhouette leaking into the comb) is caught, not just the
            # construction above. front_accent is empty here by _front_accent_zone.
            _check_front_comb_pure(
                art_carrier, (by0, by1, bx0, bx1), plate_id, spec.pattern_slug
            )
        front_grid = (sil & frame_grating) | art_carrier
        # Water scanimation: OR-in the capybara body shimmer + slit-barrier bars
        # over the water band (real barrier-grid geometry, not the stripe carrier).
        # This is the body's ONLY grating — the centerpiece-carrier branch above
        # deliberately contributes nothing on this face.
        if water_front_extra is not None:
            front_grid |= water_front_extra
        # Diffraction rainbow accent: OR-in a true 4.4 µm 45° grating clipped to
        # the accent zone. The physics REQUIRES the sub-5 µm period, so it needs
        # a raster pitch fine enough to resolve it (≥4 samples/period, i.e.
        # pitch ≤ period/4 ≈ 1.1 µm). At the coarse budget pitch this SVG bake
        # runs at (~19-82 µm) the 4.4 µm period aliases into baked noise, so we
        # SKIP the accent group here rather than write garbage — the accent zones
        # still live in the PNG masks (RAINBOW_LEVEL 200) for preview and are
        # emitted by the fine-pitch writer at the native period. See
        # effects.gratings.diffraction_accent_grating's integrator note. When it is
        # skipped the zone keeps the surrounding comb/carrier (accent_fine_ok
        # above): a missing spectral sheen is cosmetic, a hole is a scrapped plate.
        if front_accent.any():
            if accent_fine_ok:
                front_grid |= diffraction_accent_grating(front_accent, pitch)
            else:
                _log.info(
                    "skipping diffraction accent bake: raster pitch %.3g µm > "
                    "period/4 = %.3g µm (period %.3g µm would alias); accent "
                    "zones keep the surrounding carrier here and remain in the "
                    "PNG masks + fine-pitch export",
                    pitch,
                    accent_max_pitch,
                    DIFFRACTION_ACCENT_PERIOD_UM,
                )
        # SINGLE PLY: nothing extra on this layer. One sheet has no second plane
        # for a carrier to beat against, so (2026-09-10) the face writes the
        # photograph and the leaf gratings on bare glass and NOTHING else — the
        # same geometry ``_raster_compose_plate`` composes and the single-ply
        # block in ``export_fine.build_plate_fine`` writes ("fab SVG = preview
        # PNG", CLAUDE.md). The carrier this branch used to bake over the back
        # window was the earlier design; on one ply its beat with the leaves was
        # static bars, not a moiré, so it went — and a bake that still emitted it
        # would have put ~500 mm² of gold on the mask that no other path has.
        front_polys = raster_to_polygons(front_grid, pitch, (W, H))
        if is_photo:
            # The line screen is EXACT vector geometry — bands plus, inside each
            # coloured band, its diffraction sub-grating — so it bypasses the
            # coarse budget raster entirely and is concatenated (never unioned;
            # see _concat_polygons) onto the frame's polygons.
            front_polys = _concat_polygons(
                front_polys, _photo_multipolygon(spec)
            )
        front_group = _group("frame+centerpiece", _strip_svg_body(to_svg(front_polys, (W, H), background=None)))

    # --- BACK: uniform carrier grating + globe centerpiece (phase π) --------
    # Empty on a single-ply face: there is no inner ply, and the carrier that
    # would live here is already on the front layer above.
    back_group = ""
    back_w, back_h = spec.back_dims()
    if not single_ply and back_w > 0 and back_h > 0:
        win = _back_window_grid(spec, fw, fh, pitch)
        carrier_grating = _grating_grid(fw, fh, pitch, back_period, duty, base_angle)
        if is_interlace:
            # Barrier-interlace back layer: BOTH silhouettes interleaved in
            # alternating lanes (lane pitch = half the barrier pitch). A (front
            # silhouette) fills the channel-A lanes, B (back silhouette) the
            # channel-B lanes. ``barrier_lane_a`` is the SAME solved lattice the
            # front comb above is cut from — that shared lattice IS the
            # registration, so the two periods cannot drift and each open slit
            # sits on a lane boundary to the cell. This is the whole moving image
            # — the front barrier reveals one lane class per tilt. The plain
            # carrier is cleared across the union so the lanes read.
            union_art = front_art | back_art
            even_lane = barrier_lane_a[None, :]
            interleave = (front_art & even_lane) | (back_art & ~even_lane)
            back_grid = (win & carrier_grating & ~union_art) | (win & interleave)
        else:
            # Back centerpiece art filled with the vertical switch carrier, shifted
            # by half a period (phase π) so it interlaces with the front art. Empty
            # for front-only-shimmer patterns (monogram/inscription/food/capybara),
            # in which case the back is a clean uniform carrier window.
            center_grating_pi = _grating_grid(
                fw, fh, pitch, center_period, duty, center_axis, phase=0.5
            )
            # The back art overrides the carrier inside the aperture.
            back_grid = (win & carrier_grating & ~back_art) | (back_art & center_grating_pi)
        # Water scanimation: the back-art region is the WATER BAND, filled with
        # the N interleaved ripple frames (a genuine scanimation) rather than the
        # phase-π stripe globe. Clear the plain-carrier fill inside the water band
        # and OR-in the real ripple geometry (clipped to the back window rim). The
        # band excludes the submerged body, so that clear-out does NOT strip the
        # carrier from under the animal and the interleave cannot print on it —
        # unless the interleave came back empty at this raster pitch, in which case
        # the band keeps the carrier (clearing it would erase gold for nothing).
        if water_back is not None and not water_scan_empty:
            band = water_band_placed if water_band_placed is not None else back_art
            back_grid = (win & carrier_grating & ~band) | (win & water_back)
        elif water_scan_empty:
            # Degenerate interleave at this raster pitch (warned above). Keep the
            # plain carrier across the WHOLE back window — including the band —
            # rather than clearing ~30 mm of gold for an empty mask, and skip the
            # phase-π stripe fill the generic else-branch would have put in the
            # band square (that architecture was never the capybara's).
            back_grid = win & carrier_grating
        back_polys = raster_to_polygons(back_grid, pitch, (W, H))
        back_group = _group("carrier+centerpiece", _strip_svg_body(to_svg(back_polys, (W, H), background=None)))

    _publish_svg_pair(
        manifest, manifest_path, plate_id, front_svg, back_svg,
        W, H, front_group, back_group, svg_bake,
    )
    return front_svg, back_svg


def _publish_svg_pair(
    manifest: dict[str, Any],
    manifest_path: Path,
    plate_id: str,
    front_svg: Path,
    back_svg: Path,
    width_um: float,
    height_um: float,
    front_group: str,
    back_group: str,
    svg_bake: dict[str, Any],
) -> None:
    """Write the fab SVG pair and stamp the manifest that describes it.

    SVGs first, manifest last — the manifest's ``svg_bake_*`` record must never
    describe geometry that is not on disk yet. Shared with the BLANK early
    return so a blank face publishes through exactly the same steps (empty
    groups, cleared bake record) rather than a second, nearly-identical tail.
    """
    write_text_atomic(front_svg, _wrap_svg(width_um, height_um, [front_group]))
    write_text_atomic(back_svg, _wrap_svg(width_um, height_um, [back_group]))

    files = manifest.setdefault("files", {})
    files["front_svg"] = f"/data/plates/{plate_id}/front.svg"
    files["back_svg"] = f"/data/plates/{plate_id}/back.svg"
    # Stamp (or clear) the effective-bake record so the manifest that ships next
    # to these SVGs describes the geometry they actually contain. The design
    # ``fab_*`` fields are left alone — the export bundle needs BOTH numbers.
    rd_out = manifest.setdefault("recipe_data", {})
    for key in _SVG_BAKE_KEYS:
        rd_out.pop(key, None)
    rd_out.update(svg_bake)
    write_json_atomic(manifest_path, manifest)


# recipe_data keys ensure_plate_svg writes to describe the EFFECTIVE baked
# geometry when the lattice budget forced a coarser period (see the bake_scale
# block) or a coarser barrier lattice (the barrier snap). Listed here so a re-bake
# at a finer pitch can strip a stale record instead of leaving a manifest that
# claims a coarsening that no longer applies.
_SVG_BAKE_KEYS = (
    "svg_bake_coarsened",
    "svg_bake_scale",
    "svg_bake_pitch_um",
    "svg_bake_back_period_um",
    "svg_bake_front_period_um",
    "svg_bake_center_period_um",
    "svg_bake_barrier_period_um",
    "svg_bake_barrier_cell_um",
    "svg_bake_barrier_cols_per_period",
    "svg_bake_barrier_phase_um",
    "svg_bake_barrier_period_err_um",
)


# Bump when the SVG compose geometry changes: cached plate SVGs are only
# reused if they carry the current marker, so a formula fix (e.g. the
# aperture-scaling fix) invalidates stale files under unchanged spec hashes.
# v4: barrier-interlace front comb spans the full art box (was union-gated).
# v5: coarse-bake honesty — effective periods stamped in the manifest, skipped
#     diffraction-accent zones keep the surrounding carrier instead of becoming
#     bare-glass holes, the capybara body carries only its 24 µm shimmer (no
#     superimposed centerpiece carrier), and a degenerate water interleave keeps
#     the plain carrier across the band instead of erasing it. Also in v5:
#     barrier-interlace faces emit NO front diffraction accent at all (a
#     silhouette-shaped front feature can never vanish under tilt), so the
#     gear-quill-switch hub grating is gone from front.svg.
# v6: barrier registration carried onto the plate path — the front comb and the
#     back A|B lanes are cut from ONE cell-exact lattice (_barrier_plate_lattice /
#     _barrier_masks) instead of two float-phased gratings that lose registration
#     to sampling, so the baked comb period equals the baked interlace period and
#     every open-slit centre lands on a channel boundary. Interlace face geometry
#     shifts by up to a cell and the baked barrier period snaps to a multiple of
#     four raster cells (recorded in svg_bake_barrier_*).
# v7: the water scanimation bakes at the face's EFFECTIVE waterline
#     (_water_waterline_y) instead of the module constant, so a face that sets
#     the ``waterline`` param gets a band/wake matching its preview mask. Only
#     such faces change; default-param geometry is byte-identical.
# v8: the water band EXCLUDES the submerged capybara. The band the bake consumes
#     (``_capybara_and_water``'s ``water_band``) is now ``below & ~capy`` — the
#     composed preview's convention — so front.svg no longer lays slit-barrier
#     bars across the animal, back.svg no longer interleaves ripple crests under
#     it, and the plain back carrier is kept over the submerged body instead of
#     being cleared for the band. Only the capybara face changes.
# v12: production-box faces. A BLANK face bakes two empty documents; a
#     SINGLE-PLY face bakes the carrier grating into front.svg (over the back
#     window MINUS the art box) and an EMPTY back.svg; a PHOTO face bakes the
#     line-screen bands and their colour sub-gratings as exact rectangles at the
#     art box, front layer only — vector geometry that never passes through the
#     coarse budget raster, so it is period-exact even here.
# v13-v17: the PLATE FOR WRITING (2026-09-10), one entry for the five bumps of a
#     single design. The frame band moves on EVERY face — the art rim now starts
#     at the inner ply's window (``assembly.bonded_art_keepout_um``) and the band
#     is a fixed 2.4 mm at motif_scale 0.68 — and the carrier bakes at the
#     eye-sized 65.5 µm fixed pitch instead of the gap-scaled one. The lid bakes
#     no diffraction accent (one shading moiré over the whole monogram), and a
#     single-ply photo face bakes the leaf gratings of the chosen fill
#     (``SINGLE_PLY_LEAF_FILL``) rather than a common-pitch louvre.
# v18: a SINGLE-PLY face bakes NO carrier. v12 put a carrier grating in front.svg
#     over the back window minus the art box; compose stopped painting it and
#     ``export_fine.build_plate_fine`` never wrote it, so the fab SVG was the only
#     path still emitting ~500 mm² of gold that the mask does not have — the
#     exact drift "fab SVG = preview PNG" (CLAUDE.md) exists to catch. Only
#     single-ply faces change; every other face's SVG is byte-identical.
PLATE_SVG_VERSION = "plate-svg-v18"


def _svg_is_current(svg_path: Path) -> bool:
    try:
        head = svg_path.read_text(encoding="utf-8", errors="ignore")[:256]
    except OSError:
        return False
    return PLATE_SVG_VERSION in head


def _wrap_svg(width_um: float, height_um: float, groups: list[str]) -> str:
    # width/height carry an explicit physical unit (mm) so importers that
    # honor them place the plate at true scale; the viewBox keeps user units
    # = μm, matching every nested path coordinate.
    body = "".join(groups)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<!--{PLATE_SVG_VERSION}-->'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_um / 1000.0:.4f}mm" height="{height_um / 1000.0:.4f}mm" '
        f'viewBox="{-width_um/2:.2f} {-height_um/2:.2f} {width_um:.2f} {height_um:.2f}">'
        f'{body}</svg>'
    )


def _group(layer_id: str, inner: str) -> str:
    return f'<g id="{layer_id}">{inner}</g>'


def _inner_svg_paths(full_svg: str, scale: float = 1.0) -> str:
    """Strip the outer <svg>...</svg> wrapper off a drawsvg output so we can
    nest it inside our wrapper, optionally scaling the group about the shared
    centered origin (used to blow the central pattern up to the aperture).
    Drawsvg emits a fixed prelude that we don't want twice in one document.
    """
    # Drawsvg emits something like:
    #   <?xml ...?><svg ...><defs>...</defs><rect ...>...<path .../></svg>
    # Find the first '>' after '<svg' and the closing '</svg>'.
    open_idx = full_svg.find("<svg")
    if open_idx == -1:
        return full_svg
    open_end = full_svg.find(">", open_idx)
    close_idx = full_svg.rfind("</svg>")
    if open_end == -1 or close_idx == -1:
        return full_svg
    inner = full_svg[open_end + 1 : close_idx]
    if abs(scale - 1.0) > 1e-9:
        return f'<g id="central" transform="scale({scale:.8g})">{inner}</g>'
    return _group("central", inner)


def list_plates() -> list[dict[str, Any]]:
    if not PLATES_ROOT.exists():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(PLATES_ROOT.iterdir()):
        m = d / "manifest.json"
        if m.exists():
            cached = read_json_cache(m)  # None = corrupt slot, skip it
            if cached is not None:
                out.append(cached)
    return out


def get_plate(plate_id: str) -> dict[str, Any] | None:
    """The cached plate manifest, or None if absent OR unreadable.

    A truncated manifest reads as 'missing' (a 404 the caller can recover from
    by regenerating) rather than raising a 500 out of the route.
    """
    return read_json_cache(PLATES_ROOT / plate_id / "manifest.json")
