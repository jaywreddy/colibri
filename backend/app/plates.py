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
    generate_frame,
    render_scene_to_image,
    render_scene_to_svg,
    scene_to_multipolygon,
)
from .patterns.frames.api import FrameParams
from .rasterize import make_thumbnail
from .service import DATA_ROOT, materialize as materialize_pattern


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
# only (colibrí gorget, steam-curl tips, gear hub, monogram flourish tips) —
# keep them tiny so the sub-5 µm feature count stays inside the lattice budget.
RAINBOW_LEVEL = 200

# Frame-band angle-bucket palette. Levels live strictly inside the shader's
# frame window (FRAME_MIN·255 = 51 .. ART_MIN·255 = 191) so they still classify
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
FRAME_ANGLE_SPAN_DEG = 3.5


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
CARRIER_ANGLE_OFFSET_DEG = 3.0    # relative rotation between the two gratings
GRATING_DUTY = 0.5                # gold-line fraction of a period
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
# PREVIEW phase-advance pitch (μm). The shader walks one ripple phase per this
# many μm of substrate parallax. It is DELIBERATELY decoupled from both the fab
# slit pitch (60 μm, litho-verified) and the mm-scale crest spacing
# (WATER_RIPPLE_WAVELENGTH_FRAC · art_side ≈ 3 mm): dividing the phase walk by the
# crest wavelength made the water freeze (phase advanced ~0.13 of one step across
# a whole hand rock, because a realistic 3-20° tilt only shifts the substrate
# ~30-180 μm). At ~97 μm a ~4° tilt advances ~one full phase, so the current
# visibly streams as the box rocks while the crest SPACING stays fab-accurate.
# This is the water twin of the frame preview/fab period split (PREVIEW_BACK_
# PERIOD_UM vs BACK_CARRIER_PERIOD_UM) — preview timing only, no effect on the
# baked SVG/GDS scanimation. Verified: phaseStep sweeps multiple full phases and
# per-frame water coverage diff rises ~15-25× vs the frozen wavelength divisor.
WATER_PHASE_PITCH_PREVIEW_UM = 97.0
WATER_SCAN_SLUG = "capybara-scanimation"
# Barrier-interlace tilt-switch slugs (Task 3). These faces abandon the
# phase-offset construction (A on front, B on back, half-period shift) for a TRUE
# lenticular/Poemotion BARRIER INTERLACE: both silhouettes live on the BACK layer
# in alternating lanes (A even, B odd, lane pitch = half the barrier pitch) and
# the FRONT layer is a neutral slit barrier (open duty 0.5) over their union, so
# tilting one way shows ONLY A and the other ONLY B — a hard swap, not a
# redistribution. The barrier pitch is FAB_CENTER_PERIOD_UM (60 µm), so the swap
# crosses at the same comfortable ~5° tilt the old switch targeted. The preview
# shader draws the lanes/barrier procedurally (uSwitchInterlace); the fab SVG +
# fine GDS bake the same architecture at fab pitches (see ensure_plate_svg /
# export_fine), and the standalone generators mirror it.
SWITCH_INTERLACE_SLUGS = frozenset({"globe-duo-phase", "gear-quill-switch"})
# Fab (SVG/GDS) barrier-grid parameters — the standalone builder's defaults
# (60 µm slit pitch, N=4 → 15 µm slot, 24 µm body-shimmer carrier). These bake
# the REAL slit barrier + interleaved ripple frames into the box back plate.
WATER_SCAN_FAB_PITCH_UM = 60.0
WATER_SCAN_FAB_CARRIER_UM = 24.0


def _carrier_recipe_data(spec: PlateSpec) -> dict[str, Any]:
    """Per-plate moiré-carrier knobs the shader + fab path both read.

    The FRAME grating pair (perimeter foliage shimmer): the front louvre is
    rotated by ``CARRIER_ANGLE_OFFSET_DEG`` relative to the back and scaled by
    ``FRONT_GRATING_RATIO``; a per-face base angle keyed off the frame seed
    keeps each face's carrier visually distinct without changing the physics.

    The CENTERPIECE carrier (colibrí↔globe tilt switch): a finer vertical
    stripe carrier with a ±X switch axis, phase-shifted by half a period
    between the two silhouettes in the shader.
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
    front_pitch = carrier_pitch * FRONT_GRATING_RATIO
    # Preview periods (what the shader actually draws) — magnified so even the
    # 4 µm litho-floor pitch resolves on the two real planes; see
    # PREVIEW_PITCH_MAGNIFY. Proportional to the real pitch, so finer pitch reads
    # as finer fringes.
    preview_carrier = carrier_pitch * PREVIEW_PITCH_MAGNIFY
    preview_front = preview_carrier * FRONT_GRATING_RATIO
    # Water scanimation: only the capybara back face turns it on (N>0). Every
    # other face emits N=0, so the shader's water-scan branch is a no-op and the
    # centerpiece keeps its 2-phase colibrí/globe switch — pixel-identical.
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
        "switch_axis_deg": CENTER_SWITCH_AXIS_DEG,
        # Barrier-interlace tilt switch (Task 3): the shader draws the neutral
        # barrier (outer) + interleaved A/B lanes (inner) at the fab barrier pitch
        # instead of the phase-offset carrier. True only on the two hard-swap
        # faces; every other face keeps the phase-offset centerpiece.
        "switch_interlace": spec.pattern_slug in SWITCH_INTERLACE_SLUGS,
        # Water scanimation (capybara back face). N>0 makes the shader render an
        # N-phase travelling-ripple flow over the back-art (water) region of the
        # centerpiece; N=0 leaves the 2-phase colibrí/globe switch untouched.
        "water_scan_n": water_n,
        "water_ripple_wavelength_um": water_ripple_um,
        # Coarse PREVIEW phase-advance pitch (μm) — drives how fast the shader
        # walks the ripple phase with tilt, decoupled from the mm-scale crest
        # spacing so the preview water actually flows. 0 on non-water faces (the
        # branch is off there anyway). Fab timing is unaffected. See constant.
        "water_phase_pitch_preview_um": (
            WATER_PHASE_PITCH_PREVIEW_UM if is_water else 0.0
        ),
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
        # Fab (SVG/GDS) — true fine gratings baked as clipped rect arrays. These
        # are the user-tunable pitch (== carrier_period_um / slit_period_um): the
        # fab path (ensure_plate_svg, export_fine.build_plate_fine) reads these,
        # so a pitch change flows straight to the baked gold. The fab raster
        # coarsens the period only if the lattice budget forces it (see
        # ensure_plate_svg) — the design pitch here is exact.
        "fab_back_period_um": carrier_pitch,
        "fab_front_period_um": front_pitch,
        "fab_angle_offset_deg": CARRIER_ANGLE_OFFSET_DEG,
        # Fab centerpiece stripe period (comfortable ~5° switch, see constant).
        "fab_center_period_um": FAB_CENTER_PERIOD_UM,
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
    if slug == "colibri-globe-phase":
        from .patterns.motifs import colibri, globe

        n = max(64, int(n_px))
        return (
            colibri.colibri_silhouette((1.0, 1.0), n_grid=n),
            globe.globe_silhouette((1.0, 1.0), n_grid=n),
        )
    if slug == "colibri-flap-phase":
        # Wing-flap A/B tilt switch: the SAME hummingbird in two wing poses.
        # FRONT = pose "up" (hover V, carrier phase 0); BACK = pose "down"
        # (mid-downstroke + trailing speed slivers, carrier phase π). Body,
        # head, beak, neck, tail and feet are pixel-registered between poses
        # (colibri._draw_colibri only swaps the wing set), so the shader's
        # centerpiece phase-switch reads as ONE bird flapping — not two birds.
        # Mirrors the colibrí/globe branch; the shader supplies the glimmer.
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

        from .patterns.artistic.capybara_scanimation import WATERLINE_Y
        from .patterns.motifs.lab import capybara

        n = max(64, int(n_px))
        capy = capybara.capybara_silhouette((1.0, 1.0), n_grid=n)
        rows = (np.arange(n)[:, None] / n)  # y in 0..1, y-down (Pillow order)
        below = np.broadcast_to(rows >= WATERLINE_Y, capy.shape)
        # Dry capybara (above waterline) on the FRONT; water band (below the
        # line, not covered by the submerged body) on the BACK.
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
# out (colibrí gorget, steam-curl tips, gear hub, monogram flourish tips). The
# zone is defined in the same normalized 0..1 art box the motif silhouette uses
# (y-DOWN, matching the Pillow motifs), as a centered ellipse / rect, then AND-ed
# with the art silhouette so only gold pixels inside the shape become accent.
# Painting these at RAINBOW_LEVEL (instead of ART_LEVEL) makes the shader add a
# spectral sheen there and the fab bake OR-in a real 4.4 µm 45° grating.
#
# Returns (side, side) bool for the FRONT centerpiece art or None if the slug
# has no designed accent (so pre-accent behaviour is byte-identical). Kept
# front-only: every plan accent lives on the front silhouette.
def _front_accent_zone(slug: str, art: "np.ndarray") -> "np.ndarray | None":
    import numpy as np

    h, w = art.shape
    ys = (np.arange(h)[:, None] + 0.5) / h  # 0..1, y-down
    xs = (np.arange(w)[None, :] + 0.5) / w

    def _ellipse(cx: float, cy: float, rx: float, ry: float) -> "np.ndarray":
        return ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0

    if slug == "colibri-globe-phase":
        # Gorget = the hummingbird's throat patch. The colibrí faces right with
        # its head/throat upper-right in the art box; a small ellipse there.
        zone = _ellipse(0.62, 0.40, 0.10, 0.07)
    elif slug == "food-pair-chirp":
        # Steam-curl TIPS: a thin band across the top of the steam column above
        # the cup (the cup sits left; steam rises to ~y<0.30).
        zone = (xs > 0.18) & (xs < 0.42) & (ys < 0.30)
    elif slug == "gear-quill-switch":
        # Gear HUB: the central boss/bore of the centered gear.
        zone = _ellipse(0.5, 0.5, 0.11, 0.11)
    elif slug == "monogram-jp":
        # Flourish TIPS: the outer swash ends of the interlocked script, which
        # reach the lower-left and upper-right corners of the glyph box.
        zone = _ellipse(0.17, 0.80, 0.12, 0.10) | _ellipse(0.83, 0.22, 0.12, 0.10)
    else:
        return None
    z = zone & art
    return z if z.any() else None


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
    masks = _centerpiece_masks(spec.pattern_slug, side_px, spec.pattern_params)
    if masks is None:
        return
    front_art, back_art = masks

    cx = plate_w // 2
    cy = plate_h // 2
    x0 = cx - side_px // 2
    y0 = cy - side_px // 2

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

    _paste(front, front_art, with_accent=True)

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
        from .patterns.artistic.capybara_scanimation import WATERLINE_Y as _WL
        below = np.broadcast_to(rows >= _WL, (side_px, aperture_px))
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

    central_manifest = materialize_pattern(spec.pattern_slug, merged_params)
    central_pitch = central_manifest["pixel_pitch_um"]

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

    # --- FRONT: perimeter foliage frame band ---------------------------------
    # The colonize band hugs the four edges (fill_interior=False), leaving the
    # center open for the art centerpiece. Painted at FRAME_LEVEL so the shader
    # tells frame foliage apart from the ART_LEVEL centerpiece.
    active_w, active_h = spec.active_dims()
    if active_w > 0 and active_h > 0:
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
    back_w, back_h = spec.back_dims()
    if back_w > 0 and back_h > 0:
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

    _zero_rim(front, spec.weld_margin_um)
    back_margin = spec.weld_margin_um if spec.back_margin_um is None else spec.back_margin_um
    _zero_rim(back, back_margin)

    front.save(out_dir / "front.png")
    back.save(out_dir / "back.png")
    thumb = make_thumbnail(front, back, size=320)
    thumb.save(out_dir / "thumbnail.png")

    # The frame scene (potentially MBs of segment dicts) goes to a sidecar,
    # NOT into manifest.json: embedding it made every plate manifest 0.4-2.3
    # MB and dominated both the warm box regen (6 × json decode) and the cold
    # compose's manifest encode. Nothing at runtime consumes it — the
    # frontend never reads it, export strips it, and ensure_plate_svg
    # regenerates the scene deterministically from the spec — so the sidecar
    # is purely for debugging/inspection.
    frame_scene = scene.to_dict() if scene is not None else {"segments": [], "flowers": [], "leaves": [], "max_t": 0.0}
    (out_dir / "scene.json").write_text(json.dumps(frame_scene))

    return {
        "central_manifest": central_manifest,
        "pixel_pitch_um": plate_pitch,
        "min_feature_um": central_manifest["min_feature_um"],
    }


def materialize_plate(spec: PlateSpec, force: bool = False) -> dict[str, Any]:
    """Compose the plate, cache PNG/SVG/manifest under ``data/plates/<hash>/``."""
    PLATES_ROOT.mkdir(parents=True, exist_ok=True)
    pid = plate_hash(spec)
    out = PLATES_ROOT / pid
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        cached = json.loads(manifest_path.read_text())
        _log.info("materialize_plate cache_hit id=%s slug=%s", pid, spec.pattern_slug)
        return cached

    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    raster_result = _raster_compose_plate(spec, out)
    central_manifest = raster_result["central_manifest"]
    pitch = raster_result["pixel_pitch_um"]

    # SVG is built on demand by the /export endpoint (see ensure_plate_svg)
    # — the polygon path is ~40× slower than raster and the interactive UI
    # never needs it. Manifest carries empty svg paths until requested.
    svg_ok = (out / "front.svg").exists() and (out / "back.svg").exists()

    central_cls = registry[spec.pattern_slug]
    manifest = {
        "kind": "plate",
        "id": pid,
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
            **central_manifest.get("extra", {}),
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
        "recipe_data": {**central_manifest.get("recipe_data", {}), **_carrier_recipe_data(spec)},
        "files": {
            "front_png": f"/data/plates/{pid}/front.png",
            "back_png": f"/data/plates/{pid}/back.png",
            "front_svg": f"/data/plates/{pid}/front.svg" if svg_ok else "",
            "back_svg": f"/data/plates/{pid}/back.svg" if svg_ok else "",
            "thumbnail": f"/data/plates/{pid}/thumbnail.png",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    dt_ms = int((time.perf_counter() - t0) * 1000)
    _log.info(
        "materialize_plate done id=%s slug=%s %dms pitch=%.3f svg=%s",
        pid,
        spec.pattern_slug,
        dt_ms,
        pitch,
        svg_ok,
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

    Returns (front, back) paths or None if the plate is unknown.
    """
    import numpy as np

    from .patterns._helpers import MAX_LATTICE_CELLS, check_lattice_budget, raster_to_polygons

    plate_dir = PLATES_ROOT / plate_id
    manifest_path = plate_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    front_svg = plate_dir / "front.svg"
    back_svg = plate_dir / "back.svg"
    if front_svg.exists() and back_svg.exists() and _svg_is_current(front_svg):
        return front_svg, back_svg

    manifest = json.loads(manifest_path.read_text())
    spec = PlateSpec.from_dict(manifest["spec"])
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
    ideal_pitch = min(back_period, front_period) / 4.0
    budget_pitch = math.sqrt(W * H / (0.92 * MAX_LATTICE_CELLS))
    pitch = max(ideal_pitch, budget_pitch)
    if pitch > ideal_pitch:
        scale = pitch / ideal_pitch
        back_period *= scale
        front_period *= scale
    fw = max(1, int(round(W / pitch)))
    fh = max(1, int(round(H / pitch)))
    check_lattice_budget(fw * fh, "plate grating raster", pitch_um=pitch, w=fw, h=fh)

    # Centerpiece stripe carrier (vertical): fab period + the half-period phase
    # offset baked into the back globe. Coarsen with the frame grating if the
    # budget forced a coarser pitch (same scale factor keeps the beat).
    center_period = float(rd.get("fab_center_period_um", rd["center_period_um"]))
    if pitch > ideal_pitch:
        center_period *= pitch / ideal_pitch
    center_axis = float(rd["switch_axis_deg"])
    is_interlace = spec.pattern_slug in SWITCH_INTERLACE_SLUGS
    aperture = _aperture(spec)
    side_px = max(8, int(round(CENTERPIECE_FILL * aperture / pitch))) if aperture > 0 else 0
    cxg = fw // 2
    cyg = fh // 2

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
    # frames — the genuine scanimation the box back plate actually serves. Both
    # are full-grid bool overrides consumed below (None for every other slug, so
    # their fab output is byte-identical).
    water_front_extra: "np.ndarray | None" = None
    water_back: "np.ndarray | None" = None
    water_band_placed: "np.ndarray | None" = None
    if spec.pattern_slug == WATER_SCAN_SLUG and side_px > 0:
        from .patterns.artistic import capybara_scanimation as _capyscan

        # WATER FULL WIDTH: build the water band / slit comb / interleave across
        # the full WIDTH-axis aperture (`_aperture_width_um`, edge-to-edge of the
        # window) while the capybara body + flow wake stay in the centered
        # CENTERPIECE_FILL square (open current flanks the animal).
        b = _capyscan._build(
            extent_um=CENTERPIECE_FILL * aperture,
            frame_pitch_um=WATER_SCAN_FAB_PITCH_UM,
            n_phases=WATER_SCAN_N_PHASES,
            carrier_period_um=WATER_SCAN_FAB_CARRIER_UM,
            waterline_y=_capyscan.WATERLINE_Y,
            n_grid=side_px,
            water_extent_um=_aperture_width_um(spec),
        )
        # FRONT: capybara body shimmer + slit-barrier bars over the water band.
        barrier_bars = (~b["barrier"]) & b["water_band"]
        front_side = b["capy_shimmer"] | barrier_bars
        # BACK: interleaved ripple frames (all N phases packed into 1/N slots).
        back_side = b["back_water"]

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
        # Full-width water BAND (below waterline, edge-to-edge) so the back
        # carrier can be cleared across the whole band, not just the body square.
        water_band_placed = _place_side(b["water_band"])

    # --- FRONT: perimeter foliage frame (grating) + centerpiece -------------
    active_w, active_h = spec.active_dims()
    front_group = ""
    if active_w > 0 and active_h > 0:
        rect = RectFrame(width_um=active_w, height_um=active_h)
        frame_params = spec.frame.to_frame_params()
        frame_params.fill_interior = False  # perimeter band, center open for art
        scene = generate_frame(rect, frame_params)
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
        #   * barrier-interlace (globe-duo / gear-quill): a NEUTRAL slit barrier
        #     (60 µm pitch, open duty 0.5 → one 30 µm lane) over the union of both
        #     silhouettes — the front layer carries NO image, only the barrier
        #     that gates which back lane class shows on tilt. Phase −0.25 aligns
        #     the open slot to straddle an A|B lane boundary head-on (matches the
        #     preview shader's slitBarCoverage phase 0.25 bar), so ± tilt reveals
        #     A or B cleanly.
        #   * legacy phase-switch: the FRONT silhouette filled with the vertical
        #     switch carrier (phase 0).
        # Both exclude the accent zone, which gets the sub-5 µm diffraction grating.
        if is_interlace:
            union_art = front_art | back_art
            barrier_bar = _grating_grid(fw, fh, pitch, center_period, 0.5, center_axis, phase=-0.25)
            art_carrier = union_art & barrier_bar & ~front_accent
        else:
            center_grating = _grating_grid(fw, fh, pitch, center_period, duty, center_axis)
            art_carrier = front_art & center_grating & ~front_accent
        front_grid = (sil & frame_grating) | art_carrier
        # Water scanimation: OR-in the capybara body shimmer + slit-barrier bars
        # over the water band (real barrier-grid geometry, not the stripe carrier).
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
        # effects.gratings.diffraction_accent_grating's integrator note.
        if front_accent.any():
            from .patterns.effects.gratings import (
                DIFFRACTION_ACCENT_PERIOD_UM,
                diffraction_accent_grating,
            )

            accent_max_pitch = DIFFRACTION_ACCENT_PERIOD_UM / 4.0
            if pitch <= accent_max_pitch:
                front_grid |= diffraction_accent_grating(front_accent, pitch)
            else:
                _log.info(
                    "skipping diffraction accent bake: raster pitch %.3g µm > "
                    "period/4 = %.3g µm (period %.3g µm would alias); accent "
                    "zones remain in PNG masks + fine-pitch export only",
                    pitch,
                    accent_max_pitch,
                    DIFFRACTION_ACCENT_PERIOD_UM,
                )
        front_polys = raster_to_polygons(front_grid, pitch, (W, H))
        front_group = _group("frame+centerpiece", _strip_svg_body(to_svg(front_polys, (W, H), background=None)))

    # --- BACK: uniform carrier grating + globe centerpiece (phase π) --------
    back_group = ""
    back_w, back_h = spec.back_dims()
    if back_w > 0 and back_h > 0:
        win = np.zeros((fh, fw), dtype=bool)
        bw = max(1, int(round(back_w / pitch)))
        bh = max(1, int(round(back_h / pitch)))
        bx = (fw - bw) // 2
        by = (fh - bh) // 2
        win[by : by + bh, bx : bx + bw] = True
        back_margin = spec.weld_margin_um if spec.back_margin_um is None else spec.back_margin_um
        _mask_rim(win, back_margin, pitch)
        carrier_grating = _grating_grid(fw, fh, pitch, back_period, duty, base_angle)
        if is_interlace:
            # Barrier-interlace back layer: BOTH silhouettes interleaved in
            # alternating lanes (lane pitch = half the barrier pitch). A (front
            # silhouette) fills the even lanes, B (back silhouette) the odd. The
            # even-lane mask is a 0.5-duty grating at the barrier period (one lane
            # of the pair = gold half); its complement is the odd lane. This is
            # the whole moving image — the front barrier reveals one lane class per
            # tilt. The plain carrier is cleared across the union so the lanes read.
            union_art = front_art | back_art
            even_lane = _grating_grid(fw, fh, pitch, center_period, 0.5, center_axis, phase=0.0)
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
        # and OR-in the real ripple geometry (clipped to the back window rim).
        if water_back is not None:
            band = water_band_placed if water_band_placed is not None else back_art
            back_grid = (win & carrier_grating & ~band) | (win & water_back)
        back_polys = raster_to_polygons(back_grid, pitch, (W, H))
        back_group = _group("carrier+centerpiece", _strip_svg_body(to_svg(back_polys, (W, H), background=None)))

    front_svg.write_text(_wrap_svg(W, H, [front_group]), encoding="utf-8")
    back_svg.write_text(_wrap_svg(W, H, [back_group]), encoding="utf-8")

    manifest["files"]["front_svg"] = f"/data/plates/{plate_id}/front.svg"
    manifest["files"]["back_svg"] = f"/data/plates/{plate_id}/back.svg"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return front_svg, back_svg


# Bump when the SVG compose geometry changes: cached plate SVGs are only
# reused if they carry the current marker, so a formula fix (e.g. the
# aperture-scaling fix) invalidates stale files under unchanged spec hashes.
PLATE_SVG_VERSION = "plate-svg-v3"


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
            try:
                out.append(json.loads(m.read_text()))
            except Exception:  # noqa: BLE001 — skip corrupt manifests
                continue
    return out


def get_plate(plate_id: str) -> dict[str, Any] | None:
    m = PLATES_ROOT / plate_id / "manifest.json"
    if not m.exists():
        return None
    return json.loads(m.read_text())
