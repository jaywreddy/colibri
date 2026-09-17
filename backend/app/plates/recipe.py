"""How a face is DRAWN: the mask level palette, the grating constants, and
``_carrier_recipe_data`` — the shader contract every composed plate binds.

Every number here is a design decision with its reasoning attached; none of it
touches the filesystem. ``_carrier_recipe_data`` is the one function the
renderer, the fab writers and the witness plate all read, which is why it (and
the constants it reads) is a module of its own rather than a section of the
compositor.
"""
from __future__ import annotations

from typing import Any

from ..patterns.effects.gratings import beat_delta_um
from .spec import FaceKind, PlateSpec, _aperture

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
# DIFFRACTIVE graylevel. Slots into the free gap between the frame-bucket top
# (166 = 0.651) and ART_LEVEL (255): the shader carves a third window
# RAINBOW = [0.72, 0.86) (RAINBOW_MIN/ART_MIN in plate.frag) and treats these
# pixels as gold that also carries a fine sub-grating, so they flash spectral
# colour rather than plain specular metal.
#
# On the production box this level marks the two single-layer constructions the
# faces actually use: a ``region_art`` centrepiece's DIFFRACTIVE regions
# (``_paste_centerpiece``) and a photo screen's COLOURED bands
# (``_photo_band_stamp``). The period behind each such pixel is not in the mask
# PNG at all — it rides ``period_front.png`` (see ``_literal_period_raster``),
# which is what lets one graylevel stand for a whole ladder of periods.
#
# (Until 2026-09-16 the same level also marked small hand-placed "diffraction
# accent" patches — a gear hub, steam-curl tips — cut out of a two-ply
# centrepiece silhouette and OR-ed with a fixed 4.4 µm 45° grating at bake time.
# Those faces are gone with the two-ply optics; the level now means exactly
# "this gold carries a sub-grating, look up its period".)
RAINBOW_LEVEL = 200

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
# treatments. (The readability gate that measured this went with the 2D lab on
# 2026-09-16; the identity behind it is pinned in tests/test_optics_math.py.)
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
#
# HIDDEN EXEMPLAR (2026-09-16): no production face is two-ply, so this path is
# only entered by ``monogram-jp`` composed with ``single_ply=False`` — the one
# shading-moiré exemplar kept so the construction stays exercised. On the
# production lid the same slug is a ``region_art`` centrepiece and never reads
# ``fab_center_period_um`` at all.
CENTERPIECE_BEAT_UM = 1635.0
SHIMMER_MOIRE_SLUGS = frozenset({"monogram-jp"})
# FAB centerpiece stripe period. The A<->B (colibrí<->globe) switch happens
# after parallax walks HALF a period; at t=500 µm, n=1.46 the parallax is
# ~5.98 µm/deg, so a 60 µm period switches at ~5° tilt — squarely inside the
# comfortable 3-8° hand-tilt sweet spot the effects toolkit recommends (vs the
# ~14° "stiff" switch a 170 µm fab period would need). The preview stripe scale
# stays coarse (170 µm) for a visible louvre; the physical switch timing lives
# in this separate fab period, exactly like the frame preview/fab period split.
FAB_CENTER_PERIOD_UM = 60.0

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
#
# HIDDEN EXEMPLAR (2026-09-16): no production face is a barrier switch either.
# ``globe-duo-phase`` is the one kept so the construction — and the registration
# solved in ``_barrier_plate_lattice`` / ``_barrier_masks`` — stays exercised.
SWITCH_INTERLACE_SLUGS = frozenset({"globe-duo-phase"})

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
# byte-identical geometry. The FINE-pitch families (the 22 µm louvre carrier,
# the 4-6 µm diffractive ladders) deliberately do NOT scale: on thick stock
# their reveals tighten into sub-degree refraction shimmer — a chosen look, and
# the 2 µm litho floor is the binding constraint in the other direction.
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


def _carrier_recipe_data(spec: PlateSpec) -> dict[str, Any]:
    """Per-plate moiré-carrier knobs the shader + fab path both read.

    The FRAME grating pair (perimeter foliage shimmer): the front louvre is
    rotated by ``CARRIER_ANGLE_OFFSET_DEG`` relative to the back and scaled by
    ``FRONT_GRATING_RATIO``; a per-face base angle keyed off the frame seed
    keeps each face's carrier visually distinct without changing the physics.

    The CENTERPIECE carrier: a finer vertical stripe carrier with a ±X switch
    axis. On the barrier-interlace face (SWITCH_INTERLACE_SLUGS) the shader
    draws the neutral comb + interleaved lanes at this pitch instead; on the
    shading-moiré face (SHIMMER_MOIRE_SLUGS) it is the glimmer carrier. Both
    are two-ply exemplars — no production face reads either.
    """
    kind = spec.kind
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
    # carrier litho limit for very thin stock.
    if getattr(spec, "carrier_scale_mode", "gap") != "fixed":
        carrier_pitch = max(4.0, _snap_half_um(carrier_pitch * parallax_period_scale(spec)))
    front_pitch = carrier_pitch * FRONT_GRATING_RATIO
    # Preview periods (what the shader actually draws) — magnified so even the
    # 4 µm litho-floor pitch resolves on the two real planes; see
    # PREVIEW_PITCH_MAGNIFY. Proportional to the real pitch, so finer pitch reads
    # as finer fringes.
    preview_carrier = carrier_pitch * PREVIEW_PITCH_MAGNIFY
    preview_front = preview_carrier * FRONT_GRATING_RATIO
    # ART-BOX registration. The centerpiece is a SQUARE of side
    # (CENTERPIECE_FILL·aperture) centered on the plate; on a non-square plate it
    # maps to different uv half-extents per axis. The shader transforms face-uv →
    # art-box uv with these so the centerpiece constructions (today: the barrier
    # comb of the switch exemplar) are gated to the art box rather than stretched
    # across the whole face. The recipe KEY NAMES are ``water_art_*``, which is
    # where they come from — they were introduced for the capybara scanimation's
    # flow wake and outlived it; the renderer binds them by those names.
    art_side_um = CENTERPIECE_FILL * _aperture(spec)
    art_half_w_uv = (0.5 * art_side_um / spec.width_um) if spec.width_um > 0 else 0.0
    art_half_h_uv = (0.5 * art_side_um / spec.height_um) if spec.height_um > 0 else 0.0

    # Centerpiece fill. All three consumers -- the preview shader, the SVG bake
    # (ensure_plate_svg) and the fine export -- read exactly these two fields,
    # so setting them here is the whole change for a face's centerpiece grating.
    #
    # A shimmer face gets the shading-moire pair: parallel to the back carrier,
    # mismatched by CENTERPIECE_BEAT_DELTA_UM. Every other face keeps the
    # glass-derived barrier/comb pitch on the +X axis, because there the pitch
    # and axis set a SWITCH angle rather than a beat and must not be retuned for
    # fringe aesthetics.
    # WHICH two-ply exemplar, not WHICH construction: a slug-set test on
    # purpose (see FaceKind). Deliberately NOT gated on ``kind is TWO_PLY`` —
    # the production lid is the same slug on one ply, and narrowing this would
    # change the ``fab_center_period_um``/``switch_axis_deg`` it has always
    # published (unread there, but published) for no gain.
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
        "blank": kind is FaceKind.BLANK,
        # SOLID GOLD: the whole outer ply is metal (patterns/solid.py).
        "solid": kind is FaceKind.SOLID,
        # SOLID ART: the centerpiece is a line-screen picture whose bands are
        # already the tone, so the shader must fill ART_LEVEL with plain gold
        # instead of running the procedural switch carrier over it — a second
        # grating on top would multiply the halftone's duty and wash the picture
        # out. (RAINBOW_LEVEL bands keep their diffraction sheen; those ARE the
        # colour sub-grating.)
        "art_solid": kind in (FaceKind.PHOTO, FaceKind.SOLID, FaceKind.REGION),
        # SINGLE-LAYER DIFFRACTION centrepiece (region_art): the motif is a map
        # of regions each written as a vertical grating at its own period
        # (colour by region) or solid gold. Its periods ride ``period_front``
        # like the photo colour zones do; ``literal_front`` carries the metal.
        "region_art": kind is FaceKind.REGION,
        # WHICH of the five constructions wrote this plate (FaceKind), decided
        # once on the spec and published so a reader of the manifest does not
        # have to re-derive it from the slug. Inside the compose fingerprint's closure.
        "face_kind": kind.value,
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
        # ART-BOX uv rect (see above). The key names are historical — they were
        # introduced for the capybara scanimation's flow wake, which is gone —
        # but the RECT is not: it is how the renderer gates a centerpiece
        # construction to the art box. (halfW, halfH), then (centerX, centerY);
        # the centre is the plate centre because the centerpiece paste is
        # plate_w//2 / plate_h//2.
        "water_art_half_uv": [float(art_half_w_uv), float(art_half_h_uv)],
        "water_art_center_uv": [0.5, 0.5],
        # DIFFRACTIVE graylevel. The shader lights ONLY pixels at exactly this
        # level, so a face whose centerpiece writes none renders identically —
        # safe to emit unconditionally. The PERIOD behind each such pixel rides
        # ``period_front.png`` (see RAINBOW_LEVEL and _literal_period_raster),
        # not this manifest: one level, a whole ladder of periods.
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
        # Fab centerpiece stripe period — GLASS-DERIVED so the ~5° switch
        # crossing holds on any stock (see fab_center_period_um).
        "fab_center_period_um": center_period,
    }
