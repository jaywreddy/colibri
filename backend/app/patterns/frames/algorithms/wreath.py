"""Ordered garland wreath anchored to a rectangular perimeter.

Where ``colonize`` grows an organic (but *scattered*) mat by space
colonization, this generator lays a DELIBERATE wreath: a continuous guiding
vine spline hugging the frame, with leaves attached in ordered ranks ALONG the
spline — shingled, alternating sides, tips trailing with the flow — the way a
classic laurel / botanical wreath engraving reads.

Composition, edge by edge:

  * Each edge carries a guiding spline that runs corner-to-corner at a base
    depth inside the rim, with a gentle value-noise S-undulation so it reads
    hand-drawn, not machined.
  * The spline flows OUTWARD from each edge midpoint toward the two corners
    (mirror symmetry about every edge midpoint) — the deliberate symmetry a
    wreath has. Leaves trail their tips WITH that flow (away from the midpoint),
    so both halves sweep toward the corners like laurel.
  * Leaves attach at evenly-spaced arc-length stations, alternating outboard /
    inboard side, each rotated to lie shingle-like along the vine with ~30 %
    overlap between successive leaves. Size swells toward edge centers and
    corners (a rank is fuller there) and is otherwise consistent.
  * Corners are resolved on purpose: the two edge splines converge into a
    knotted crossing of stems, capped by a corner medallion — a small radiating
    fan of fronds plus an accent bloom.

Everything is emitted as the same renderer-agnostic ``Scene`` the colonize path
produces (Segments + LeafSprite/FlowerSprite carrying the existing motif ``type``
strings), so the plate compositor's per-species angle-bucket painting keeps
working unchanged — no coordination needed beyond "the type strings still exist".

Three named compositions are exposed via ``wreath_style``:

  * ``laurel``   — dense single-species rank, tight overlap, restrained blooms.
  * ``garland``  — mixed tropical foliage, looser rank, more blooms + clusters.
  * ``clusters`` — sparse vine punctuated by deliberate leaf+bloom rosettes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..geometry import Mulberry32, RectFrame, ValueNoise2D
from ..scene import FlowerSprite, LeafSprite, Scene, Segment


# --- style presets -----------------------------------------------------------
# Each preset tunes the rank spacing, leaf sizing, overlap, bloom frequency and
# whether the fill is a continuous rank or discrete clusters. Values are chosen
# so all three read as coherent wreaths, differing in character not quality.
@dataclass(frozen=True)
class _Style:
    name: str
    leaf_spacing_frac: float   # station spacing as a fraction of the shorter side
    leaf_len_frac: float       # leaf length as a fraction of the shorter side
    leaf_lean_deg: float       # how far a leaf tilts off the vine tangent
    swell: float               # extra size at edge-centers/corners (0..1)
    bloom_every: int           # place a bloom every Nth station (0 = none along rank)
    single_species: bool       # laurel-style: one leaf species for the whole rank
    cluster_mode: bool         # discrete rosettes instead of a continuous rank
    understory: float          # small backing-leaf pass density (0 disables)
    inner_tendrils: float      # tiny inner-edge tendril curls (0 disables)
    # --- lush mixed-tropical extras (garland2) ---
    mixed_rank: bool = False   # interleave broad species + periodic feathery/accent
    feathery_every: int = 0    # drop a feathery frond every Nth station (0 = none)
    accent_leaf_every: int = 0 # drop a big accent blade every Nth station
    size_jitter: float = 0.08  # ± fractional scale jitter per leaf (~30% -> 0.30)
    inner_rank: float = 0.0    # a second rank leaning toward the aperture (fills depth)
    understory_depth: float = 0.30  # how far outboard the understory ribbon sits (band frac)


STYLES: dict[str, _Style] = {
    "laurel": _Style(
        name="laurel",
        leaf_spacing_frac=0.028,
        leaf_len_frac=0.070,
        leaf_lean_deg=42.0,
        swell=0.35,
        bloom_every=0,
        single_species=True,
        cluster_mode=False,
        understory=0.5,
        inner_tendrils=0.0,
    ),
    "garland": _Style(
        name="garland",
        leaf_spacing_frac=0.034,
        leaf_len_frac=0.070,
        leaf_lean_deg=48.0,
        swell=0.40,
        bloom_every=5,
        single_species=False,
        cluster_mode=False,
        understory=0.55,
        inner_tendrils=0.35,
    ),
    "clusters": _Style(
        name="clusters",
        leaf_spacing_frac=0.115,   # rosette spacing (sparser)
        leaf_len_frac=0.078,
        leaf_lean_deg=58.0,
        swell=0.30,
        bloom_every=1,             # a bloom at every rosette
        single_species=False,
        cluster_mode=True,
        understory=0.35,
        inner_tendrils=0.5,
    ),
    # LUSH MIXED-TROPICAL GARLAND — the restored default. Keeps the laurel's
    # ordered guiding vine, outward flow and resolved corners, but brings back
    # the diversity + fill of the old colonize band: the main rank interleaves
    # broad species (philodendron/plantain) with periodic feathery fronds
    # (fern/wax_palm) and oversized accent blades, blooms nest into the rank at
    # intervals, leaf scale varies ~30 %, a fine feathery understory ribbon
    # packs the outer sliver, and a second inner rank leans toward the aperture
    # so the WHOLE band depth is filled (not just a single shingled centreline).
    # Because species drive the plate's angle-bucket graylevel, mixing species
    # automatically restores the multi-direction moiré shimmer laurel flattened.
    "garland2": _Style(
        name="garland2",
        leaf_spacing_frac=0.045,   # OPEN garland: distinct leaves, breathing gaps
        leaf_len_frac=0.072,
        leaf_lean_deg=46.0,        # graceful outward-forward sweep off the vine
        swell=0.20,                # gentle end-fullness (corners not overloaded)
        bloom_every=6,             # sparse jewel blooms at intervals
        single_species=False,
        cluster_mode=False,
        understory=0.32,           # light feathery sprigs, not a solid ribbon
        inner_tendrils=0.28,
        mixed_rank=True,
        feathery_every=5,          # feathery sprig accents in the rank
        accent_leaf_every=6,       # an oversized accent blade at intervals
        size_jitter=0.22,
        inner_rank=0.0,            # no fill rank — keep the band open + legible
        understory_depth=0.40,
    ),
}
DEFAULT_STYLE = "garland2"


# --- spline helpers ----------------------------------------------------------


def _catmull(p0, p1, p2, p3, t):
    """Centripetal-ish Catmull-Rom interpolation of x or y at parameter t."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


@dataclass
class _Sample:
    x: float
    y: float
    tx: float   # unit tangent (direction of travel)
    ty: float
    nx: float   # unit inward normal
    ny: float
    s: float    # arc length from the spline start


def _resample_polyline(pts: list[tuple[float, float]], step: float) -> list[_Sample]:
    """Turn a control polyline into arc-length samples with tangent + normal.

    The garland spline is defined by a handful of control points (edge start,
    undulation waypoints, edge end). We Catmull-Rom through them, then walk the
    resulting curve at a fixed ``step`` so leaves can be dropped at even
    arc-length stations. Inward normal is the tangent rotated +90° toward the
    rectangle interior (callers pass control points ordered so this holds).
    """
    if len(pts) < 2:
        return []
    # Pad ends for Catmull-Rom endpoint tangents.
    ctrl = [pts[0]] + pts + [pts[-1]]
    dense: list[tuple[float, float]] = []
    per_seg = 14
    for i in range(1, len(ctrl) - 2):
        p0, p1, p2, p3 = ctrl[i - 1], ctrl[i], ctrl[i + 1], ctrl[i + 2]
        for k in range(per_seg):
            t = k / per_seg
            x = _catmull(p0[0], p1[0], p2[0], p3[0], t)
            y = _catmull(p0[1], p1[1], p2[1], p3[1], t)
            dense.append((x, y))
    dense.append(pts[-1])

    # Walk the dense polyline at fixed arc-length step.
    out: list[_Sample] = []
    acc = 0.0
    next_s = 0.0
    for i in range(len(dense) - 1):
        x0, y0 = dense[i]
        x1, y1 = dense[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-9:
            continue
        while next_s <= acc + seg:
            f = (next_s - acc) / seg
            x = x0 + (x1 - x0) * f
            y = y0 + (y1 - y0) * f
            tx = (x1 - x0) / seg
            ty = (y1 - y0) / seg
            out.append(_Sample(x=x, y=y, tx=tx, ty=ty, nx=0.0, ny=0.0, s=next_s))
            next_s += step
        acc += seg
    return out


# --- main generator ----------------------------------------------------------


def generate(
    rect: RectFrame,
    *,
    seed: int = 1,
    density: float = 1.0,
    bloom: float = 0.6,
    foliage: float = 0.6,
    band_um: float | None = None,
    flower_types: Sequence[str] = ("orchid",),
    leaf_types: Sequence[str] = ("fern",),
    max_iter: int = 500,           # unused; kept for signature parity
    fill_interior: bool = False,   # unused here (whole-face fill stays colonize)
    edge_gradient: float = 1.0,    # unused; band-shape knobs below supersede
    understory: float = 1.0,
    border_vine: float = 1.0,
    corner_fans: float = 1.0,
    wreath_style: str = DEFAULT_STYLE,
) -> Scene:
    """Lay an ordered garland wreath and return a Scene.

    ``band_um`` defaults to ~13 % of the shorter side (matching the frame
    default). The guiding vine rides at ~45 % band depth; leaves fan out over
    most of the band width. Style presets (``wreath_style``) select the overall
    character; ``density``/``bloom``/``foliage`` scale spacing, blooms and the
    understory within a style. ``understory``/``border_vine``/``corner_fans``
    are honored as multiplicative knobs so the existing per-face FrameParams
    still steer the look.
    """
    short = min(rect.width_um, rect.height_um)
    if band_um is None:
        band_um = 0.13 * short
    style = STYLES.get(wreath_style, STYLES[DEFAULT_STYLE])

    rng = Mulberry32(seed)
    noise = ValueNoise2D(seed ^ 0x5F356495)
    scene = Scene()

    leaf_types = list(leaf_types) or ["fern"]
    flower_types = list(flower_types) or ["orchid"]
    # Laurel wants a single broad species so the rank reads uniform; prefer a
    # broad blade if the theme has one, else the first leaf.
    broad_pref = [t for t in leaf_types if t in ("philodendron", "plantain")]
    laurel_species = (broad_pref or leaf_types)[0]

    # Vine gauge — a firm drawn stem, >= a few raster px at plate pitch.
    base_w = 0.0050 * short
    max_w = 0.0092 * short

    # The guiding vine rides at this depth in from the rim; leaves splay across
    # most of the band. Corner inset keeps the spline clear of the sharp corner
    # so the medallion owns it.
    ride_depth = band_um * 0.46
    corner_inset = band_um * 1.15   # arc-length gap left at each corner

    # --- build the four edge garland splines ---------------------------------
    # Each edge spline flows OUTWARD from the edge midpoint toward both corners
    # (mirror symmetry). We build it as TWO half-splines (mid->cornerA,
    # mid->cornerB) so leaf flow direction is trivially "away from midpoint".
    edges = _edge_specs(rect)
    step = style.leaf_spacing_frac * short * (1.0 / max(0.35, density))

    for edge in edges:
        _lay_edge(
            scene, rect, edge, noise, rng, short, band_um, ride_depth,
            corner_inset, base_w, max_w, step, style, leaf_types,
            flower_types, laurel_species, bloom, foliage,
            understory, border_vine,
        )

    # --- corners: knotted crossing stems + medallion -------------------------
    if corner_fans > 0.0:
        _lay_corners(
            scene, rect, noise, rng, short, band_um, ride_depth,
            base_w, max_w, style, leaf_types, flower_types, laurel_species,
            bloom, corner_fans,
        )

    scene.max_t = 1.0
    return scene


# --- edge specification ------------------------------------------------------


@dataclass
class _Edge:
    # midpoint of the edge (on the rim), inward normal, tangent toward corner B,
    # and the two corner points.
    mx: float
    my: float
    nx: float
    ny: float
    tx: float
    ty: float
    half_len: float
    key: str


def _edge_specs(rect: RectFrame) -> list[_Edge]:
    w, h = rect.width_um, rect.height_um
    L, R, B, T = rect.left, rect.right, rect.bottom, rect.top
    return [
        _Edge(mx=0.0, my=B, nx=0.0, ny=1.0, tx=1.0, ty=0.0,
              half_len=w / 2.0, key="bottom"),
        _Edge(mx=R, my=0.0, nx=-1.0, ny=0.0, tx=0.0, ty=1.0,
              half_len=h / 2.0, key="right"),
        _Edge(mx=0.0, my=T, nx=0.0, ny=-1.0, tx=-1.0, ty=0.0,
              half_len=w / 2.0, key="top"),
        _Edge(mx=L, my=0.0, nx=1.0, ny=0.0, tx=0.0, ty=-1.0,
              half_len=h / 2.0, key="left"),
    ]


def _lay_edge(
    scene: Scene,
    rect: RectFrame,
    edge: _Edge,
    noise: ValueNoise2D,
    rng: Mulberry32,
    short: float,
    band_um: float,
    ride_depth: float,
    corner_inset: float,
    base_w: float,
    max_w: float,
    step: float,
    style: _Style,
    leaf_types: Sequence[str],
    flower_types: Sequence[str],
    laurel_species: str,
    bloom: float,
    foliage: float,
    understory: float,
    border_vine: float,
) -> None:
    """Lay both mirror halves of one edge: spline, ranked leaves, understory."""
    # Two halves: dir = +1 travels mid -> corner along +tangent, dir = -1 the
    # mirror. Each half is an independent spline so its leaves trail outward.
    for direction in (+1.0, -1.0):
        _lay_half_edge(
            scene, rect, edge, direction, noise, rng, short, band_um,
            ride_depth, corner_inset, base_w, max_w, step, style,
            leaf_types, flower_types, laurel_species, bloom, foliage,
            understory, border_vine,
        )


def _lay_half_edge(
    scene, rect, edge, direction, noise, rng, short, band_um, ride_depth,
    corner_inset, base_w, max_w, step, style, leaf_types, flower_types,
    laurel_species, bloom, foliage, understory, border_vine,
) -> None:
    # Build control points from the midpoint out to (corner - inset).
    span = edge.half_len - corner_inset
    if span < step * 1.5:
        return
    tx, ty = edge.tx * direction, edge.ty * direction
    nx, ny = edge.nx, edge.ny

    # Undulation: the guiding line breathes between ride_depth ± amplitude.
    # A deliberate single graceful bow per half-edge (the vine dips toward the
    # aperture at mid-span then sweeps back out to the corner) dominates, with a
    # little value-noise on top so it reads hand-drawn rather than machined.
    amp = band_um * 0.20
    n_ctrl = max(4, int(span / (0.10 * short)) + 2)
    ctrl: list[tuple[float, float]] = []
    for i in range(n_ctrl + 1):
        f = i / n_ctrl
        along = f * span
        u = noise.fbm((edge.key.__hash__() & 255) * 0.3 + along * 0.0016, 7.3,
                      octaves=2)
        wob = (u - 0.5) * 2.0 * amp * 0.55 + math.sin(f * math.pi) * amp * 0.9
        depth = ride_depth + wob
        depth = max(band_um * 0.14, min(band_um * 0.86, depth))
        x = edge.mx + tx * along + nx * depth
        y = edge.my + ty * along + ny * depth
        ctrl.append((x, y))

    samples = _resample_polyline(ctrl, step)
    if len(samples) < 2:
        return

    # Fill in inward-normal for each sample from its tangent (rotate tangent
    # -90° * winding to point inward). We know the rim inward normal (edge.n);
    # pick the tangent-perpendicular that best aligns with it.
    for smp in samples:
        # two perpendiculars
        px, py = -smp.ty, smp.tx
        if px * nx + py * ny < 0:
            px, py = smp.ty, -smp.tx
        smp.nx, smp.ny = px, py

    # --- stroke the guiding vine (tapering toward the corner tip) ---
    if border_vine > 0.0:
        for i in range(len(samples) - 1):
            a, b = samples[i], samples[i + 1]
            f = a.s / max(samples[-1].s, 1e-6)
            # thick near the midpoint, tapering out toward the corner
            w = max_w * (1.0 - 0.45 * f) * min(1.0, 0.7 + 0.5 * border_vine)
            scene.segments.append(Segment(x1=a.x, y1=a.y, x2=b.x, y2=b.y,
                                          w=max(base_w, w), t=1.0))

    # --- ranked leaves along the spline ---
    total_s = samples[-1].s
    leaf_len_base = style.leaf_len_frac * short * (0.9 + 0.2 * foliage)
    lean = math.radians(style.leaf_lean_deg)

    if style.cluster_mode:
        _rank_clusters(scene, samples, rng, style, leaf_types, flower_types,
                       leaf_len_base, lean, bloom, short, band_um)
    else:
        _rank_continuous(scene, samples, rng, style, leaf_types, flower_types,
                         laurel_species, leaf_len_base, lean, bloom, total_s)

    # --- understory: a fine backing rank tucked just outboard of the vine ---
    if understory > 0.0 and style.understory > 0.0:
        _rank_understory(scene, samples, rng, style, leaf_types, short,
                         band_um, understory * style.understory, foliage)

    # --- inner rank: a second rank of leaves leaning toward the aperture ---
    # This is what restores the FILLED band DEPTH the old colonize band had:
    # the main rank + understory pack the outer half, this pass reaches the
    # inner half so gold spans the whole band instead of a single centreline.
    if style.inner_rank > 0.0 and not style.cluster_mode:
        _inner_rank(scene, samples, rng, style, leaf_types, flower_types,
                    leaf_len_base, lean, bloom, total_s, band_um)

    # --- inner tendril curls reaching gently toward the aperture ---
    if style.inner_tendrils > 0.0:
        _inner_tendrils(scene, samples, rng, short, band_um, base_w,
                        style.inner_tendrils, rect)


def _rank_continuous(scene, samples, rng, style, leaf_types, flower_types,
                     laurel_species, leaf_len_base, lean, bloom, total_s):
    """Shingled leaves alternating side, tips trailing WITH the flow.

    Laurel mode lays a single broad species uniformly. The mixed garland2 mode
    instead INTERLEAVES species along the rank — mostly broad blades, but a
    feathery frond every ``feathery_every`` stations and an oversized accent
    blade every ``accent_leaf_every`` — so the rank reads as diverse tropical
    foliage (and, because species map to plate angle-buckets, each shimmers in
    its own moiré direction). Scale varies by ``size_jitter`` (±30 %).
    """
    n = len(samples)
    # Broad blades shingle into a clean silhouette; feathery species (fern /
    # wax_palm) read as texture accents. Split the theme leaves into the two
    # groups; fall back to the whole set if a group is empty.
    broad = [t for t in leaf_types if t not in ("fern", "wax_palm")] or list(leaf_types)
    feathery = [t for t in leaf_types if t in ("fern", "wax_palm")] or list(leaf_types)
    jit = max(0.02, style.size_jitter)
    for idx, smp in enumerate(samples):
        # alternate outboard / inboard so the rank reads two-ranked like laurel
        side = 1.0 if (idx % 2 == 0) else -1.0
        # Swell toward the two ends of this half-spline (midpoint & corner) so
        # the rank is fuller at edge-centers and corners.
        f = smp.s / max(total_s, 1e-6)
        end_bias = 1.0 + style.swell * (math.cos(f * math.pi) ** 2)  # ends fuller

        tang = math.atan2(smp.ty, smp.tx)
        # trailing shingle: each leaf sweeps off the vine toward its side by
        # `lean`, tip trailing WITH the flow, so the rank reads as a graceful
        # alternating laurel sweep with distinct, legible blades.
        angle = tang + side * lean

        # --- pick species + size for this station -------------------------------
        if style.single_species:
            ltype = laurel_species
            size = leaf_len_base * end_bias * rng.uniform(1.0 - jit, 1.0 + jit)
        elif style.mixed_rank:
            # Periodic accents take priority: an oversized broad blade, then a
            # feathery frond, else a normal broad blade.
            if style.accent_leaf_every and (idx % style.accent_leaf_every == 0) and idx > 0:
                ltype = rng.choice(broad)
                size = leaf_len_base * end_bias * 1.35 * rng.uniform(1.0 - jit, 1.0 + jit)
            elif style.feathery_every and (idx % style.feathery_every == 0):
                ltype = rng.choice(feathery)
                # feathery fronds read longer + thinner; give them extra reach.
                # Force them OUTBOARD (away from the aperture) and lying flat
                # along the vine so they stay body texture, never inner-edge spikes.
                side = -1.0
                size = leaf_len_base * end_bias * 0.98 * rng.uniform(1.0 - jit, 1.0 + jit)
                angle = tang + side * (lean * 0.55)
            else:
                ltype = rng.choice(broad)
                size = leaf_len_base * end_bias * rng.uniform(1.0 - jit, 1.0 + jit)
        else:
            ltype = rng.choice(broad)
            size = leaf_len_base * end_bias * rng.uniform(1.0 - jit, 1.0 + jit)

        off = size * 0.10
        bx = smp.x + smp.nx * (side * off)
        by = smp.y + smp.ny * (side * off)
        scene.leaves.append(LeafSprite(
            x=bx, y=by, angle=angle, size=size, type=ltype, t=1.0,
            seed=rng.next_uint32(),
        ))

        # occasional bloom nested into the rank. In mixed garland mode the blooms
        # are the accent jewels — nudge them a touch INBOARD (toward the open
        # aperture) and size them up so they read PROUD of the dense canopy
        # instead of drowning under it.
        if style.bloom_every and (idx % style.bloom_every == 0) and idx > 0:
            if rng.next_float() < (0.35 + 0.35 * bloom):
                # Jewel scale: a bloom is punctuation, not a subject. Nudge it
                # inboard so it sits PROUD of the canopy against open aperture.
                inb = 0.34 if style.mixed_rank else 0.12
                bsz = 0.46 if style.mixed_rank else 0.42
                scene.flowers.append(FlowerSprite(
                    x=smp.x + smp.nx * (size * inb),
                    y=smp.y + smp.ny * (size * inb),
                    size=size * bsz,
                    t=1.0, type=rng.choice(flower_types),
                    rot=tang, seed=rng.next_uint32(),
                ))


def _rank_clusters(scene, samples, rng, style, leaf_types, flower_types,
                   leaf_len_base, lean, bloom, short, band_um):
    """Discrete rosettes: a small fan of leaves + a bloom at spaced stations."""
    n = len(samples)
    for idx, smp in enumerate(samples):
        tang = math.atan2(smp.ty, smp.tx)
        n_leaves = 3
        for k in range(n_leaves):
            frac = (k / (n_leaves - 1)) - 0.5   # -0.5..0.5
            side = 1.0 if frac >= 0 else -1.0
            angle = tang + frac * 2.0 * lean
            size = leaf_len_base * (0.85 + 0.3 * (1.0 - abs(frac) * 2.0)) \
                * rng.uniform(0.9, 1.1)
            scene.leaves.append(LeafSprite(
                x=smp.x, y=smp.y, angle=angle, size=size,
                type=rng.choice(leaf_types), t=1.0, seed=rng.next_uint32(),
            ))
        # accent bloom sits at the rosette center, lifted slightly outboard
        scene.flowers.append(FlowerSprite(
            x=smp.x + smp.nx * (leaf_len_base * 0.10),
            y=smp.y + smp.ny * (leaf_len_base * 0.10),
            size=leaf_len_base * 0.66 * (0.85 + 0.3 * bloom),
            t=1.0, type=rng.choice(flower_types), rot=tang,
            seed=rng.next_uint32(),
        ))


def _rank_understory(scene, samples, rng, style, leaf_types, short, band_um,
                     density, foliage):
    """A fine backing rank of small leaves tucked OUTBOARD of the guiding vine.

    These fill the sliver between the vine and the rim so the wreath has a
    fuller back edge, biased to feathery species so they read as texture, not
    a second rank of subjects.
    """
    feathery = [t for t in leaf_types if t in ("fern", "wax_palm")] or leaf_types
    small = min(band_um * 0.34, 0.034 * short)
    stride = max(1, int(round(1.0 / max(0.15, density))))
    depth_frac = getattr(style, "understory_depth", 0.30)
    for idx in range(0, len(samples), stride):
        smp = samples[idx]
        tang = math.atan2(smp.ty, smp.tx)
        # tuck outboard: opposite the inward normal, out toward the rim, with a
        # little scatter in depth so the ribbon has body rather than a single line.
        out = small * (0.5 + rng.uniform(0.0, band_um * depth_frac / max(small, 1e-6)))
        # clamp the outboard reach to the understory depth band.
        out = min(out, band_um * depth_frac)
        bx = smp.x - smp.nx * out
        by = smp.y - smp.ny * out
        side = 1.0 if (idx % 2 == 0) else -1.0
        # backing fronds fan to alternating sides for a woven texture read.
        angle = tang + side * math.radians(38.0)
        scene.leaves.append(LeafSprite(
            x=bx, y=by, angle=angle,
            size=small * rng.uniform(0.7, 1.05),
            type=rng.choice(feathery), t=1.0, seed=rng.next_uint32(),
        ))


def _inner_rank(scene, samples, rng, style, leaf_types, flower_types,
                leaf_len_base, lean, bloom, total_s, band_um):
    """A second rank of leaves sitting INBOARD of the vine, leaning toward the
    aperture, so the inner half of the band is filled too.

    Mirrors the main rank's shingle but on the inboard side, slightly smaller,
    with its own species draw (kept mixed) so the depth-fill still reads as
    diverse foliage rather than a solid gold wall. Density is scaled by
    ``inner_rank``: we step through the samples, placing a leaf at a fraction of
    the stations so the inner rank is a touch airier than the primary.
    """
    # BROAD leaves ONLY, laid as a RHYTHMIC SCALLOP that defines the contour
    # bordering the open aperture. The elegance trick: each blade lies mostly
    # ALONG the vine (trailing with the flow) but is offset inboard so its belly
    # bulges toward the aperture — consecutive blades at a regular cadence make
    # a repeating scallop wave (like a laurel margin), NOT a random serration.
    # Feathery fronds are banished (they read as spiky weeds on the edge), size
    # jitter is small (rhythm over chaos), and the tips all trail one way.
    broad = [t for t in leaf_types if t not in ("fern", "wax_palm")] or list(leaf_types)
    n = len(samples)
    if n < 2:
        return
    # cadence: one scallop crest every ~0.62 leaf-lengths of arc so blades
    # overlap ~40 % — distinct lobes, no gaps, no fine sawtooth.
    step_s = total_s / max(1, n - 1)
    # HEAVY overlap (~65 %) so each blade's POINTED TIP is buried under the next
    # blade's rounded belly — the contour becomes the smooth envelope of the
    # bellies (a flowing gold edge with gentle undulation) instead of a row of
    # exposed pointed tips reading as sawtooth teeth.
    scallop_gap = leaf_len_base * 0.34
    stride = max(1, int(round(scallop_gap / max(step_s, 1e-6))))
    depth_off = band_um * 0.30   # push the belly decisively to the aperture edge
    for idx in range(0, n, stride):
        smp = samples[idx]
        f = smp.s / max(total_s, 1e-6)
        end_bias = 1.0 + style.swell * (math.cos(f * math.pi) ** 2)
        tang = math.atan2(smp.ty, smp.tx)
        # lie almost flat along the flow so the rounded side-margin (belly), not
        # the tip, faces the aperture; tight jitter keeps the wave graceful.
        angle = tang + (lean * 0.20) + rng.uniform(-0.03, 0.03)
        ltype = rng.choice(broad)
        size = leaf_len_base * end_bias * 1.0 * rng.uniform(0.97, 1.04)
        bx = smp.x + smp.nx * depth_off
        by = smp.y + smp.ny * depth_off
        scene.leaves.append(LeafSprite(
            x=bx, y=by, angle=angle, size=size, type=ltype, t=1.0,
            seed=rng.next_uint32(),
        ))


def _inner_tendrils(scene, samples, rng, short, band_um, base_w, strength,
                    rect):
    """Sparse fine curls reaching from the vine gently toward the aperture."""
    stride = max(3, int(6 / max(0.2, strength)))
    curl = band_um * 0.32
    for idx in range(stride // 2, len(samples), stride):
        smp = samples[idx]
        # a short 2-segment curl leaning inward then along the tangent
        ix = smp.x + smp.nx * curl * 0.6
        iy = smp.y + smp.ny * curl * 0.6
        # keep inside the rect
        if rect.dist_frame(ix, iy) < band_um * 0.05:
            continue
        cx = ix + smp.tx * curl * 0.5
        cy = iy + smp.ty * curl * 0.5
        w = base_w * 0.7
        scene.segments.append(Segment(x1=smp.x, y1=smp.y, x2=ix, y2=iy, w=w, t=1.0))
        scene.segments.append(Segment(x1=ix, y1=iy, x2=cx, y2=cy, w=w * 0.8, t=1.0))


def _lay_corners(
    scene, rect, noise, rng, short, band_um, ride_depth, base_w, max_w,
    style, leaf_types, flower_types, laurel_species, bloom, reach,
):
    """A tidy corner medallion that RESOLVES where the two edge ranks meet.

    Each half-edge spline stops ``corner_inset`` short of the sharp corner, so
    the corner would otherwise read as a gap. We fill it deliberately: two short
    stems tuck back ALONG each incoming edge (so the guiding line visually
    continues into the corner and "knots"), then a compact symmetric fan of
    leaves radiates along the diagonal bisector, capped by a single accent
    bloom at the knot. The whole medallion is mirror-symmetric about the
    diagonal — the read is "the garland turns the corner on purpose".
    """
    corners = [
        (rect.left, rect.bottom, 1.0, 1.0),
        (rect.right, rect.bottom, -1.0, 1.0),
        (rect.right, rect.top, -1.0, -1.0),
        (rect.left, rect.top, 1.0, -1.0),
    ]
    inv = 1.0 / math.sqrt(2.0)
    inset = band_um * 0.60
    fan_len = band_um * 0.46 * reach
    broad = [t for t in leaf_types if t not in ("fern", "wax_palm")] or list(leaf_types)
    for cx, cy, sx, sy in corners:
        dxn, dyn = sx * inv, sy * inv
        # medallion origin: inside the corner along the diagonal
        ox = cx + dxn * inset
        oy = cy + dyn * inset
        bisector = math.atan2(dyn, dxn)

        # (Knot stems removed: they floated in the negative space as a stray
        # right-angle "staple" instead of reading as a reconnected vine. The
        # sprig + the two incoming edge garlands resolve the corner on their own.)

        # --- a small terminal SPRIG that turns the corner ---------------------
        # Three modest blades STAGGERED along the diagonal (not all radiating
        # from one point) so they read as a distinct little sprig capping the
        # garland, never a merged gold lump. Tips sweep outward along the
        # diagonal; the pair flanks a central blade.
        layout = [(-0.42, 0.32), (0.42, 0.32), (0.0, 0.60)]  # (angle frac, along)
        for afrac, along in layout:
            ang = bisector + afrac * math.radians(46.0)
            px = ox + dxn * fan_len * along
            py = oy + dyn * fan_len * along
            L = fan_len * (0.9 if along < 0.5 else 1.0) * rng.uniform(0.94, 1.04)
            ltype = laurel_species if style.single_species else rng.choice(broad)
            scene.leaves.append(LeafSprite(
                x=px, y=py, angle=ang,
                size=L, type=ltype, t=1.0, seed=rng.next_uint32(),
            ))

        # --- accent bloom at the knot (over the fan base) ---
        scene.flowers.append(FlowerSprite(
            x=ox + math.cos(bisector) * fan_len * 0.10,
            y=oy + math.sin(bisector) * fan_len * 0.10,
            size=(style.leaf_len_frac * short) * 0.52 * (0.9 + 0.3 * bloom),
            t=1.0, type=rng.choice(flower_types), rot=bisector,
            seed=rng.next_uint32(),
        ))
