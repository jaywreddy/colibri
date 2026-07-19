"""Space colonization (Runions 2005) anchored to a rectangular perimeter.

Attractors are rejection-sampled inside a band hugging the frame; seed nodes
sit at the four corners and four edge midpoints. Each iteration:

  1. Every live attractor finds its nearest node within the *attraction*
     radius, and that node accumulates the unit vector toward the attractor.
  2. Each influenced node grows a new child node one *segment* step in the
     average pull direction, with a small value-noise wiggle.
  3. Attractors within the *kill* radius of any node are removed.

Stops when no attractor influences any node, or a hard iteration cap is hit.

Per-segment line width is √(subtree-size), so trunks are thick and tips are
thin — same look as the JS prototype. Tips get flowers (probability tuned by
``bloom``) or a leaf; interior nodes occasionally get a large leaf.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..geometry import Mulberry32, RectFrame, ValueNoise2D
from ..scene import FlowerSprite, LeafSprite, Scene, Segment


@dataclass
class _Node:
    x: float
    y: float
    parent: int | None  # index into the node list
    t: float            # growth time (iteration at which it was created)
    children: list[int] = field(default_factory=list)
    subtree: int = 1    # number of descendants including self (recomputed at end)


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
    max_iter: int = 500,
    fill_interior: bool = False,
    edge_gradient: float = 1.0,
    understory: float = 1.0,
    border_vine: float = 1.0,
    corner_fans: float = 1.0,
) -> Scene:
    """Run the colonization and return a renderer-agnostic Scene.

    ``band_um`` defaults to ~12% of the shorter rectangle side so the
    decoration always stays in the same visual fraction of the frame
    regardless of plate size. Override per call to change frame width.

    ``fill_interior`` switches from the perimeter-band border look to a
    WHOLE-FACE fill: attractors scatter over the full rectangle AREA (not just
    an edge band) and seed nodes drop on a jittered interior grid as well as
    the perimeter, so the foliage grows everywhere at once and fills the face
    evenly instead of radiating in from the edges (which starved the center).
    Used by the box-first moiré front carrier, where the foliage silhouette IS
    the artwork across the whole plate.

    Frame-band composition knobs (ignored when ``fill_interior``) — all default
    to 1.0 (the tuned "nice frame" look); dial per-face for variety:

    * ``edge_gradient`` — how hard coverage packs toward the OUTER edge vs the
      inner aperture. 0 = flat depth distribution; 1 = strong outer bias so the
      band reads densest at the plate rim and airier toward the aperture.
    * ``understory`` — density of the secondary small-leaf/tendril pass confined
      to the outer 40 % of the band (the "infill" the frame brief asked for).
      0 disables it; >1 packs it tighter.
    * ``border_vine`` — presence of the continuous running-ornament vine hugging
      the very outer edge (the crisp dense border LINE). 0 disables.
    * ``corner_fans`` — size/reach of the deliberate corner fan compositions.
      0 disables (corners fall back to whatever the growth happens to leave).
    """
    short = min(rect.width_um, rect.height_um)
    if band_um is None:
        band_um = 0.12 * short

    # All scale-dependent radii anchor off ``short`` so dialing extent doesn't
    # silently re-tune the algorithm. Shorter segments + tighter kill radius
    # than the old tuning: this grows a denser, bushier mat that fills the
    # whole band instead of a few long stringy runners.
    # Coarser growth for the whole-face fill: longer segments + wider kill
    # radius mean fewer, chunkier vines cover the same area in far fewer
    # iterations (the O(A×N) attraction scan is the cost). The band look keeps
    # its original fine tuning.
    if fill_interior:
        segment_len = 0.020 * short
        attraction_r = 0.11 * short
        kill_r = 0.030 * short
    else:
        # A richly-filled engraved FRAME. Shorter segments + a tighter kill
        # radius than the old airy tuning let the denser attractor field grow a
        # continuous, well-woven border of vines (not a mottled fused mat, and
        # not a few stringy runners either). Motifs stay individually readable
        # because bloom/foliage keep them the minority and each is sized under
        # the band width — density lives in the vine skeleton, not fused blobs.
        segment_len = 0.014 * short
        attraction_r = 0.085 * short
        kill_r = 0.032 * short
    wiggle = 0.30  # fraction of a segment perturbed by noise

    rng = Mulberry32(seed)
    noise = ValueNoise2D(seed ^ 0x9E3779B1)

    perim = rect.perimeter
    attractors: list[tuple[float, float]] = []
    nodes: list[_Node] = []
    node_xy: list[tuple[float, float]] = []  # parallel coords for numpy

    if fill_interior:
        # --- WHOLE-FACE fill --------------------------------------------------
        # Attractors scatter over the full rectangle AREA on a jittered grid so
        # every region (center included) has a pull target. A jittered grid
        # (vs pure random) guarantees no empty patches the way perimeter
        # rejection sampling left the middle bare. Count scales with area but
        # is hard-capped for performance (each attractor is an O(N) scan/iter).
        area = rect.width_um * rect.height_um
        target = int(area / (segment_len * segment_len * 6.0) * max(0.05, density))
        n_attractors = max(180, min(700, target))
        # Grid dims proportional to the aspect ratio.
        aspect = rect.width_um / max(1e-6, rect.height_um)
        gy = max(4, int(round(math.sqrt(n_attractors / max(1e-6, aspect)))))
        gx = max(4, int(round(n_attractors / gy)))
        margin = segment_len  # keep just inside the rect
        for j in range(gy):
            for i in range(gx):
                fx = (i + 0.15 + 0.7 * rng.next_float()) / gx
                fy = (j + 0.15 + 0.7 * rng.next_float()) / gy
                ax = rect.left + margin + fx * (rect.width_um - 2 * margin)
                ay = rect.bottom + margin + fy * (rect.height_um - 2 * margin)
                attractors.append((ax, ay))
        if not attractors:
            return Scene(max_t=0.0)

        # Seed nodes: dense perimeter ring PLUS a coarse jittered interior grid,
        # so growth ignites across the whole face simultaneously (no long
        # runners from the edge, which is what starved the middle before).
        seed_spacing = max(segment_len * 3.0, short * 0.06)
        n_seeds = max(8, int(perim / seed_spacing))
        for i in range(n_seeds):
            s = (i / n_seeds) * perim
            (px, py), _t, (nx, ny) = rect.perim_info(s)
            x = px + nx * segment_len * 0.5
            y = py + ny * segment_len * 0.5
            nodes.append(_Node(x=x, y=y, parent=None, t=0.0))
            node_xy.append((x, y))
        # Interior seed grid — roughly 1 seed per ~5 attractor cells.
        sx = max(2, gx // 3)
        sy = max(2, gy // 3)
        for j in range(sy):
            for i in range(sx):
                fx = (i + 0.5 + 0.5 * (rng.next_float() - 0.5)) / sx
                fy = (j + 0.5 + 0.5 * (rng.next_float() - 0.5)) / sy
                x = rect.left + margin + fx * (rect.width_um - 2 * margin)
                y = rect.bottom + margin + fy * (rect.height_um - 2 * margin)
                nodes.append(_Node(x=x, y=y, parent=None, t=0.0))
                node_xy.append((x, y))
    else:
        # --- perimeter BAND (border decoration) -------------------------------
        # STRATIFIED by edge so density is balanced on all four sides. The old
        # uniform rejection sample left whole corners nearly empty (dense
        # top-left, bare bottom-right) because rejection favors wherever the
        # RNG happened to cluster. Here we walk each edge's arc-length in even
        # strata and jitter a point into the band, guaranteeing coverage around
        # the entire perimeter.
        #
        # This is a DELIBERATE FRAME, not a scatter of sprigs: many more
        # attractors than the old airy tuning, with depth pulled toward the
        # OUTER edge by ``edge_gradient`` so the band reads DENSE at the plate
        # rim and graduates to open negative space near the aperture. The
        # earlier "mush" regression came from fat motifs fusing, not from too
        # many vines — so we can safely pack the growth skeleton tight and keep
        # the motifs individually sized/spaced (bloom/foliage below).
        n_attractors = min(340, int(240 * max(0.05, density)))
        # Depth-shaping exponent: a bigger exponent pushes the sampled depth
        # toward 0 (the outer edge). edge_gradient in [0,1] maps to exp [1, 3.2].
        depth_exp = 1.0 + 2.2 * max(0.0, min(1.0, edge_gradient))
        for i in range(n_attractors):
            s = ((i + rng.next_float()) / n_attractors) * perim
            (px, py), _t, (nx, ny) = rect.perim_info(s)
            # rng**depth_exp biases toward 0; band*(0.06 .. 0.96) keeps a hair
            # of outer margin and never quite touches the aperture edge.
            depth = band_um * (0.06 + 0.90 * rng.next_float() ** depth_exp)
            attractors.append((px + nx * depth, py + ny * depth))
        if not attractors:
            return Scene(max_t=0.0)

        # Seed nodes along the perimeter. DENSE outer-edge ring so the vine
        # closes into a continuous running ornament hugging the border, rather
        # than a handful of stringy runners that leave gaps. Corners get an
        # extra dedicated seed each for intentional framing (see corner pass).
        seed_spacing = max(segment_len * 2.4, band_um * 0.55)
        n_seeds = max(8, int(perim / seed_spacing))
        for i in range(n_seeds):
            s = (i / n_seeds) * perim
            (px, py), _t, (nx, ny) = rect.perim_info(s)
            # Ride right at the outer edge so growth launches from the rim.
            x = px + nx * segment_len * 0.35
            y = py + ny * segment_len * 0.35
            nodes.append(_Node(x=x, y=y, parent=None, t=0.0))
            node_xy.append((x, y))

    # --- iterate -------------------------------------------------------------
    # The attraction scan is numpy-vectorized: an (A × N) distance matrix with
    # np.argmin replaces the O(attractors × nodes) Python loop (the dominant
    # cost of a cold plate compose). np.argmin's first-occurrence tie-break
    # matches the loop's strict-< first-index semantics, and elementwise
    # dx*dx+dy*dy is IEEE-identical to the scalar code, so output scenes are
    # byte-identical to the original implementation. The per-attractor pull
    # accumulation stays a (short) Python loop in attractor order to preserve
    # float accumulation order and dict insertion order (hence RNG call order).
    A = np.asarray(attractors, dtype=np.float64)  # (A, 2), live attractors
    ar2 = attraction_r * attraction_r
    kr2 = kill_r * kill_r
    iteration = 0
    # Whole-face fill converges fast (long segments, wide kill radius); cap
    # iterations tighter so a stray un-killed attractor can't spin the loop.
    iter_cap = 140 if fill_interior else max_iter
    while iteration < iter_cap and len(A):
        N = np.asarray(node_xy, dtype=np.float64)  # (N, 2)
        dx_m = A[:, 0, None] - N[None, :, 0]       # (A, N)
        dy_m = A[:, 1, None] - N[None, :, 1]
        d2_m = dx_m * dx_m + dy_m * dy_m
        best_idx = np.argmin(d2_m, axis=1)
        best_d2 = d2_m[np.arange(len(A)), best_idx]
        influenced = best_d2 < ar2

        node_pull: dict[int, list[float]] = {}  # node_idx -> [vx, vy, count]
        for ai in np.nonzero(influenced)[0]:
            bi = int(best_idx[ai])
            dx = A[ai, 0] - nodes[bi].x
            dy = A[ai, 1] - nodes[bi].y
            inv = 1.0 / max(math.hypot(dx, dy), 1e-9)
            bucket = node_pull.setdefault(bi, [0.0, 0.0, 0.0])
            bucket[0] += dx * inv
            bucket[1] += dy * inv
            bucket[2] += 1.0

        if not node_pull:
            break

        # Spawn a new node per influenced parent.
        new_node_positions: list[tuple[int, float, float]] = []
        for parent_idx, (vx, vy, count) in node_pull.items():
            nx = vx / count
            ny = vy / count
            mag = math.hypot(nx, ny)
            if mag < 1e-9:
                continue
            nx /= mag
            ny /= mag
            # Wiggle via value-noise sampled at the parent's position.
            n = noise.fbm(nodes[parent_idx].x * 0.005, nodes[parent_idx].y * 0.005, octaves=3)
            theta = (n - 0.5) * 2.0 * wiggle
            cs, sn = math.cos(theta), math.sin(theta)
            wx = cs * nx - sn * ny
            wy = sn * nx + cs * ny
            new_x = nodes[parent_idx].x + wx * segment_len
            new_y = nodes[parent_idx].y + wy * segment_len
            # Constrain new node to stay inside the rect (clamp inward if it
            # tried to escape).
            if rect.dist_frame(new_x, new_y) < 0:
                # Project back to the nearest interior point at ~1 px of margin.
                new_x = max(rect.left + 1.0, min(rect.right - 1.0, new_x))
                new_y = max(rect.bottom + 1.0, min(rect.top - 1.0, new_y))
            new_node_positions.append((parent_idx, new_x, new_y))

        if not new_node_positions:
            break

        iteration += 1
        for parent_idx, new_x, new_y in new_node_positions:
            child_idx = len(nodes)
            nodes.append(_Node(x=new_x, y=new_y, parent=parent_idx, t=float(iteration)))
            node_xy.append((new_x, new_y))
            nodes[parent_idx].children.append(child_idx)

        # Kill attractors within kill_r of any FRESH node from this iteration.
        # Old nodes already killed their neighbors in prior iterations, so
        # re-checking the whole node list every iteration is wasted work
        # (and made the algorithm O(iter * N_attractors * N_nodes) — the
        # OOM-er at default density on a 6-face box). Vectorized: boolean
        # any() over the (A × fresh) distance matrix is order-independent.
        F = np.asarray([(fx, fy) for _, fx, fy in new_node_positions], dtype=np.float64)
        fdx = A[:, 0, None] - F[None, :, 0]
        fdy = A[:, 1, None] - F[None, :, 1]
        killed = ((fdx * fdx + fdy * fdy) < kr2).any(axis=1)
        A = A[~killed]

    # --- compute subtree sizes for line-width modulation ---------------------
    # Post-order traversal: children first, then parent.
    order = sorted(range(len(nodes)), key=lambda i: nodes[i].t, reverse=True)
    for i in order:
        n = nodes[i]
        n.subtree = 1 + sum(nodes[c].subtree for c in n.children)

    # --- build Scene ---------------------------------------------------------
    scene = Scene()
    # Vine gauge: a firm minimum so no stroke drops to a 1-2px scribble at
    # plate raster (~33μm pitch → base ≈ 5px, trunk ≈ 12px). Taper is by
    # √(subtree) as before, giving thick trunks that graduate to firm tips.
    # Delicate vine for a border-engraving look: a firm-but-fine line that
    # reads as a drawn stem, not a chunky branch. Still >= 3-4 raster px at
    # plate pitch so it never drops to a hairline scribble.
    base_w = 0.0048 * short   # μm, thinnest tip stroke
    max_w = 0.0100 * short    # μm, trunk
    max_subtree = max((n.subtree for n in nodes), default=1)
    for n in nodes:
        if n.parent is None:
            continue
        p = nodes[n.parent]
        w_frac = math.sqrt(n.subtree / max_subtree)
        seg_w = base_w + (max_w - base_w) * w_frac
        scene.segments.append(Segment(
            x1=p.x, y1=p.y, x2=n.x, y2=n.y,
            w=seg_w,
            t=n.t,
        ))

    # --- scatter flowers + leaves --------------------------------------------
    # LEAF-FORWARD composition. The old tuning bloomed ~75% of tips into
    # flowers, and the flowers were large enough to fuse into solid discs at
    # plate scale (the "splotches"). Botanical framing reads as lush foliage
    # punctuated by occasional blooms — so flowers are the minority, and every
    # motif is sized to stay a crisp silhouette (well under the band width).
    flower_prob = max(0.0, min(0.55, 0.22 + 0.24 * bloom))
    # Interior leaves are the main clutter culprit — near-zero so the band reads
    # as a spare vine studded with the occasional broad leaf, not a continuous
    # hedge. Almost every interior node stays bare vine (negative space); motifs
    # live on the growth TIPS, which the wide kill radius already spaces out.
    interior_leaf_prob = max(0.0, min(0.25, 0.02 + 0.10 * foliage))
    # Motifs are BIGGER now (they're the readable subjects, spaced apart) but
    # still capped against the band so a single motif never overruns the frame.
    # Each tropical element should span most of the band width so it reads as
    # its species at frame scale.
    flower_size = min(band_um * 0.72, 0.045 * short) * (0.85 + 0.30 * bloom)
    leaf_size_base = min(band_um * 0.92, 0.058 * short) * (0.85 + 0.25 * foliage)

    flower_types = list(flower_types) or ["orchid"]
    leaf_types = list(leaf_types) or ["fern"]

    def _edge_size_factor(x: float, y: float) -> float:
        """Motif size multiplier by band depth (band mode only).

        Full-size at the outer rim, tapering to ~0.6 at the aperture edge so
        the canopy reads heaviest at the border and the inner edge stays tidy
        (no big blades crowding the aperture). No-op for the whole-face fill.
        """
        if fill_interior or band_um <= 0:
            return 1.0
        d = max(0.0, rect.dist_frame(x, y))
        frac = min(1.0, d / band_um)  # 0 at rim, 1 at inner band edge
        return 1.0 - 0.40 * frac ** 1.3

    for n in nodes:
        is_tip = not n.children and n.parent is not None
        if is_tip:
            roll = rng.next_float()
            esf = _edge_size_factor(n.x, n.y)
            if roll < flower_prob:
                ftype = rng.choice(flower_types)
                scene.flowers.append(FlowerSprite(
                    x=n.x, y=n.y,
                    size=flower_size * esf * rng.uniform(0.85, 1.15),
                    t=n.t,
                    type=ftype,
                    rot=rng.uniform(-math.pi, math.pi),
                    seed=rng.next_uint32(),
                ))
            else:
                ltype = rng.choice(leaf_types)
                # Orient leaf along the parent → tip direction.
                p = nodes[n.parent]
                angle = math.atan2(n.y - p.y, n.x - p.x)
                scene.leaves.append(LeafSprite(
                    x=n.x, y=n.y,
                    angle=angle,
                    size=leaf_size_base * esf * rng.uniform(0.8, 1.1),
                    type=ltype,
                    t=n.t,
                    seed=rng.next_uint32(),
                ))
        else:
            # Interior node — occasional leaf for body.
            if n.parent is not None and rng.next_float() < interior_leaf_prob:
                p = nodes[n.parent]
                angle = math.atan2(n.y - p.y, n.x - p.x) + (rng.uniform(-1.0, 1.0)) * 0.6
                ltype = rng.choice(leaf_types)
                scene.leaves.append(LeafSprite(
                    x=n.x, y=n.y,
                    angle=angle,
                    size=leaf_size_base * 1.2 * _edge_size_factor(n.x, n.y)
                    * rng.uniform(0.85, 1.15),
                    type=ltype,
                    t=n.t,
                    seed=rng.next_uint32(),
                ))

    # --- FRAME composition passes (band mode only) ---------------------------
    # These turn the raw growth into a deliberate engraved FRAME: a crisp dense
    # border line hugging the rim, a denser small-leaf understory in the outer
    # band, and intentional corner fans. Skipped for the whole-face fill.
    if not fill_interior:
        _add_border_vine(
            scene, rect, rng, noise, short, band_um,
            base_w, max_w, iteration, strength=border_vine,
        )
        _add_understory(
            scene, rect, rng, short, band_um, leaf_types,
            iteration, density=understory, foliage=foliage,
        )
        _add_corner_fans(
            scene, rect, rng, short, band_um, leaf_types, flower_types,
            leaf_size_base, flower_size, bloom, iteration, reach=corner_fans,
        )

    scene.max_t = float(iteration)
    return scene


# --- frame composition helpers ----------------------------------------------
# Understory leaves are biased toward the FEATHERY species (fern / wax_palm)
# when the theme offers them: they read as fine "filler" texture distinct from
# the broad canopy blades, and (bonus) they carry the high angle buckets, so
# in the moiré preview the understory shimmers in its own direction — an easy
# way to make the two frame layers visually separate without a dedicated
# bucket. INTEGRATOR NOTE: if you want the understory on a guaranteed-distinct
# fringe (independent of theme leaf mix), add an "understory" key to
# plates.MOTIF_ANGLE_BUCKET (e.g. bucket 6, N_FRAME_BUCKETS -> 7, top level 180
# still < ART_MIN·255=191) and tag these sprites with type="understory".
_FEATHERY = ("fern", "wax_palm")


def _outer_edge_point(
    rect: RectFrame, s: float, depth: float
) -> tuple[float, float, float, float]:
    """Point ``depth`` μm inward from the perimeter at arc-length ``s``.

    Returns ``(x, y, tx, ty)`` — position plus the perimeter tangent, so a
    caller can lay a running ornament ALONG the edge.
    """
    (px, py), (tx, ty), (nx, ny) = rect.perim_info(s)
    return px + nx * depth, py + ny * depth, tx, ty


def _add_border_vine(
    scene: Scene,
    rect: RectFrame,
    rng: Mulberry32,
    noise: ValueNoise2D,
    short: float,
    band_um: float,
    base_w: float,
    max_w: float,
    t: float,
    *,
    strength: float,
) -> None:
    """A continuous running vine hugging the very outer edge.

    This is the crisp dense border LINE that makes the band read as a framed
    ornament rather than scattered growth. It rides at a shallow, gently
    undulating depth just inside the rim the whole way around the perimeter,
    stroked as a firm mid-gauge vine.
    """
    if strength <= 0.0:
        return
    perim = rect.perimeter
    # One sample every ~1.2% of the shorter side → smooth curve, cheap.
    step = max(0.012 * short, band_um * 0.05)
    n = max(16, int(perim / step))
    base_depth = band_um * 0.16          # nominal ride depth (near the rim)
    wobble = band_um * 0.10 * strength   # undulation amplitude
    w = base_w + (max_w - base_w) * 0.45  # firm, mid-gauge border stroke
    prev: tuple[float, float] | None = None
    for i in range(n + 1):
        s = (i / n) * perim
        # Undulate the depth with value noise so the border reads hand-drawn,
        # not a machined offset. Sample noise along arc-length.
        u = noise.fbm(s * 0.0015, 13.7, octaves=2)
        depth = base_depth + (u - 0.5) * 2.0 * wobble
        depth = max(band_um * 0.05, depth)
        x, y, _tx, _ty = _outer_edge_point(rect, s, depth)
        if prev is not None:
            scene.segments.append(Segment(
                x1=prev[0], y1=prev[1], x2=x, y2=y, w=w, t=t,
            ))
        prev = (x, y)


def _add_understory(
    scene: Scene,
    rect: RectFrame,
    rng: Mulberry32,
    short: float,
    band_um: float,
    leaf_types: Sequence[str],
    t: float,
    *,
    density: float,
    foliage: float,
) -> None:
    """Secondary small-leaf / tendril pass in the OUTER 40 % of the band.

    This is the edge-density gradient's payload: it packs extra fine foliage
    against the rim (densest at the very edge, thinning toward the 40 % line),
    so the frame's border is visibly richer than its interior. Leaves are
    SMALL (they're texture, not subjects) and biased feathery, and each drops a
    short tendril stroke so the understory reads as woven growth, not confetti.
    """
    if density <= 0.0:
        return
    leaf_types = list(leaf_types) or ["fern"]
    feathery = [t_ for t_ in leaf_types if t_ in _FEATHERY] or leaf_types
    perim = rect.perimeter
    # Count scales with perimeter and the density knob; capped for perf.
    n = min(520, int(perim / (0.010 * short) * density * (0.6 + 0.5 * foliage)))
    small = min(band_um * 0.34, 0.026 * short)
    for i in range(n):
        s = ((i + rng.next_float()) / n) * perim
        # Depth confined to the outer 40 % of the band, packed toward the rim
        # (squared → denser at the edge).
        depth = band_um * (0.03 + 0.37 * rng.next_float() ** 2.0)
        (px, py), (tx, ty), (nx, ny) = rect.perim_info(s)
        x = px + nx * depth
        y = py + ny * depth
        # Orient the tendril mostly ALONG the edge (tangent) with only a slight
        # inward lean, so the understory runs as a ribbon parallel to the rim
        # and keeps a tidy inner boundary instead of splaying into the aperture.
        tl = small * rng.uniform(0.6, 1.1)
        tang = math.atan2(ty, tx)
        along = 1.0 if rng.next_float() < 0.5 else -1.0
        ang = tang + along * 0.0 + rng.uniform(-0.5, 0.5)
        # Bias the leaf tip to travel tangentially (cos/sin of tangent) and only
        # a little inward, capping how far inward any tip can reach.
        inward = 0.30 * rng.next_float()
        dirx = math.cos(ang) * along * (1.0 - inward) + nx * inward
        diry = math.sin(ang) * along * (1.0 - inward) + ny * inward
        dmag = math.hypot(dirx, diry) or 1.0
        dirx /= dmag
        diry /= dmag
        ex = x + dirx * tl
        ey = y + diry * tl
        # Keep the whole understory inside the outer half of the band.
        if rect.dist_frame(ex, ey) > band_um * 0.5:
            ex, ey = x, y
        if rect.dist_frame(ex, ey) >= 0:
            scene.segments.append(Segment(
                x1=x, y1=y, x2=ex, y2=ey, w=small * 0.14, t=t,
            ))
        ltype = rng.choice(feathery)
        scene.leaves.append(LeafSprite(
            x=ex, y=ey,
            angle=math.atan2(diry, dirx),
            size=small * rng.uniform(0.7, 1.05),
            type=ltype,
            t=t,
            seed=rng.next_uint32(),
        ))


def _add_corner_fans(
    scene: Scene,
    rect: RectFrame,
    rng: Mulberry32,
    short: float,
    band_um: float,
    leaf_types: Sequence[str],
    flower_types: Sequence[str],
    leaf_size_base: float,
    flower_size: float,
    bloom: float,
    t: float,
    *,
    reach: float,
) -> None:
    """Intentional corner compositions — a radiating fan of fronds + a bloom.

    Each of the four corners gets a small fan of leaves splaying along the
    corner bisector (into the band) plus a single accent flower, so the frame
    turns its corners on purpose instead of leaving them to chance.
    """
    if reach <= 0.0:
        return
    leaf_types = list(leaf_types) or ["fern"]
    flower_types = list(flower_types) or ["orchid"]
    corners = [
        (rect.left, rect.bottom, 1.0, 1.0),    # bottom-left, bisector +x+y
        (rect.right, rect.bottom, -1.0, 1.0),  # bottom-right
        (rect.right, rect.top, -1.0, -1.0),    # top-right
        (rect.left, rect.top, 1.0, -1.0),      # top-left
    ]
    inset = band_um * 0.18
    fan_len = band_um * 0.78 * reach
    n_fronds = 5
    for cx, cy, sx, sy in corners:
        # Fan origin: a touch inside the corner along the diagonal.
        diag = math.hypot(sx, sy)
        ox = cx + (sx / diag) * inset
        oy = cy + (sy / diag) * inset
        bisector = math.atan2(sy, sx)
        spread = math.radians(70.0)
        for k in range(n_fronds):
            frac = k / (n_fronds - 1) - 0.5      # -0.5 .. 0.5
            ang = bisector + frac * spread
            L = fan_len * rng.uniform(0.75, 1.05)
            tipx = ox + math.cos(ang) * L
            tipy = oy + math.sin(ang) * L
            # Clamp the frond tip inside the rect.
            if rect.dist_frame(tipx, tipy) < 0:
                tipx = max(rect.left + 1.0, min(rect.right - 1.0, tipx))
                tipy = max(rect.bottom + 1.0, min(rect.top - 1.0, tipy))
            scene.segments.append(Segment(
                x1=ox, y1=oy, x2=tipx, y2=tipy,
                w=(0.006 * short) * (1.0 - 0.4 * abs(frac) * 2.0), t=t,
            ))
            ltype = rng.choice(leaf_types)
            scene.leaves.append(LeafSprite(
                x=tipx, y=tipy,
                angle=ang,
                size=leaf_size_base * rng.uniform(0.85, 1.1),
                type=ltype,
                t=t,
                seed=rng.next_uint32(),
            ))
        # A single accent bloom nestled at the fan base.
        scene.flowers.append(FlowerSprite(
            x=ox + math.cos(bisector) * fan_len * 0.28,
            y=oy + math.sin(bisector) * fan_len * 0.28,
            size=flower_size * (0.95 + 0.25 * bloom),
            t=t,
            type=rng.choice(flower_types),
            rot=bisector,
            seed=rng.next_uint32(),
        ))
