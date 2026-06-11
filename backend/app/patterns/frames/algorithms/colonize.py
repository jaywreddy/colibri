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
) -> Scene:
    """Run the colonization and return a renderer-agnostic Scene.

    ``band_um`` defaults to ~12% of the shorter rectangle side so the
    decoration always stays in the same visual fraction of the frame
    regardless of plate size. Override per call to change frame width.
    """
    short = min(rect.width_um, rect.height_um)
    if band_um is None:
        band_um = 0.12 * short

    # All scale-dependent radii anchor off ``short`` so dialing extent doesn't
    # silently re-tune the algorithm.
    segment_len = 0.012 * short
    attraction_r = 0.06 * short
    kill_r = 0.018 * short
    wiggle = 0.25  # fraction of a segment perturbed by noise

    rng = Mulberry32(seed)
    noise = ValueNoise2D(seed ^ 0x9E3779B1)

    # --- scatter attractors in the perimeter band ----------------------------
    # Attractor count matches the spec's reference of ~560 attractors at
    # density 1.0. With the raster-pen path the cost is negligible (μs per
    # motif) so we can stay close to the spec for natural density.
    n_attractors = min(800, int(560 * max(0.05, density)))
    attractors: list[tuple[float, float]] = []
    # Hard cap on attempts so degenerate inputs don't spin forever.
    max_attempts = max(n_attractors * 10, 200)
    attempts = 0
    while len(attractors) < n_attractors and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(rect.left, rect.right)
        y = rng.uniform(rect.bottom, rect.top)
        if rect.in_band(x, y, band_um):
            attractors.append((x, y))
    # If we starved (e.g. extreme aspect ratio), trim n_attractors silently.
    if not attractors:
        return Scene(max_t=0.0)

    # --- seed nodes at corners + edge midpoints ------------------------------
    perim_anchors = [
        rect.perim_info(0.0),                                          # bottom-left corner
        rect.perim_info(rect.width_um),                                # bottom-right corner
        rect.perim_info(rect.width_um + rect.height_um),               # top-right
        rect.perim_info(2 * rect.width_um + rect.height_um),           # top-left
        rect.perim_info(0.5 * rect.width_um),                          # bottom mid
        rect.perim_info(rect.width_um + 0.5 * rect.height_um),         # right mid
        rect.perim_info(1.5 * rect.width_um + rect.height_um),         # top mid
        rect.perim_info(2 * rect.width_um + 1.5 * rect.height_um),     # left mid
    ]
    nodes: list[_Node] = []
    node_xy: list[tuple[float, float]] = []  # parallel coords for numpy
    for (px, py), _, (nx, ny) in perim_anchors:
        # Nudge inward by one segment so the first growth step doesn't try
        # to leave the rectangle through the frame.
        x = px + nx * segment_len * 0.5
        y = py + ny * segment_len * 0.5
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
    while iteration < max_iter and len(A):
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
    base_w = 0.0035 * short   # μm, thinnest tip stroke
    max_w = 0.011 * short     # μm, trunk
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
    flower_prob = max(0.0, min(1.0, 0.62 + 0.22 * bloom))
    interior_leaf_prob = max(0.0, min(1.0, 0.18 * foliage))
    flower_size = 0.04 * short * (0.85 + 0.35 * bloom)
    leaf_size_base = 0.045 * short * (0.85 + 0.30 * foliage)

    flower_types = list(flower_types) or ["orchid"]
    leaf_types = list(leaf_types) or ["fern"]

    for n in nodes:
        is_tip = not n.children and n.parent is not None
        if is_tip:
            roll = rng.next_float()
            if roll < flower_prob:
                ftype = rng.choice(flower_types)
                scene.flowers.append(FlowerSprite(
                    x=n.x, y=n.y,
                    size=flower_size * rng.uniform(0.85, 1.15),
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
                    size=leaf_size_base * rng.uniform(0.8, 1.1),
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
                    size=leaf_size_base * 1.2 * rng.uniform(0.85, 1.15),
                    type=ltype,
                    t=n.t,
                    seed=rng.next_uint32(),
                ))

    scene.max_t = float(iteration)
    return scene
