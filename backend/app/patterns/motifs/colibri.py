from __future__ import annotations

import math

import numpy as np
from PIL import ImageDraw

from ._pillow import render_silhouette


# ---------------------------------------------------------------------------
# Geometry helpers — everything is authored in a normalized 0..1 art box and
# scaled to the raster grid by ``P``. Curves are built from cubic Béziers
# sampled into dense polylines so the silhouette reads as flowing art-nouveau
# line, never faceted. Interior detail (eye, wing slotting, gorget, tail
# streamer gaps) is punched as negative space AFTER the body is laid down.
# ---------------------------------------------------------------------------


def _bezier(p0, p1, p2, p3, n=48):
    """Sample a cubic Bézier into ``n`` points."""
    out = []
    for i in range(n + 1):
        t = i / n
        mt = 1.0 - t
        x = (mt**3) * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
        y = (mt**3) * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
        out.append((x, y))
    return out


def _closed(*bez_chains):
    """Concatenate several Bézier point chains into one closed polygon."""
    pts: list[tuple[float, float]] = []
    for chain in bez_chains:
        pts.extend(chain)
    return pts


def _rotate_about(pts, pivot, deg):
    """Rotate normalized (x, y) points about ``pivot`` by ``deg`` (screen-CW+).

    y grows DOWN (Pillow), so a positive ``deg`` rotates clockwise on screen —
    i.e. sweeps an up-swept wing tip DOWN toward the belly, which is exactly the
    downstroke motion we want for pose B.
    """
    px, py = pivot
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    out = []
    for x, y in pts:
        dx, dy = x - px, y - py
        out.append((px + dx * c - dy * s, py + dx * s + dy * c))
    return out


def _wing_far_up():
    """Far wing (tall blade of the V), pose A — swept UP-right."""
    return _closed(
        _bezier((0.47, 0.50), (0.58, 0.34), (0.74, 0.22), (0.90, 0.15)),  # leading edge
        _bezier((0.90, 0.15), (0.88, 0.22), (0.84, 0.28), (0.80, 0.33)),  # tip
        _bezier((0.80, 0.33), (0.70, 0.42), (0.60, 0.48), (0.52, 0.51)),  # trailing edge
        _bezier((0.52, 0.51), (0.50, 0.505), (0.48, 0.50), (0.47, 0.50)),  # back to shoulder
    )


def _wing_near_up():
    """Near wing (lower blade of the V), pose A — swept up-and-out."""
    return _closed(
        _bezier((0.50, 0.56), (0.62, 0.49), (0.76, 0.47), (0.90, 0.49)),  # leading edge
        _bezier((0.90, 0.49), (0.85, 0.53), (0.80, 0.56), (0.75, 0.585)),  # tip
        _bezier((0.75, 0.585), (0.66, 0.60), (0.58, 0.60), (0.52, 0.59)),  # trailing edge
        _bezier((0.52, 0.59), (0.51, 0.58), (0.505, 0.57), (0.50, 0.56)),
    )


# Shoulder pivot both wings rotate about for pose B. Sits at the wing root where
# the two blades of the V spring from the back, so a rigid rotation reads as the
# whole wing beating down about the shoulder joint (not a shear).
_SHOULDER = (0.495, 0.53)
_DOWNSTROKE_DEG = 60.0  # sweep the tips down ~60° about the shoulder


def _wing_far_down():
    """Far wing, pose B — mid-downstroke: pose-A blade rotated ~60° down about
    the shoulder and slightly extended so the tip reaches past the belly."""
    up = _wing_far_up()
    down = _rotate_about(up, _SHOULDER, _DOWNSTROKE_DEG)
    # Slight extension: push points radially out from the shoulder by 6% so the
    # downstroke wing looks stretched at full beat, not just tipped over.
    px, py = _SHOULDER
    return [(px + (x - px) * 1.06, py + (y - py) * 1.06) for x, y in down]


def _wing_near_down():
    """Near wing, pose B — mid-downstroke twin of the near wing."""
    up = _wing_near_up()
    down = _rotate_about(up, _SHOULDER, _DOWNSTROKE_DEG)
    px, py = _SHOULDER
    return [(px + (x - px) * 1.06, py + (y - py) * 1.06) for x, y in down]


def _draw_colibri(draw: ImageDraw.ImageDraw, n: int, pose: str = "up") -> None:
    """Art-nouveau hovering hummingbird (colibrí), facing left.

    Authored in a normalized art box where x grows right, y grows DOWN (Pillow
    convention). The bird hovers with its long decurved beak sweeping to the
    upper-left, head high, breast curving down to a slim belly, and a long forked
    streamer tail trailing down-right. Interior negative space: an eye dot, slots
    between the primaries, a covert scallop line, and a throat-gorget crescent.

    ``pose`` selects the wing set. Body, head, beak, neck, tail, feet and all
    negative space are drawn IDENTICALLY in both poses so the two silhouettes are
    pixel-registered — only the wings differ:

      * ``"up"``   — pose A: both wings swept up in a V (the classic hover).
      * ``"down"`` — pose B: both wings rotated ~60° down about the shoulder and
                     slightly extended, i.e. caught mid-downstroke. A few short
                     speed slivers trail behind the wing tips as a motion hint.

    Rocking the box left/right phase-switches A↔B so the wings beat.
    """
    s = float(n)

    def P(x, y):
        return (x * s, y * s)

    def poly(pts, fill=255):
        draw.polygon([P(x, y) for x, y in pts], fill=fill)

    # ----- SHARED BODY PARTS (identical in every pose) ----------------------
    # Drawn as one reusable pass so we can RE-lay it on top of the wing after
    # the wing's feather slots are punched. That re-lay is what guarantees the
    # body silhouette is pixel-identical between pose A and pose B: any slot that
    # happens to overlap the back gets painted back to gold, in the same place,
    # regardless of where the (pose-dependent) wing sat.
    def _draw_body_parts() -> None:
        # BODY — a teardrop: rounded breast at the head end (left), tapering to
        # the tail base (right). Four Bézier arcs around the silhouette.
        body_top = _bezier((0.34, 0.505), (0.44, 0.44), (0.58, 0.45), (0.66, 0.52))
        body_tail = _bezier((0.66, 0.52), (0.70, 0.55), (0.70, 0.58), (0.66, 0.60))
        body_belly = _bezier((0.66, 0.60), (0.56, 0.70), (0.42, 0.70), (0.34, 0.62))
        body_breast = _bezier((0.34, 0.62), (0.30, 0.585), (0.30, 0.545), (0.34, 0.505))
        poly(_closed(body_top, body_tail, body_belly, body_breast))

        # HEAD — rounded head set forward-left and slightly up off the breast.
        head_cx, head_cy, head_r = 0.315, 0.505, 0.072
        draw.ellipse(
            [P(head_cx - head_r, head_cy - head_r), P(head_cx + head_r, head_cy + head_r)],
            fill=255,
        )
        # Neck filler between head and body.
        neck = _closed(
            _bezier((0.33, 0.45), (0.37, 0.46), (0.40, 0.48), (0.42, 0.50)),
            _bezier((0.42, 0.50), (0.40, 0.55), (0.37, 0.57), (0.33, 0.57)),
        )
        poly(neck)

        # BEAK — long, gracefully decurved needle sweeping to the upper-left.
        beak_upper = _bezier((0.255, 0.475), (0.17, 0.44), (0.08, 0.40), (0.03, 0.36))
        beak_lower = _bezier((0.03, 0.375), (0.10, 0.435), (0.19, 0.475), (0.255, 0.505))
        poly(_closed(beak_upper, list(reversed(beak_lower))))

        # TAIL — long forked streamers trailing down-right.
        tail = _closed(
            _bezier((0.64, 0.57), (0.78, 0.62), (0.90, 0.72), (0.97, 0.86)),
            _bezier((0.97, 0.86), (0.93, 0.85), (0.90, 0.84), (0.87, 0.82)),
            _bezier((0.87, 0.82), (0.90, 0.77), (0.88, 0.73), (0.83, 0.71)),
            _bezier((0.83, 0.71), (0.87, 0.80), (0.84, 0.88), (0.78, 0.94)),
            _bezier((0.78, 0.94), (0.75, 0.90), (0.73, 0.85), (0.72, 0.80)),
            _bezier((0.72, 0.80), (0.68, 0.70), (0.65, 0.63), (0.64, 0.59)),
        )
        poly(tail)

        # FEET — tiny, tucked under the belly.
        foot = _closed(
            _bezier((0.42, 0.685), (0.43, 0.72), (0.44, 0.74), (0.45, 0.75)),
            _bezier((0.45, 0.75), (0.455, 0.73), (0.455, 0.71), (0.45, 0.69)),
        )
        poly(foot)

    _draw_body_parts()

    # ----- WINGS (pose-dependent — the ONLY parts that move) ----------------
    # A long scythe rising from the shoulder. Leading edge is a clean sweep;
    # trailing edge scallops back to the shoulder. Primary slots are punched
    # afterward (in the pose's own frame) so the tip reads as separated feathers.
    if pose == "none":
        wing_far = wing_near = None
    elif pose == "down":
        wing_far = _wing_far_down()
        wing_near = _wing_near_down()
    else:
        wing_far = _wing_far_up()
        wing_near = _wing_near_up()
    if wing_far is not None:
        poly(wing_far)
        poly(wing_near)

    # --- Wing interior linework (moves WITH the far wing per pose) ----------
    # Authored in pose-A (up) coordinates on the far wing; for pose B we apply
    # the same rigid rotation + 6% extension as the wing itself so the coverts
    # and primary slots stay locked to the feathers instead of punching the body.
    def _wing_frame(pts):
        if pose != "down":
            return pts
        rot = _rotate_about(pts, _SHOULDER, _DOWNSTROKE_DEG)
        px, py = _SHOULDER
        return [(px + (x - px) * 1.06, py + (y - py) * 1.06) for x, y in rot]

    if pose != "none":
        # Covert scallop — one flowing slit separating the wing coverts from the
        # primaries on the far wing, following its leading sweep.
        covert = _closed(
            _bezier((0.53, 0.49), (0.62, 0.40), (0.72, 0.33), (0.82, 0.28)),
            _bezier((0.83, 0.30), (0.73, 0.36), (0.63, 0.43), (0.55, 0.505)),
        )
        poly(_wing_frame(covert), fill=0)

        # Primary-feather slots on the FAR wing — three tapered gaps between the
        # feather tips so the wing reads as separated primaries, not a solid blade.
        for (bx0, by0, cx0, cy0, cx1, cy1, bx1, by1) in (
            (0.865, 0.185, 0.82, 0.25, 0.77, 0.31, 0.74, 0.355),
            (0.815, 0.215, 0.76, 0.28, 0.71, 0.35, 0.68, 0.40),
            (0.755, 0.255, 0.70, 0.32, 0.65, 0.40, 0.62, 0.45),
        ):
            slot = _closed(
                _bezier((bx0, by0), (cx0, cy0), (cx1, cy1), (bx1, by1), n=24),
                _bezier(
                    (bx1 + 0.010, by1 + 0.004),
                    (cx1 + 0.010, cy1 + 0.004),
                    (cx0 + 0.010, cy0 + 0.004),
                    (bx0 + 0.010, by0 + 0.004),
                    n=24,
                ),
            )
            poly(_wing_frame(slot), fill=0)

    # ----- RE-LAY the shared body on top ------------------------------------
    # Critical for registration: the wing (and its feather slots) may overlap
    # the back/tail differently per pose. Painting the body a second time now
    # restores every shared pixel to gold in EXACTLY the same place regardless
    # of pose, so pose_up and pose_down share a byte-identical body region. The
    # wings still show wherever they extend BEYOND the body.
    _draw_body_parts()

    # =======================================================================
    # BODY NEGATIVE SPACE (pose-independent facial detail) — punched last so it
    # survives the body re-lay. Identical in every pose.
    # =======================================================================

    # Eye: a clean round dot near the front of the head.
    eye_cx, eye_cy, eye_r = 0.295, 0.492, 0.017
    draw.ellipse(
        [P(eye_cx - eye_r, eye_cy - eye_r), P(eye_cx + eye_r, eye_cy + eye_r)],
        fill=0,
    )

    # Throat gorget — a short vertical crescent at the front of the throat,
    # just behind the beak base. Vertical (not a horizontal smile) so it reads
    # as a plumage mark, not a mouth.
    gorget = _closed(
        _bezier((0.285, 0.53), (0.30, 0.545), (0.30, 0.565), (0.285, 0.58)),
        _bezier((0.285, 0.58), (0.315, 0.565), (0.315, 0.545), (0.285, 0.53)),
    )
    poly(gorget, fill=0)

    # --- Motion hint (pose B only): speed slivers behind the wing tips -------
    # Three short gold streaks trailing UP-and-back from where the pose-A tips
    # were, suggesting the wing has just swept down through that air. Positive
    # fill (gold) afterimage streaks, placed above the back so they never touch
    # the shared body silhouette.
    if pose == "down":
        for (sx0, sy0, sx1, sy1, w) in (
            (0.86, 0.17, 0.79, 0.30, 0.018),
            (0.80, 0.21, 0.74, 0.33, 0.016),
            (0.74, 0.26, 0.69, 0.37, 0.014),
        ):
            sliver = _closed(
                _bezier((sx0, sy0), (sx0 - 0.02, sy0 + 0.04),
                        (sx1 + 0.01, sy1 - 0.03), (sx1, sy1), n=16),
                _bezier((sx1 + w * 0.4, sy1 + w), (sx1 + 0.01 + w, sy1 - 0.03 + w),
                        (sx0 - 0.02 + w, sy0 + 0.04 + w), (sx0 + w * 0.4, sy0 + w), n=16),
            )
            poly(sliver, fill=255)


def colibri_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
    pose: str = "up",
) -> np.ndarray:
    """Binary bool grid (n_grid × n_grid) — art-nouveau colibrí facing left.

    ``pose`` = ``"up"`` (wings V, default) or ``"down"`` (mid-downstroke). Body,
    head, beak, tail and feet are identical between poses; only the wings move,
    so the two grids are pixel-registered (body-mask XOR is wing-only). See
    :func:`_draw_colibri`.
    """
    del extent_um  # silhouette is scale-free; caller controls cell_um to hit extent
    return render_silhouette(lambda d, n: _draw_colibri(d, n, pose=pose), n_grid)


def colibri_body_silhouette(
    extent_um: tuple[float, float] | float,
    n_grid: int = 256,
) -> np.ndarray:
    """Binary bool grid — colibrí body ONLY (no wings, no motion slivers).

    Used to verify pose registration: ``pose_up ^ pose_down`` must contain only
    wing pixels, and subtracting this body from that XOR must leave nothing on
    the shared silhouette."""

    def _draw_body(draw: ImageDraw.ImageDraw, n: int) -> None:
        _draw_colibri(draw, n, pose="none")

    del extent_um
    return render_silhouette(_draw_body, n_grid)
