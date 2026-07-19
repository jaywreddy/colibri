"""Side-by-side demo renderers for the fine-pitch grating / moiré toolkit.

Each ``render_*`` returns a PIL Image comparing carrier periods 20/12/8/6/4 µm
for one effect. Two views per period are shown because the effect lives at two
very different length scales:

  * FAB view — a magnified TILE of the actual gold pattern the litho writes
    (true µm periods, blown up so the 4–20 µm lines resolve on screen). This is
    "what the mask looks like".
  * VIEW view — what the eye/shader sees at viewing scale: the moiré beat, the
    phase-switch state, the magnified ghost, the animation frame. This is "what
    the effect looks like".

The moiré beat is computed the honest way: SUPERPOSE the two gold gratings
(front over back, back parallax-shifted by tilt) at true µm periods on a fine
grid, then BOX-DOWNSAMPLE (area-average) to the viewing resolution. The
downsample IS the eye's low-pass — the beat survives, the carrier washes to
grey — so the VIEW panels are physically faithful, not hand-drawn.

Rendered by the throwaway ``_run_demos`` at the bottom (``python -m`` style) to
the scratchpad as ``fx_*.png``.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import gratings as gr
from . import moire as mo

CARRIER_PERIODS_UM = (20.0, 12.0, 8.0, 6.0, 4.0)

# Gold-on-dark palette to match the plate look (gold lines on near-black glass).
GOLD = (212, 175, 55)
DARK = (18, 16, 12)
INK = (235, 230, 215)


def _font(size: int = 13):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _colorize(gray: np.ndarray) -> np.ndarray:
    """Map a 0..1 gray coverage array to a gold-on-dark RGB uint8 image."""
    g = np.clip(gray, 0.0, 1.0)[..., None]
    rgb = (1.0 - g) * np.array(DARK, float) + g * np.array(GOLD, float)
    return rgb.astype(np.uint8)


def _superpose_pair(
    front: np.ndarray, back: np.ndarray
) -> np.ndarray:
    """Two-layer stack coverage: gold where EITHER layer is gold (opaque gold on
    either face blocks/looks gold). Returns float 0/1 grid.

    This is the physically correct combination for absorptive gold: the stack
    transmits only where BOTH faces are open, so apparent gold coverage is the
    OR of the two masks. The moiré is the spatial beat of that OR.
    """
    return (front | back).astype(np.float32)


def _area_downsample(grid: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Box (area-average) downsample a float grid to (H, W) — the eye's low-pass.

    The carrier (period ≪ output pixel) averages to its duty grey; the beat
    (period ≫ output pixel) survives. Uses integer-factor reshape when possible,
    else a PIL BILINEAR resize as a fallback.
    """
    oh, ow = out_hw
    h, w = grid.shape
    if h % oh == 0 and w % ow == 0:
        fy, fx = h // oh, w // ow
        return grid.reshape(oh, fy, ow, fx).mean(axis=(1, 3))
    im = Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8), "L")
    im = im.resize((ow, oh), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _fab_tile(period_um: float, angle_deg: float, tile_px: int = 130,
              px_per_um: float = 4.0, phase: float = 0.0,
              period_b_um: float | None = None, angle_b_deg: float | None = None,
              back_shift_um: float = 0.0) -> np.ndarray:
    """Magnified tile of the true gold pattern (one or two superposed gratings).

    ``px_per_um`` blows the µm pattern up so 4–20 µm lines resolve. If
    ``period_b_um`` is given, a second (back) grating is superposed, optionally
    shifted by ``back_shift_um`` (parallax) — the fab-scale view of the beat.
    """
    span_um = tile_px / px_per_um
    n = tile_px
    a = math.radians(angle_deg)
    xs = (np.arange(n) - (n - 1) / 2.0) / px_per_um
    ys = ((n - 1) / 2.0 - np.arange(n)) / px_per_um
    X, Y = np.meshgrid(xs, ys)
    fa = ((X * math.cos(a) + Y * math.sin(a)) / period_um + phase) % 1.0 < 0.5
    if period_b_um is not None:
        ab = math.radians(angle_b_deg if angle_b_deg is not None else angle_deg)
        coord = ((X + back_shift_um) * math.cos(ab) + Y * math.sin(ab)) / period_b_um
        fb = coord % 1.0 < 0.5
        grid = (fa | fb).astype(np.float32)
    else:
        grid = fa.astype(np.float32)
    return grid


def _beat_view(period_front_um: float, period_back_um: float,
               angle_off_deg: float, view_px: int, view_um: float,
               back_shift_um: float = 0.0, super_sample: int = 6) -> np.ndarray:
    """Physically-faithful moiré VIEW: superpose the two true-µm gratings on a
    supersampled grid over ``view_um`` of plate, then area-downsample to
    ``view_px``. ``back_shift_um`` slides the back grating (parallax under tilt).
    """
    hi = view_px * super_sample
    xs = (np.arange(hi) - (hi - 1) / 2.0) * (view_um / hi)
    ys = ((hi - 1) / 2.0 - np.arange(hi)) * (view_um / hi)
    X, Y = np.meshgrid(xs, ys)
    front = (X / period_front_um) % 1.0 < 0.5
    ab = math.radians(angle_off_deg)
    coord = ((X + back_shift_um) * math.cos(ab) + Y * math.sin(ab)) / period_back_um
    back = coord % 1.0 < 0.5
    stack = _superpose_pair(front, back)
    return _area_downsample(stack, (view_px, view_px))


def _panel(period_um: float, fab: np.ndarray, view: np.ndarray,
           caption: str, cell_w: int = 300) -> Image.Image:
    """Compose one period's column: fab tile (top) + view (bottom) + captions."""
    pad = 8
    tile_side = 130
    view_side = cell_w - 2 * pad
    fab_im = Image.fromarray(_colorize(fab)).resize((tile_side, tile_side), Image.NEAREST)
    view_im = Image.fromarray(_colorize(view)).resize((view_side, view_side), Image.NEAREST)
    header_h = 26
    label_h = 18
    total_h = header_h + tile_side + label_h + view_side + label_h + pad
    cell = Image.new("RGB", (cell_w, total_h), DARK)
    d = ImageDraw.Draw(cell)
    f = _font(15)
    fs = _font(11)
    d.text((pad, 5), f"period {period_um:g} µm", fill=INK, font=f)
    y = header_h
    cell.paste(fab_im, ((cell_w - tile_side) // 2, y))
    y += tile_side
    d.text((pad, y + 2), "FAB mask (×tile)", fill=(150, 145, 130), font=fs)
    y += label_h
    cell.paste(view_im, (pad, y))
    y += view_side
    d.text((pad, y + 2), caption, fill=(150, 145, 130), font=fs)
    return cell


def _grid_image(title: str, subtitle: str, panels: list[Image.Image]) -> Image.Image:
    top_h = 46
    gap = 6
    w = sum(p.width for p in panels) + gap * (len(panels) + 1)
    h = top_h + max(p.height for p in panels) + gap
    img = Image.new("RGB", (w, h), (10, 9, 7))
    d = ImageDraw.Draw(img)
    d.text((12, 8), title, fill=INK, font=_font(20))
    d.text((12, 30), subtitle, fill=(150, 145, 130), font=_font(12))
    x = gap
    for p in panels:
        img.paste(p, (x, top_h))
        x += p.width + gap
    return img


# ---------------------------------------------------------------------------
# (a) phase-switch pair — colibrí phase vs globe phase, switched by tilt
# ---------------------------------------------------------------------------

# The phase-switch lives at a COARSER period than the 4–20 µm shimmer carriers:
# the A→B switch happens after parallax walks half a period, and parallax is
# ~5.98 µm/deg, so a comfortable 3–8° switch needs period/2 ≈ 18–48 µm →
# PERIOD ≈ 36–96 µm. We sweep that switch-relevant band here (a twitchy fine
# one, the comfortable middle, a stiff coarse one) rather than 4–20 µm, where
# the switch would fire at <2° (unusable). This IS the key finding for the
# centerpiece carrier.
SWITCH_PERIODS_UM = (24.0, 48.0, 60.0, 72.0, 96.0)


def _interlace_switch(period_um: float, imgA: np.ndarray, imgB: np.ndarray,
                      shift_um: float, view_px: int, view_um: float,
                      super_sample: int = 5) -> np.ndarray:
    """Render one state of a two-image phase switch.

    imgA/imgB are bool silhouettes (same shape, the viewing grid). The BACK
    layer writes imgA into the first half of every ``period`` stripe and imgB
    into the second half. The FRONT slit grating (duty 0.5) is open on the first
    half. Parallax ``shift_um`` slides the back stripes under the front slit:
    at shift 0 the slit sees imgA's stripes; at shift = period/2 it sees imgB's.
    Downsampled, that reads as imgA fading out and imgB fading in.
    """
    hi = view_px * super_sample
    xs = (np.arange(hi) - (hi - 1) / 2.0) * (view_um / hi)
    X, _Y = np.meshgrid(xs, xs)
    A = np.asarray(Image.fromarray((imgA * 255).astype(np.uint8)).resize((hi, hi), Image.NEAREST)) > 127
    B = np.asarray(Image.fromarray((imgB * 255).astype(np.uint8)).resize((hi, hi), Image.NEAREST)) > 127
    back_slot = ((X + shift_um) / period_um) % 1.0     # which half-stripe on back
    back_gold = np.where(back_slot < 0.5, A, B)         # A in slot 0, B in slot 1
    front_open = (X / period_um) % 1.0 < 0.5            # front slit open on slot 0
    visible = back_gold & front_open
    return _area_downsample(visible.astype(np.float32), (view_px, view_px))


def render_phase_switch() -> Image.Image:
    """Colibrí (A) ↔ globe (B) two-image tilt switch across the SWITCH-relevant
    coarse period band. Each panel: head-on state (left half) | tilted-to-switch
    state (right half). Green switch tilt = comfortable 3–8°.
    """
    view_px, view_um = 150, 900.0
    from ..motifs import colibri, globe
    A = colibri.colibri_silhouette((1.0, 1.0), n_grid=view_px)
    B = globe.globe_silhouette((1.0, 1.0), n_grid=view_px)
    panels = []
    for p in SWITCH_PERIODS_UM:
        ps = mo.phase_switch_tilt_deg(p)
        head = _interlace_switch(p, A, B, 0.0, view_px, view_um)
        switched = _interlace_switch(p, A, B, p / 2.0, view_px, view_um)
        combo = np.concatenate([head[:, :view_px // 2], switched[:, view_px // 2:]], axis=1)
        # FAB tile: the front SLIT grating alone (single layer) so the interlace
        # stripe structure is legible; duty 0.5 = half open, half gold.
        fab = _fab_tile(p, 0.0, px_per_um=1.6)
        flag = "ok" if ps.comfortable else ("twitchy" if ps.half_switch_tilt_deg < 3 else "stiff")
        cap = f"switch @ {ps.half_switch_tilt_deg:.1f}°  [{flag}]"
        panels.append(_panel(p, fab, combo, cap))
    return _grid_image(
        "(a) phase switch  —  colibrí (left, head-on) ↔ globe (right, tilted)",
        "SWITCH periods 24–96 µm (not the 4–20 µm shimmer band): switch fires "
        "after parallax walks ½ period. sweet spot 60–72 µm → 5–6° switch",
        panels,
    )


# ---------------------------------------------------------------------------
# (b) leaf-silhouette shimmer vs a back carrier
# ---------------------------------------------------------------------------

def _leaf_silhouette(n: int) -> np.ndarray:
    """A simple pointed leaf bool mask (n×n) — stand-in for the frame motifs."""
    ys, xs = np.mgrid[0:n, 0:n]
    x = (xs - n / 2) / (n / 2)
    y = (ys - n / 2) / (n / 2)
    # lanceolate leaf: |x| < profile(y)
    prof = np.cos(y * math.pi / 2.2) ** 0.7 * 0.55
    leaf = (np.abs(x) < prof) & (np.abs(y) < 0.95)
    return leaf


def render_leaf_shimmer() -> Image.Image:
    """A leaf silhouette filled with a front grating over a uniform back carrier;
    the fill shimmers where the two beat. Show head-on and tilted.
    """
    view_px, view_um = 150, 700.0
    leaf = _leaf_silhouette(view_px * 5)
    panels = []
    for p in CARRIER_PERIODS_UM:
        p_front = p * 1.06
        def shimmer(shift_um):
            hi = view_px * 5
            xs = (np.arange(hi) - (hi - 1) / 2.0) * (view_um / hi)
            X, Y = np.meshgrid(xs, xs)
            front = (X / p_front) % 1.0 < 0.5
            ab = math.radians(3.0)
            back = ((X + shift_um) * math.cos(ab) + Y * math.sin(ab)) / p % 1.0 < 0.5
            stack = (front | back) & leaf
            return _area_downsample(stack.astype(np.float32), (view_px, view_px))
        head = shimmer(0.0)
        shift = mo.parallax_shift_um(6.0)
        tilt = shimmer(shift)
        combo = np.concatenate([head[:, :view_px // 2], tilt[:, view_px // 2:]], axis=1)
        fab = _fab_tile(p_front, 0.0, period_b_um=p, angle_b_deg=3.0)
        beat = mo.beat_period_parallel(p_front, p)
        panels.append(_panel(p, fab, combo, f"beat {beat:.0f} µm"))
    return _grid_image(
        "(b) leaf shimmer  —  front grating in leaf vs back carrier (head-on | 6° tilt)",
        "front period = 1.06× back, +3° offset. Coarse beat = slow legible "
        "shimmer; fine period = tight busy shimmer",
        panels,
    )


# ---------------------------------------------------------------------------
# (c) moiré magnification of a small motif
# ---------------------------------------------------------------------------

def render_magnification() -> Image.Image:
    """A tiny repeated motif (period p) sampled by a grating of period 1.04×p
    produces a magnified ghost at M = p/(p−p_s). Show the ghost per carrier.
    """
    view_px, view_um = 150, 1200.0
    panels = []
    for p in CARRIER_PERIODS_UM:
        p_sample = p * 1.04
        M = mo.moire_magnification(p_sample, p)
        def ghost():
            hi = view_px * 6
            xs = (np.arange(hi) - (hi - 1) / 2.0) * (view_um / hi)
            X, Y = np.meshgrid(xs, xs)
            # image layer: a small "diamond" motif repeated at period p in both axes
            ix = (X / p) % 1.0 - 0.5
            iy = (Y / p) % 1.0 - 0.5
            motif = (np.abs(ix) + np.abs(iy)) < 0.30
            # sampling layer: dot lattice at p_sample
            sx = (X / p_sample) % 1.0 - 0.5
            sy = (Y / p_sample) % 1.0 - 0.5
            sample = (sx * sx + sy * sy) < 0.25 * 0.25
            stack = (motif | sample).astype(np.float32)
            return _area_downsample(stack, (view_px, view_px))
        fab = _fab_tile(p, 0.0, px_per_um=6.0, period_b_um=p_sample)
        panels.append(_panel(p, fab, ghost(), f"M = {M:.0f}× (ghost {abs(M) * p:.0f} µm)"))
    return _grid_image(
        "(c) moiré magnification  —  motif pitch p sampled at 1.04·p → ghost |M|=25×",
        "small 2-D motif + slightly-mismatched sampling lattice → large floating "
        "ghost copy. |M| grows as mismatch shrinks (twitchier)",
        panels,
    )


# ---------------------------------------------------------------------------
# (d) scanimation 3-phase motion
# ---------------------------------------------------------------------------

def render_scanimation() -> Image.Image:
    """3-phase barrier-grid animation: back interleaves 3 frames of a dot moving
    across; front slit (duty 1/3) reveals one. Show the 3 revealed frames as a
    tilt sequence, per carrier period (== frame pitch here, ×N for the slots).
    """
    view_px, view_um = 150, 900.0
    n_phases = 3
    panels = []
    for p in CARRIER_PERIODS_UM:
        pitch = p * n_phases  # frame pitch holds 3 slots of width p (each >= floor)
        plan = mo.scanimation_plan(n_phases, pitch)

        def frame(k, w_px, h_px):
            """One revealed frame (w_px×h_px): the back interleaves 3 bar-frames
            (bar steps DOWN with k); the front slit (⅓ duty) reveals slot==k, so
            only frame k's bar shows through. A thin divider is added by caller."""
            sy, sx = h_px * 6, w_px * 6
            xs = (np.arange(sx) - (sx - 1) / 2.0) * (view_um / sx)
            ys = ((sy - 1) / 2.0 - np.arange(sy)) * (view_um / sy)
            X, Y = np.meshgrid(xs, ys)
            slot = np.floor((X / pitch % 1.0) * n_phases).astype(int)  # 0,1,2
            reveal = (slot == k)
            yc = (1 - k) * (view_um * 0.26)            # +0.26→0→−0.26 across k
            bar = np.abs(Y - yc) < view_um * 0.11
            vis = (reveal & bar).astype(np.float32)
            return _area_downsample(vis, (h_px, w_px))
        # three revealed frames as a 3-up strip (full height); the bright bar
        # steps down left→right = the animation advancing with tilt.
        fw = view_px // 3
        strip = [frame(k, fw, view_px) for k in range(3)]
        div = np.zeros((view_px, 2), dtype=np.float32)
        combo = np.concatenate([strip[0], div, strip[1], div, strip[2]], axis=1)
        fab = _fab_tile(pitch, 0.0, px_per_um=2.5)
        cap = f"{plan.tilt_per_frame_deg:.1f}°/frame"
        panels.append(_panel(p, fab, combo, cap))
    return _grid_image(
        "(d) scanimation (3-phase)  —  three tilt frames stitched left→right",
        "back interleaves 3 frames; front slit (⅓ duty) reveals one; tilt "
        "advances the frame. bar steps up per frame = motion",
        panels,
    )


# ---------------------------------------------------------------------------
# (e) radial rosette shimmer
# ---------------------------------------------------------------------------

def render_rosette() -> Image.Image:
    """A radial ring grating over a linear back carrier → a circular moiré
    rosette that pinwheels with tilt. Show head-on and tilted.
    """
    view_px, view_um = 150, 900.0
    panels = []
    for p in CARRIER_PERIODS_UM:
        def rosette(shift_um):
            hi = view_px * 6
            xs = (np.arange(hi) - (hi - 1) / 2.0) * (view_um / hi)
            X, Y = np.meshgrid(xs, xs)
            R = np.hypot(X, Y)
            rings = (R / p) % 1.0 < 0.5
            back = ((X + shift_um) / p) % 1.0 < 0.5
            stack = (rings | back).astype(np.float32)
            return _area_downsample(stack, (view_px, view_px))
        head = rosette(0.0)
        tilt = rosette(mo.parallax_shift_um(6.0))
        combo = np.concatenate([head[:, :view_px // 2], tilt[:, view_px // 2:]], axis=1)
        fab = gr.radial_grating_mask(max(p, 4.0), (600.0, 600.0), coarsen=True)
        # magnified fab tile from the radial generator (center region)
        m = fab.mask
        c = m.shape[0] // 2
        half = min(65, c)
        tile = m[c - half:c + half, c - half:c + half].astype(np.float32)
        panels.append(_panel(p, tile, combo, "rings × carrier"))
    return _grid_image(
        "(e) radial rosette shimmer  —  concentric rings vs linear carrier (head-on | tilt)",
        "circular moiré rosette; pinwheels/breathes with tilt. fine rings add "
        "diffraction sparkle at the rim",
        panels,
    )


# ---------------------------------------------------------------------------
# summary strip: physics numbers per period
# ---------------------------------------------------------------------------

def render_physics_table() -> Image.Image:
    rows = []
    for p in CARRIER_PERIODS_UM:
        ps = mo.phase_switch_tilt_deg(p)
        diff = mo.diffraction_onset(p)
        beat = mo.beat_period_parallel(p * 1.06, p)
        rows.append((p, ps.half_switch_tilt_deg, ps.comfortable, beat, diff.note))
    w, rowh, top = 940, 26, 60
    img = Image.new("RGB", (w, top + rowh * (len(rows) + 1) + 10), (10, 9, 7))
    d = ImageDraw.Draw(img)
    d.text((12, 8), "physics summary per carrier period", fill=INK, font=_font(18))
    d.text((12, 34), "500 µm fused silica, n=1.46, parallax ≈ 5.98 µm/deg",
           fill=(150, 145, 130), font=_font(12))
    f = _font(13)
    hdr = ["period µm", "½-switch tilt", "comfy?", "beat(1.06×) µm", "diffraction (0.4–0.7µm first order)"]
    xs = [12, 110, 250, 330, 470]
    for x, htxt in zip(xs, hdr):
        d.text((x, top), htxt, fill=(200, 190, 160), font=f)
    for i, (p, tilt, comfy, beat, note) in enumerate(rows):
        y = top + rowh * (i + 1)
        c = (120, 220, 120) if comfy else (220, 140, 120)
        d.text((xs[0], y), f"{p:g}", fill=INK, font=f)
        d.text((xs[1], y), f"{tilt:.1f}°", fill=c, font=f)
        d.text((xs[2], y), "yes" if comfy else "no", fill=c, font=f)
        d.text((xs[3], y), f"{beat:.0f}", fill=INK, font=f)
        d.text((xs[4], y), note[:60], fill=(190, 185, 170), font=_font(11))
    return img


ALL_DEMOS = {
    "fx_a_phase_switch": render_phase_switch,
    "fx_b_leaf_shimmer": render_leaf_shimmer,
    "fx_c_magnification": render_magnification,
    "fx_d_scanimation": render_scanimation,
    "fx_e_rosette": render_rosette,
    "fx_physics_table": render_physics_table,
}


def _run_demos(out_dir: str) -> list[str]:
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, fn in ALL_DEMOS.items():
        img = fn()
        path = os.path.join(out_dir, name + ".png")
        img.save(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for p in _run_demos(out):
        print(p)
