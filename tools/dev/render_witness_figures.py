"""Figures for the whitepaper that a formula alone does not carry.

Diagrams are inline SVG so they render in both themes. Simulated panels use the
same appearance model as the box renderer — specular gold (eta0 = 0.25) plus
the first-order sheen (eta1 = 0.10) from the CIE table — because a render that
drops the specular shows black wherever the first order leaves the visible band.

    uv run --directory backend python ../tools/dev/render_witness_figures.py OUTDIR
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app import witness_moire as wm  # noqa: E402
from app.export_witness import SWITCH_COMB_LADDER_UM  # noqa: E402
from app.witness_geom import (BOX_CARRIER_UM, BOX_COMB_UM, GLASS_MATERIAL, GLASS_N,  # noqa: E402
                              PARALLAX_UM_PER_DEG, PLY_UM, swap_deg)
from app.diffraction import bake_lut, lamellar_order_efficiency  # noqa: E402
from app.patterns.bitmap.colourzone import hue_ladder  # noqa: E402
from app.patterns.bitmap.imageprep import linear_to_srgb, srgb_to_linear  # noqa: E402

BG = (13, 17, 19)
FG = (214, 224, 227)
DIM = (150, 165, 172)
ACC = (79, 185, 203)
WARN = (224, 138, 114)
OK = (95, 185, 133)
TAN = (200, 170, 120)
GOLD = np.array([0.84, 0.66, 0.29], np.float32)
LUT = bake_lut(duty=0.5, u_max_um=10.0, size=1024)
ETA0 = lamellar_order_efficiency(0, 0.5)
ETA1 = lamellar_order_efficiency(1, 0.5)
K0, GAIN = 0.11, 0.42
"""Geometry factor sin(theta_out) - sin(theta_in) at rest, and its rate with
tilt. K0 = 0.11 puts 550 nm from a 5 um pitch at the viewer with the lamp about
65 deg off the normal and the eye near it; the hue then WALKS with tilt. With
the lamp behind the viewer it would not."""


def font(sz=12):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def sheen(u):
    i = np.clip(np.abs(u) / 10.0, 0, 1) * (LUT.shape[0] - 1)
    return LUT[i.astype(np.int32)]


def _canvas(w, h):
    im = Image.new("RGB", (w, h), BG)
    return im, ImageDraw.Draw(im)


# --------------------------------------------------------------- swatch
def swatch_png(out: Path):
    TILTS = (0.0, 0.5, 1.0, 2.0)
    rows = []
    for spread in (1.20, 1.45, 1.90):
        strip, d = _canvas(4 * 214 + 8, 5 * 44 + 66)
        for j, base in enumerate((4.0, 5.0, 6.5, 8.0)):
            lad = hue_ladder(12, spread)
            for r in range(5):
                for i, sc in enumerate(lad):
                    p_ = base * sc
                    if r < 4:
                        u = p_ * (K0 + math.sin(math.radians(TILTS[r])) * GAIN)
                        col = (GOLD * ETA0 + sheen(np.array([u], np.float32))[0] * ETA1) / ETA0
                    else:
                        sh = sheen(np.array([p_ * K0], np.float32))[0]
                        col = sh / max(1e-6, float(sh.max()))
                    rgb = tuple(int(v * 255) for v in np.clip(linear_to_srgb(np.clip(col, 0, 1)), 0, 1))
                    x0 = 4 + j * 214 + i * 17
                    d.rectangle([x0, 4 + r * 44, x0 + 16, 4 + r * 44 + 42], fill=rgb)
            printable = base * min(lad) * 0.5 >= 2.0
            d.text((6 + j * 214, 5 * 44 + 8),
                   f"SW {base:g}/{spread:.2f}" + ("" if printable else "   blue end < 2 um"),
                   fill=DIM if printable else WARN, font=font(11))
        dl = 550.0 * (math.sqrt(spread) - 1.0 / math.sqrt(spread))
        d.text((6, 5 * 44 + 26),
               f"D-SWATCH  spread {spread:.2f}  ->  ladder spans about {dl:.0f} nm at a fixed view.  "
               f"Rows: tilt 0 / 0.5 / 1 / 2 deg as seen;  bottom row: first order alone",
               fill=FG, font=font(12))
        d.text((6, 5 * 44 + 44),
               "Blue end left (shortest pitch). As seen, the specular gold (eta0 = 0.25) carries the "
               "first order (eta1 = 0.10) -- pastel by physics.", fill=DIM, font=font(10))
        rows.append(strip)
    im = Image.new("RGB", (rows[0].width, sum(r.height + 6 for r in rows) + 6), BG)
    y = 6
    for r in rows:
        im.paste(r, (0, y))
        y += r.height + 6
    im.save(out)
    print("  saved", out, im.size)


# --------------------------------------------------------------- harmonic chart
def _amp(k, c):
    x = math.pi * k * c
    return abs(c * math.sin(x) / x) if x else 0.0


def visible_beats(screen_um: float = 44.0, carrier_um: float = BOX_CARRIER_UM):
    """The screen-over-carrier beats the eye resolves (> 87 um), split into the
    ones with an EVEN carrier harmonic (they vanish at exactly 50% duty) and
    the rest (present at every duty)."""
    rows = [t for t in wm.harmonic_beats(screen_um, carrier_um, 0.42, max_order=4) if t["beat_um"] > 87.0]
    even = sorted([t for t in rows if t["n"] % 2 == 0], key=lambda t: -t["amplitude"])
    odd = sorted([t for t in rows if t["n"] % 2 == 1], key=lambda t: -t["amplitude"])
    return even, odd


def harmonic_chart_png(out: Path):
    """Modulation of the visible screen-over-carrier beats against the screen's
    local duty (= the picture's local tone), for THIS glass's carrier.

    Two families. An even-n beat (carrier harmonic n even) has amplitude
    a_m(c_s) a_n(c_c): with the carrier held at 0.50 it is exactly zero at every
    tone, so it appears only through process bias (both gratings at c, dashed).
    An odd-n beat is present at every duty and scales with the screen's own
    harmonic a_m(c_s) -- that is the texture a photograph carries regardless.
    """
    even, odd = visible_beats()
    e = even[0] if even else None
    o = odd[0] if odd else None
    cs = np.linspace(0.05, 0.95, 181)
    W, H, L, B, top = 760, 320, 60, 44, 45.0
    im, d = _canvas(W, H + 96)

    def X(c):
        return L + (c - 0.05) / 0.90 * (W - 20 - L)

    def Y(v):
        return H - B - min(v * 100.0, top) / top * (H - B - 10)

    d.rectangle([L, 10, W - 20, H - B], outline=(58, 68, 73))
    for y_pct in range(0, 46, 5):
        d.line([L, Y(y_pct / 100), W - 20, Y(y_pct / 100)], fill=(40, 48, 52))
        d.text((8, Y(y_pct / 100) - 7), f"{y_pct:2d}%", fill=DIM, font=font(11))
    lines = []
    if o is not None:
        # tone case in transmission: screen at c, carrier at 0.50; mean (1-c)(0.5)
        tone_T = np.array([2 * _amp(o["m"], c) * _amp(o["n"], 0.5) / ((1 - c) * 0.5) for c in cs])
        tone_R = tone_T * (1 - cs) / (1 + cs)
        d.line([(X(c), Y(v)) for c, v in zip(cs, tone_T)], fill=WARN, width=3)
        d.line([(X(c), Y(v)) for c, v in zip(cs, tone_R)], fill=TAN, width=1)
        for c in (0.20, 0.42, 0.50, 0.65):
            v = 2 * _amp(o["m"], c) * _amp(o["n"], 0.5) / ((1 - c) * 0.5)
            d.ellipse([X(c) - 4, Y(v) - 4, X(c) + 4, Y(v) + 4], fill=WARN)
            d.text((X(c) + 6, Y(v) - 16), f"{v*100:.1f}%", fill=WARN, font=font(10))
        lines.append((WARN, f"solid: the ({o['m']},{o['n']}) beat at {o['beat_um']:.0f} um ({o['arcmin']:.1f}'), TONE case in transmission "
                            f"(screen at c, carrier at 0.50): present at every duty, {tone_T[cs.searchsorted(0.5)]*100:.0f}% at c = 0.50"))
        lines.append((TAN, "thin: the same beat in REFLECTION off gold, x (1-c)/(1+c)"))
    if e is not None:
        bias_c = cs[(cs >= 0.30) & (cs <= 0.70)]
        bias_T = np.array([2 * _amp(e["m"], c) * _amp(e["n"], c) / (1 - c) ** 2 for c in bias_c])
        pts = [(X(c), Y(v)) for c, v in zip(bias_c, bias_T)]
        for k in range(0, len(pts) - 1, 4):
            d.line(pts[k:k + 3], fill=ACC, width=2)
        for c in (0.42, 0.58):
            v = 2 * _amp(e["m"], c) * _amp(e["n"], c) / (1 - c) ** 2
            d.ellipse([X(c) - 4, Y(v) - 4, X(c) + 4, Y(v) + 4], fill=ACC)
            d.text((X(c) + 6, Y(v) + 4), f"{v*100:.1f}%", fill=ACC, font=font(10))
        v42 = 2 * _amp(e["m"], 0.42) * _amp(e["n"], 0.42) / 0.58 ** 2
        v58 = 2 * _amp(e["m"], 0.58) * _amp(e["n"], 0.58) / 0.42 ** 2
        lines.append((ACC, f"dashed: the ({e['m']},{e['n']}) beat at {e['beat_um']:.0f} um ({e['arcmin']:.1f}'), BIAS case (both gratings at c, "
                           f"as HARM is built): {v42*100:.1f}% / 0 / {v58*100:.1f}% at 0.42 / 0.50 / 0.58 -- zero at 0.50 at EVERY tone"))
    d.line([X(0.5), 10, X(0.5), H - B], fill=(58, 68, 73), width=1)
    d.line([L, Y(0.005), W - 20, Y(0.005)], fill=OK, width=1)
    d.text((W - 200, Y(0.005) - 14), "~0.5-1% visible", fill=OK, font=font(10))
    for c in (0.1, 0.3, 0.5, 0.7, 0.9):
        d.text((X(c) - 10, H - B + 6), f"{c:.1f}", fill=DIM, font=font(11))
    d.text((L, H - B + 22), "local duty c of the 44 um screen  (= local tone of the picture)", fill=DIM, font=font(11))
    d.text((8, H + 4), f"2A/I of the visible beats, 44 um screen over the {BOX_CARRIER_UM:g} um carrier ({PLY_UM/1000:g} mm {GLASS_MATERIAL})", fill=FG, font=font(12))
    y = H + 22
    for col, txt in lines:
        d.text((8, y), txt, fill=col, font=font(10))
        y += 16
    d.text((8, y), f"Even carrier harmonics vanish only at exactly 0.50: process bias creates that beat, it does not move it. Caveat: {BOX_CARRIER_UM:g}/44 = {BOX_CARRIER_UM/44:.2f} is commensurate,", fill=DIM, font=font(10))
    d.text((8, y + 16), "so the odd (3,7) term shares the (1,2) period and leaves a few-percent floor at 0.50 -- a minimum, not a null.", fill=DIM, font=font(10))
    im.save(out)
    print("  saved", out, im.size)


# --------------------------------------------------------------- P-SWAP plot
def swap_plot_png(out: Path):
    ps = np.linspace(50, 400, 176)

    def swap(p, t, n):
        return np.degrees(np.arcsin(np.clip(n * np.sin(np.arctan((p / 4) / t)), -1, 1)))

    this = swap(ps, PLY_UM, GLASS_N)
    ref = swap(ps, 1500.0, 1.52)
    lane300 = ps / 2 / 300000.0 * (180 / math.pi) * 60
    lane200 = ps / 2 / 200000.0 * (180 / math.pi) * 60
    W, H, L, R, B = 760, 320, 60, 60, 44
    im, d = _canvas(W, H + 60)

    def X(p):
        return L + (p - 50) / 350 * (W - L - R)

    def YL(deg):
        return H - B - deg / 6.0 * (H - B - 10)

    def YR(am):
        return H - B - am / 2.4 * (H - B - 10)

    d.rectangle([L, 10, W - R, H - B], outline=(58, 68, 73))
    for dg in range(0, 7):
        d.line([L, YL(dg), W - R, YL(dg)], fill=(40, 48, 52))
        d.text((8, YL(dg) - 7), f"{dg} deg", fill=DIM, font=font(10))
    for am in (0.5, 1.0, 1.5, 2.0):
        d.text((W - R + 6, YR(am) - 7), f"{am:.1f}'", fill=DIM, font=font(11))
    d.line([(X(p), YL(v)) for p, v in zip(ps, this)], fill=ACC, width=3)
    d.line([(X(p), YL(v)) for p, v in zip(ps, ref)], fill=WARN, width=1)
    d.line([(X(p), YR(v)) for p, v in zip(ps, lane300)], fill=TAN, width=2)
    pts = [(X(p), YR(v)) for p, v in zip(ps, lane200)]
    for k in range(0, len(pts) - 1, 4):
        d.line(pts[k:k + 3], fill=TAN, width=1)
    d.line([L, YR(1.0), W - R, YR(1.0)], fill=OK, width=1)
    d.text((L + 6, YR(1.0) - 14), "1 arcmin: the lane becomes visible at 300 mm (p = 174 um)", fill=OK, font=font(10))
    for p in SWITCH_COMB_LADDER_UM:
        d.line([X(p), 10, X(p), H - B], fill=(58, 68, 73))
        d.text((X(p) - 22, 14), f"SWAP {p:g}", fill=FG, font=font(10))
        d.text((X(p) - 20, 28), f"{swap_deg(p):.2f} deg", fill=DIM, font=font(10))
    for p in (100, 200, 300, 400):
        d.text((X(p) - 12, H - B + 6), f"{p}", fill=DIM, font=font(11))
    d.text((L, H - B + 22), "comb pitch p (um)", fill=DIM, font=font(11))
    d.text((8, H + 4), "P-SWAP: swap angle (left axis) and lane subtense (right axis) against comb pitch", fill=FG, font=font(12))
    d.text((8, H + 22), f"swap = asin(n sin(atan(p/4t)))  --  this plate and box (blue, {PLY_UM/1000:g} mm {GLASS_MATERIAL}, "
           f"{PARALLAX_UM_PER_DEG:.1f} um/deg); thin orange: the 1.5 mm soda-lime design for reference.  Lane p/2D at 300 mm (solid) and 200 mm (dashed).", fill=DIM, font=font(10))
    d.text((8, H + 38), f"The box comb is {BOX_COMB_UM:g} um: lane {BOX_COMB_UM/2:.0f} um = {BOX_COMB_UM/2/300000*180/math.pi*60:.2f}' at 300 mm, "
           f"a barrier the eye can see. Swap at {swap_deg(BOX_COMB_UM):.2f} deg by construction; SWAP 350 is coarser still.", fill=DIM, font=font(10))
    im.save(out)
    print("  saved", out, im.size)


# --------------------------------------------------------------- H-WEDGE
def wedge_png(out: Path):
    strips = []
    for sp in (20.0, 44.0, 60.0):
        a = wm.build_step_wedge(0, 0, 30000.0, 4000.0, line_period_um=sp,
                                tone_steps=min(22, int(sp / 2)))
        px = 2.0
        nx, ny = int(30000 / px), int(4000 / px)
        g = np.zeros((ny, nx), np.float32)
        for x0, x1, y0, y1 in a.front:
            c0, c1 = int((x0 + 15000) / px), int(math.ceil((x1 + 15000) / px))
            r0, r1 = int((2000 - y1) / px), int(math.ceil((2000 - y0) / px))
            g[max(0, r0):min(ny, r1), max(0, c0):min(nx, c1)] = 1.0
        cov = g.mean(axis=0)
        strip, d = _canvas(760, 96)
        for i in range(0, nx, 4):
            v = float(np.clip(linear_to_srgb(np.array([cov[i]]))[0], 0, 1))
            xx = 20 + i * 720 / nx
            d.line([xx, 6, xx, 50], fill=(int(v * 214), int(v * 168), int(v * 74)), width=1)
        levels = a.stats["levels"]
        pw = 720 / len(levels)
        seen = set()
        for i, lv in enumerate(levels):
            dup = lv in seen
            seen.add(lv)
            d.text((22 + i * pw, 54), f"{lv/a.stats['tone_steps']:.2f}", fill=WARN if dup else DIM, font=font(9))
        d.text((20, 74), f"WEDGE {sp:g}um  --  {a.stats['tone_steps']} levels; {len(set(levels))} distinct of 16 patches"
               + ("   (repeats in orange: read as within-cell uniformity)" if len(set(levels)) < 16 else ""),
               fill=FG, font=font(11))
        strips.append(strip)
    W, H, L, B = 760, 300, 50, 40
    chart, d = _canvas(W, H)

    def X(v):
        return L + v * (W - 20 - L)

    def Y(v):
        return H - B - v * (H - B - 10)

    d.rectangle([L, 10, W - 20, H - B], outline=(58, 68, 73))
    d.line([X(0), Y(0), X(1), Y(1)], fill=(58, 68, 73), width=1)
    xs = np.linspace(0, 1, 101)
    d.line([(X(v), Y(float(srgb_to_linear(np.array([v]))[0]))) for v in xs], fill=WARN, width=3)
    d.ellipse([X(0.25) - 4, Y(0.054) - 4, X(0.25) + 4, Y(0.054) + 4], fill=WARN)
    d.ellipse([X(0.54) - 4, Y(0.25) - 4, X(0.54) + 4, Y(0.25) + 4], fill=ACC)
    d.text((X(0.25) + 8, Y(0.054) - 16), "sRGB 0.25 -> coverage 0.054", fill=WARN, font=font(10))
    d.text((X(0.54) - 200, Y(0.25) - 16), "coverage 0.25 reads back as sRGB 0.54", fill=ACC, font=font(10))
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        d.text((X(v) - 10, H - B + 6), f"{v:.2f}", fill=DIM, font=font(10))
        d.text((8, Y(v) - 7), f"{v:.2f}", fill=DIM, font=font(10))
    d.text((L, H - B + 22), "source value (sRGB)   --   y: metal coverage", fill=DIM, font=font(11))
    d.text((L + 8, 14), "coverage = linearise(sRGB); the identity line is the classic error", fill=FG, font=font(11))
    im = Image.new("RGB", (760, sum(s.height + 6 for s in strips) + 6 + chart.height + 6), BG)
    y = 6
    for s in strips:
        im.paste(s, (0, y))
        y += s.height + 6
    im.paste(chart, (0, y))
    im.save(out)
    print("  saved", out, im.size)


# --------------------------------------------------------------- SVG diagrams
SVG_STACK = """<svg viewBox="0 0 760 310" xmlns="http://www.w3.org/2000/svg" font-family="IBM Plex Mono, monospace" font-size="12">
  <defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <g stroke="currentColor" fill="none" stroke-width="1.5">
    <rect x="80" y="70" width="520" height="14" fill="currentColor" fill-opacity=".08"/>
    <rect x="80" y="200" width="520" height="14" fill="currentColor" fill-opacity=".08"/>
    <rect x="80" y="84" width="520" height="116" fill="currentColor" fill-opacity=".03" stroke-dasharray="4 3"/>
  </g>
  <g fill="currentColor">
    <rect x="94" y="70" width="28" height="14"/><rect x="150" y="70" width="28" height="14"/><rect x="206" y="70" width="28" height="14"/><rect x="262" y="70" width="28" height="14"/><rect x="318" y="70" width="28" height="14"/><rect x="374" y="70" width="28" height="14"/><rect x="430" y="70" width="28" height="14"/><rect x="486" y="70" width="28" height="14"/><rect x="542" y="70" width="28" height="14"/>
  </g>
  <g fill="currentColor" fill-opacity=".9">
    <rect x="80" y="200" width="28" height="14"/><rect x="136" y="200" width="28" height="14"/><rect x="192" y="200" width="28" height="14"/><rect x="248" y="200" width="28" height="14"/><rect x="304" y="200" width="28" height="14"/><rect x="360" y="200" width="28" height="14"/><rect x="416" y="200" width="28" height="14"/><rect x="472" y="200" width="28" height="14"/><rect x="528" y="200" width="28" height="14"/>
  </g>
  <g fill="currentColor" fill-opacity=".35">
    <rect x="108" y="200" width="28" height="14"/><rect x="164" y="200" width="28" height="14"/><rect x="220" y="200" width="28" height="14"/><rect x="276" y="200" width="28" height="14"/><rect x="332" y="200" width="28" height="14"/><rect x="388" y="200" width="28" height="14"/><rect x="444" y="200" width="28" height="14"/><rect x="500" y="200" width="28" height="14"/><rect x="556" y="200" width="28" height="14"/>
  </g>
  <g stroke="currentColor" stroke-width="1.5" fill="none" marker-end="url(#a)" color="currentColor">
    <path d="M 304 20 L 304 70"/>
    <path d="M 304 84 L 318 200" stroke-dasharray="5 3"/>
  </g>
  <g stroke="currentColor" stroke-width="1" fill="none" opacity=".6">
    <path d="M 304 84 L 304 200" stroke-dasharray="2 4"/>
    <path d="M 304 226 L 318 226" marker-end="url(#a)"/>
  </g>
  <g fill="currentColor">
    <text x="620" y="82">front ply: comb, slit = p/2</text>
    <text x="620" y="206">back ply: lanes A (dark) B (light), p/2 each</text>
    <text x="620" y="146">gap t, index n</text>
    <text x="240" y="40">tilt θ (in air)</text>
    <text x="330" y="150">θ′ = asin(sin θ / n)</text>
    <text x="330" y="246">shift = t · tan θ′ = p/4 → clean A;  −p/4 → clean B</text>
    <text x="80" y="275" opacity=".75">Head-on the slit straddles the A/B boundary: a 50/50 blend. __PARALLAX__ µm/° through the __PLY__ mm ply, box and plate alike.</text>
    <text x="80" y="295" opacity=".75">The plies do not move; the line of sight does.</text>
  </g>
</svg>"""

SVG_UNION = """<svg viewBox="0 0 760 250" xmlns="http://www.w3.org/2000/svg" font-family="IBM Plex Mono, monospace" font-size="12">
  <g fill="currentColor">
    <text x="30" y="24" font-weight="700">one plane: metal = A₁ ∪ A₂</text>
    <text x="410" y="24" font-weight="700">two plies at zero gap: T = (1−A₁)(1−A₂)</text>
  </g>
  <g fill="currentColor" fill-opacity=".9">
    <rect x="30" y="50" width="22" height="40"/><rect x="82" y="50" width="22" height="40"/><rect x="134" y="50" width="22" height="40"/><rect x="186" y="50" width="22" height="40"/><rect x="238" y="50" width="22" height="40"/><rect x="290" y="50" width="22" height="40"/>
  </g>
  <g fill="currentColor" fill-opacity=".45">
    <rect x="30" y="50" width="22" height="40"/><rect x="87" y="50" width="22" height="40"/><rect x="144" y="50" width="22" height="40"/><rect x="201" y="50" width="22" height="40"/><rect x="258" y="50" width="22" height="40"/><rect x="315" y="50" width="22" height="40"/>
  </g>
  <g fill="currentColor" fill-opacity=".9">
    <rect x="410" y="42" width="22" height="18"/><rect x="462" y="42" width="22" height="18"/><rect x="514" y="42" width="22" height="18"/><rect x="566" y="42" width="22" height="18"/><rect x="618" y="42" width="22" height="18"/><rect x="670" y="42" width="22" height="18"/>
    <rect x="410" y="72" width="22" height="18"/><rect x="467" y="72" width="22" height="18"/><rect x="524" y="72" width="22" height="18"/><rect x="581" y="72" width="22" height="18"/><rect x="638" y="72" width="22" height="18"/><rect x="695" y="72" width="22" height="18"/>
  </g>
  <g fill="currentColor" opacity=".85">
    <text x="30" y="120">clear where NEITHER has metal: 1 − 1[A₁∪A₂]</text>
    <text x="410" y="120">light passes where BOTH are clear: (1 − 1[A₁])(1 − 1[A₂])</text>
    <text x="30" y="150">inclusion–exclusion: 1 − A₁ − A₂ + A₁A₂ — the A₁A₂ term is the beat, and it is identical in both.</text>
    <text x="30" y="176">Assumptions that make the identity optical: binary amplitude masks · geometric optics (no diffraction between the planes)</text>
    <text x="30" y="194">· zero gap, so the two lattices hold a fixed relative phase. Open a gap and the phase rides on the line of sight (parallax);</text>
    <text x="30" y="212">open it past the near-field limit and the product stops being a product.</text>
    <text x="30" y="238" opacity=".7">In reflection off chrome the bright regions are the metal UNION rather than the clear intersection — same cross term, inverted brightness.</text>
  </g>
</svg>"""

SVG_CELL = """<svg viewBox="0 0 520 220" xmlns="http://www.w3.org/2000/svg" font-family="IBM Plex Mono, monospace" font-size="12">
  <g stroke="currentColor" fill="none">
    <rect x="60" y="20" width="300" height="140" stroke-width="1.5" stroke-dasharray="3 3" opacity=".6"/>
    <rect x="60" y="20" width="300" height="140" fill="currentColor" fill-opacity=".05"/>
  </g>
  <g fill="currentColor">
    <rect x="62" y="170" width="6" height="8"/><rect x="70" y="170" width="6" height="8"/><rect x="80" y="170" width="6" height="8"/><rect x="92" y="170" width="6" height="8"/><rect x="100" y="170" width="6" height="8"/><rect x="108" y="170" width="6" height="8"/>
    <text x="124" y="178" font-weight="700">SP 44um</text>
    <text x="380" y="40">cell box (outline layer only)</text>
    <text x="380" y="62">drawn geometry = clear data:</text>
    <text x="380" y="84">chrome comes OFF here</text>
    <text x="380" y="180">label: chrome, bottom-left,</text>
    <text x="380" y="198">carries the parameter VALUE</text>
    <text x="60" y="210" opacity=".7">label band scales with the cell (0.4–0.9 mm); rows are packed by height, so a raster scan finds every label in the same corner</text>
  </g>
</svg>"""


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    swatch_png(out / "witness_swatch.png")
    harmonic_chart_png(out / "witness_harmonic.png")
    swap_plot_png(out / "witness_swap.png")
    wedge_png(out / "witness_wedge.png")
    stack = SVG_STACK.replace("__PARALLAX__", f"{PARALLAX_UM_PER_DEG:.2f}").replace("__PLY__", f"{PLY_UM/1000:g}")
    for name, s in (("fig_stack.svg", stack), ("fig_union.svg", SVG_UNION), ("fig_cell.svg", SVG_CELL)):
        (out / name).write_text(s, encoding="utf-8")
    print("  saved 3 SVG diagrams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
