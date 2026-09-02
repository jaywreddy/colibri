"""Figures the whitepaper was missing: diagrams as SVG, and a swatch simulation
that actually shows hue.

Three diagrams the review found absent — the two-ply stack with a tilted ray,
union-vs-product on one plane against two, and a cell's anatomy with the etched
label — are written as inline SVG so they render in both themes. The swatch
matrix is re-rendered with spectral colour per rung from the CIE table: the
earlier panel drew metal coverage as flat gold, which for an instrument whose
whole purpose is hue separation showed nothing at all.

    uv run --directory backend python ../tools/dev/render_witness_figures.py OUTDIR
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.diffraction import bake_lut, lamellar_order_efficiency  # noqa: E402
from app.patterns.bitmap.colourzone import hue_ladder  # noqa: E402
from app.patterns.bitmap.imageprep import linear_to_srgb  # noqa: E402

BG = (13, 17, 19)
FG = (214, 224, 227)
DIM = (150, 165, 172)
GOLD = np.array([0.84, 0.66, 0.29], np.float32)
LUT = bake_lut(duty=0.5, u_max_um=10.0, size=1024)
ETA0 = lamellar_order_efficiency(0, 0.5)
ETA1 = lamellar_order_efficiency(1, 0.5)
K0, GAIN = 0.11, 0.42


def font(sz=12):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def sheen(u):
    i = np.clip(np.abs(u) / 10.0, 0, 1) * (LUT.shape[0] - 1)
    return LUT[i.astype(np.int32)]


def swatch_png(out: Path):
    """Each rung coloured by its own period at three tilts: gold specular + sheen."""
    rows = []
    for spread in (1.20, 1.45, 1.90):
        strip = Image.new("RGB", (4 * 214 + 8, 3 * 44 + 62), BG)
        d = ImageDraw.Draw(strip)
        for j, base in enumerate((4.0, 5.0, 6.5, 8.0)):
            lad = hue_ladder(12, spread)
            for r, tilt in enumerate((-4.0, 0.0, 4.0)):
                for i, sc in enumerate(lad):
                    dper = base * sc
                    u = dper * (K0 + math.sin(math.radians(tilt)) * GAIN)
                    col = (GOLD * ETA0 + sheen(np.array([u], np.float32))[0] * ETA1) / ETA0
                    rgb = tuple(int(v * 255) for v in np.clip(linear_to_srgb(np.clip(col, 0, 1)), 0, 1))
                    x0 = 4 + j * 214 + i * 17
                    y0 = 4 + r * 44
                    d.rectangle([x0, y0, x0 + 16, y0 + 42], fill=rgb)
            printable = base * min(lad) * 0.5 >= 2.0
            d.text((6 + j * 214, 3 * 44 + 8),
                   f"base {base:g} um" + ("" if printable else "  blue end < floor"),
                   fill=DIM if printable else (224, 138, 114), font=font(11))
        d.text((6, 3 * 44 + 26), f"D-SWATCH  spread {spread:.2f}   rows: tilt -4 / 0 / +4 deg",
               fill=FG, font=font(12))
        d.text((6, 3 * 44 + 42),
               "12 rungs, red end left. Specular gold + first-order sheen at a "
               "lamp; colour is what separates the rungs, and the ratio survives the tilt",
               fill=DIM, font=font(10))
        rows.append(strip)
    im = Image.new("RGB", (rows[0].width, sum(r.height + 6 for r in rows) + 6), BG)
    y = 6
    for r in rows:
        im.paste(r, (0, y))
        y += r.height + 6
    im.save(out)
    print("  saved", out, im.size)


def harmonic_chart_png(out: Path):
    """Michelson contrast of the (2,3) beat vs the SCREEN's local duty."""
    def amp(k, c):
        x = math.pi * k * c
        return abs(c * math.sin(x) / x) if x else 0.0
    cs = np.linspace(0.05, 0.95, 181)
    m = np.array([2 * amp(2, c) * amp(3, 0.5) / ((1 - c) * 0.5) for c in cs])
    W, H, L, B = 720, 300, 60, 40
    im = Image.new("RGB", (W, H + 70), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([L, 10, W - 20, H - B], outline=(58, 68, 73))
    top = 12.0
    for y_pct in (0, 2, 4, 6, 8, 10, 12):
        y = H - B - (y_pct / top) * (H - B - 10)
        d.line([L, y, W - 20, y], fill=(40, 48, 52))
        d.text((8, y - 7), f"{y_pct:2d}%", fill=DIM, font=font(11))
    pts = [(L + (c - 0.05) / 0.90 * (W - 20 - L), H - B - min(m_, top) / top * (H - B - 10))
           for c, m_ in zip(cs, m)]
    d.line(pts, fill=(224, 138, 114), width=3)
    d.line([L + (0.45) / 0.90 * (W - 20 - L), 10, L + 0.45 / 0.90 * (W - 20 - L), H - B],
           fill=(79, 185, 203), width=1)
    thr = H - B - 0.5 / top * (H - B - 10)
    d.line([L, thr, W - 20, thr], fill=(95, 185, 133), width=1)
    d.text((W - 210, thr - 14), "~0.5% detection threshold at 9 c/deg", fill=(95, 185, 133), font=font(10))
    for c in (0.1, 0.3, 0.5, 0.7, 0.9):
        x = L + (c - 0.05) / 0.90 * (W - 20 - L)
        d.text((x - 10, H - B + 6), f"{c:.1f}", fill=DIM, font=font(11))
    d.text((L, H - B + 22), "local duty of the 44 um screen (= local tone of the picture)", fill=DIM, font=font(11))
    d.text((L + 0.45 / 0.90 * (W - 20 - L) + 6, 14), "zero only at exactly 0.50", fill=(79, 185, 203), font=font(11))
    d.text((8, H + 4), "(2,3) beat, 559 um = 9.4 cycles/deg, screen over a 50% carrier: Michelson contrast vs the screen's local duty",
           fill=FG, font=font(12))
    d.text((8, H + 22), "A halftone spans this whole axis by design, so the beat is a tone-dependent banding across the picture;",
           fill=DIM, font=font(10))
    d.text((8, H + 36), "process bias moves the null, it does not create the effect. B-HARM measures the curve at 0.42 / 0.50 / 0.58.",
           fill=DIM, font=font(10))
    im.save(out)
    print("  saved", out, im.size)


SVG_STACK = """<svg viewBox="0 0 760 300" xmlns="http://www.w3.org/2000/svg" font-family="IBM Plex Mono, monospace" font-size="12">
  <defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <g stroke="currentColor" fill="none" stroke-width="1.5">
    <rect x="80" y="70" width="520" height="14" fill="currentColor" fill-opacity=".08"/>
    <rect x="80" y="200" width="520" height="14" fill="currentColor" fill-opacity=".08"/>
    <rect x="80" y="84" width="520" height="116" fill="currentColor" fill-opacity=".03" stroke-dasharray="4 3"/>
  </g>
  <g fill="currentColor">
    <rect x="120" y="70" width="16" height="14"/><rect x="176" y="70" width="16" height="14"/><rect x="232" y="70" width="16" height="14"/><rect x="288" y="70" width="16" height="14"/><rect x="344" y="70" width="16" height="14"/><rect x="400" y="70" width="16" height="14"/><rect x="456" y="70" width="16" height="14"/><rect x="512" y="70" width="16" height="14"/>
    <rect x="120" y="200" width="16" height="14"/><rect x="176" y="200" width="16" height="14"/><rect x="232" y="200" width="16" height="14"/><rect x="288" y="200" width="16" height="14"/><rect x="344" y="200" width="16" height="14"/><rect x="400" y="200" width="16" height="14"/><rect x="456" y="200" width="16" height="14"/><rect x="512" y="200" width="16" height="14"/>
  </g>
  <g stroke="currentColor" stroke-width="1.5" fill="none" marker-end="url(#a)" color="currentColor">
    <path d="M 316 20 L 316 70"/>
    <path d="M 316 84 L 340 200" stroke-dasharray="5 3"/>
  </g>
  <g stroke="currentColor" stroke-width="1" fill="none" opacity=".6">
    <path d="M 316 84 L 316 200" stroke-dasharray="2 4"/>
    <path d="M 316 226 L 340 226" marker-end="url(#a)"/>
  </g>
  <g fill="currentColor">
    <text x="620" y="82">front ply — comb / grating</text>
    <text x="620" y="212">back ply — lanes / carrier</text>
    <text x="620" y="146">gap t, index n</text>
    <text x="250" y="40">tilt θ (in air)</text>
    <text x="330" y="150">θ′ = asin(sin θ / n)</text>
    <text x="352" y="246">shift = t · tan θ′</text>
    <text x="80" y="285" opacity=".75">17.22 µm/° at 1.5 mm soda lime (the box) · 27.40 µm/° at 2.29 mm quartz (this plate) — the two plies do not move; the line of sight does.</text>
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
    <text x="30" y="120">clear where NEITHER has metal: 1 − 1<tspan font-size="9" dy="3">A₁∪A₂</tspan></text>
    <text x="410" y="120">light passes where BOTH are clear: (1 − 1<tspan font-size="9" dy="3">A₁</tspan><tspan dy="-3">)(1 − 1</tspan><tspan font-size="9" dy="3">A₂</tspan><tspan dy="-3">)</tspan></text>
    <text x="30" y="150">inclusion–exclusion: 1 − A₁ − A₂ + A₁A₂  — the A₁A₂ term IS the moiré, and it is identical in both.</text>
    <text x="30" y="176">Assumptions that make the identity optical: binary amplitude masks · geometric optics (no diffraction between the planes)</text>
    <text x="30" y="194">· zero gap, so the two lattices have a fixed relative phase. Open a gap and the phase rides on the line of sight (parallax);</text>
    <text x="30" y="212">open it further than the near-field limit and the product itself stops being a product.</text>
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
    <text x="380" y="62">geometry: CLEAR regions</text>
    <text x="380" y="84">— chrome comes OFF here</text>
    <text x="380" y="180">label: gold, bottom-left,</text>
    <text x="380" y="198">carries the parameter VALUE</text>
    <text x="60" y="210" opacity=".7">label band scales with the cell (0.4–0.9 mm); rows are packed by height, so a raster scan finds every label in the same corner</text>
  </g>
</svg>"""


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    swatch_png(out / "witness_swatch.png")
    harmonic_chart_png(out / "witness_harmonic.png")
    (out / "fig_stack.svg").write_text(SVG_STACK, encoding="utf-8")
    (out / "fig_union.svg").write_text(SVG_UNION, encoding="utf-8")
    (out / "fig_cell.svg").write_text(SVG_CELL, encoding="utf-8")
    print("  saved 3 SVG diagrams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
