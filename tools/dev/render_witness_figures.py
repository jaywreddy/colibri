"""Figures for the whitepaper that a formula alone does not carry.

Diagrams are inline SVG so they render in both themes. Simulated panels use the
same appearance model as the box renderer — specular gold (eta0 = 0.25) plus
the first-order sheen (eta1 = 0.10) from the CIE table — because a render that
drops the specular shows black wherever the first order leaves the visible band.

Single-layer only (2026-09-15): the two-ply figures (screen-over-carrier
harmonic chart, P-SWAP barrier-comb plot, the front/back-ply stack diagram,
the zero-gap union-identity diagram) rendered cells retired with the bond —
see docs/archived/witness-physics-plan.md secs 1.2/1.3/2.1/2.4/4.3/4.4 for
the record. What is left: the colour-ladder swatch (4.2, still the box's own
diffraction-colour physics), the H-WEDGE tone linearisation figure (4.5, a
surviving bench cell), and the generic cell-anatomy diagram (4.6).

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
# SVG_STACK (front/back-ply parallax stack) and SVG_UNION (zero-gap union
# identity) rendered the two-ply mechanics; retired with the bond. Their
# markup is in git history (see docs/archived/witness-physics-plan.md secs
# 1.3 / 2.1 for the prose). SVG_CELL is generic — any bench cell, not just
# retired ones — so it stays.
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
    wedge_png(out / "witness_wedge.png")
    (out / "fig_cell.svg").write_text(SVG_CELL, encoding="utf-8")
    print("  saved 1 SVG diagram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
